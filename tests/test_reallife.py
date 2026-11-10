from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ricochet.config import AppConfig
from ricochet.reallife import (
    CausalConfig,
    ClusterConfig,
    IncidentBundleIn,
    RealtimeAlarm,
    RunbookAgent,
    RunbookConfig,
    build_causal_graph,
    cluster_series,
    prepare_incident,
)
from ricochet.precheck.runbooks import RunbookRegistry
from ricochet.ranking import PairRanker, RankerConfig


def _bundle() -> IncidentBundleIn:
    start = datetime(2026, 1, 1, 12, 0)
    alarms = []
    for k in range(60):
        t = start + timedelta(seconds=20 * k)
        for offset, system, alert_group, node in [
            (0, "snmp", "A", "r1.msk"),
            (20, "syslog", "B", "r2.msk"),
            (40, "bgp", "C", "r3.msk"),
        ]:
            alarms.append(RealtimeAlarm(
                ts=t + timedelta(seconds=offset), integration="ipcore",
                system=system, alert_group=alert_group,
                node=node, node_alias=node.split(".")[0], region="MSK", severity=4))
    for k in range(0, 60, 7):
        alarms.append(RealtimeAlarm(
            ts=start + timedelta(seconds=20 * k + 7), integration="dcim",
            system="power", alert_group="D", node="p1.spb", node_alias="p1",
            region="SPB", severity=3))
    return IncidentBundleIn(incident_id="INC-RT-1", region="MSK", alarms=alarms)


def _causal_config(**overrides):
    params = dict(max_lag_seconds=120, min_abs_corr=0.2, min_causal_power=0.2,
                  max_cond_size=0, remove_common_factor=False, use_topology=False)
    params.update(overrides)
    return CausalConfig(**params)


def test_intensity_and_ewma_pipeline():
    prepared = prepare_incident(_bundle(), step_seconds=10, ewma_half_life_seconds=30)
    assert prepared.keys
    assert prepared.intensity.shape[0] == len(prepared.grid)
    assert prepared.intensity.shape[1] == len(prepared.keys)
    assert prepared.intensity.to_numpy().sum() > 0
    assert prepared.ewma.to_numpy().max() <= prepared.intensity.to_numpy().max() + 1e-9
    assert all(key in prepared.meta for key in prepared.keys)


def test_causal_graph_is_dag_and_finds_lead():
    prepared = prepare_incident(_bundle(), step_seconds=10, ewma_half_life_seconds=30)
    graph = build_causal_graph(prepared, _causal_config())
    ids = {node["id"] for node in graph["nodes"]}
    assert "ipcore|snmp|A" in ids
    assert "ipcore|syslog|B" in ids
    assert graph["edges"]
    assert len(graph["topological_order"]) == len(graph["nodes"])
    assert graph["method"] == "lagged_pc"
    # All oriented edges must respect the topological order.
    pos = {node_id: i for i, node_id in enumerate(graph["topological_order"])}
    for edge in graph["edges"]:
        assert pos[edge["from"]] <= pos[edge["to"]]
        assert edge["lag_seconds"] > 0


def test_cluster_micro_storms():
    prepared = prepare_incident(_bundle(), step_seconds=10, ewma_half_life_seconds=30)
    result = cluster_series(prepared, ClusterConfig(method="kmeans", n_clusters=2, top_k=10))
    assert result["series_total"] == len(prepared.keys)
    assert result["clusters"]
    assert result["clusters"][0]["size"] >= 1
    assert result["window"]["step_seconds"] == 10
    assert all("anomaly_score" in cluster for cluster in result["clusters"])


def test_runbook_uses_ranker_and_registry():
    prepared = prepare_incident(_bundle(), step_seconds=10, ewma_half_life_seconds=30)
    graph = build_causal_graph(prepared, _causal_config())
    clusters = cluster_series(prepared, ClusterConfig(method="kmeans", n_clusters=2))
    ranker = PairRanker(RankerConfig(state_dir=Path(__import__("tempfile").mkdtemp()),
                                     auto_train=False))
    registry = RunbookRegistry([
        dict(id="rb-A", system="snmp", alert_group="A", steps=["check A path"], owner="ipcore"),
        dict(id="rb-B", system="syslog", alert_group="B", steps=["check B path"], owner="ipcore"),
    ])
    result = RunbookAgent().generate(
        prepared, graph, clusters, ranker=ranker, runbooks=registry,
        config=RunbookConfig(top_k=3))
    assert result["root_causes"]
    assert result["summary"]
    assert result["confidence"] > 0
    assert any(step["source"].startswith("runbook:") for step in result["steps"])
    assert result["evidence"]["ranker"]["scored"] > 0


def test_realtime_api_endpoints(tmp_path):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    from ricochet.api.main import create_app

    data = tmp_path / "data"
    cfg = AppConfig(
        data_dir=data, raw_dir=data / "raw", processed_dir=data / "processed",
        state_dir=tmp_path / "state", alarms_path=data / "processed" / "alarms.parquet",
        use_synthetic_when_empty=False,
    )
    body = _bundle().model_dump(mode="json")
    with TestClient(create_app(cfg)) as client:
        r = client.post("/api/v1/topology/causal-graph",
                        params=dict(max_cond_size=0, remove_common_factor=False),
                        json=body)
        assert r.status_code == 200, r.text
        assert r.json()["edges"]

        r = client.post("/api/v1/anomalies/clusters",
                        params=dict(method="kmeans", n_clusters=2), json=body)
        assert r.status_code == 200, r.text
        assert r.json()["clusters"]

        r = client.post("/api/v1/agent/runbook",
                        params=dict(max_cond_size=0, top_k=3), json=body)
        assert r.status_code == 200, r.text
        runbook = r.json()
        assert runbook["summary"]
        assert runbook["steps"]

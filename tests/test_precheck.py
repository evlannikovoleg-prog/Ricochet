from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ricochet.config import EngineConfig
from ricochet.canonical import validate_alarms
from ricochet.ingest.synthetic import generate_synthetic
from ricochet.leadlag import run_butterfly
from ricochet.verdicts import finalize_pairs
from ricochet.maintenance import MaintenanceStore, MaintenanceWindow
from ricochet.precheck.similar import HistoricalIncidents
from ricochet.precheck.runbooks import RunbookRegistry
from ricochet.precheck.context import assemble_case


@pytest.fixture(scope="module")
def world():
    df, meta = generate_synthetic(days=20, seed=7)
    df = validate_alarms(df)
    cfg = EngineConfig(min_events=100, min_bins=30, min_support=10, min_abs_corr=0.3,
                       lags=6, min_devices=3, hub_degree=40)
    res = run_butterfly(df, cfg, holdout_last=0)
    pairs = finalize_pairs(df, res["pairs"], cfg)
    store = MaintenanceStore(None)
    w = meta["maintenance_window"]
    store.add(MaintenanceWindow(start=w["start"], end=w["end"], region=w.get("region", "")))
    incidents = HistoricalIncidents(meta["incidents"])
    runbooks = RunbookRegistry([
        dict(id="rb-mpls-001", system="snmp", alert_group="mplsTunnelRerouted",
             url="http://runbooks.local/mpls-reroute", steps=["check LSP path"], owner="ipcore"),
        dict(id="rb-nokia-dry", system="nokia", alert_group="dryContactAlarm",
             url="http://runbooks.local/dry-contact", steps=["verify contact"], owner="transport"),
    ])
    return df, meta, pairs, store, incidents, runbooks


def _follower_alarms(meta, n=6):
    ka, kb = meta["butterfly"]["key_a"], meta["butterfly"]["key_b"]
    base = datetime(2026, 1, 10, 12, 0)
    out = []
    for i in range(n):
        out.append(dict(
            ts=base + timedelta(minutes=5 * i),
            integration=kb[0], system=kb[1], alert_group=kb[2],
            node=f"jmx{i:03d}.msk", node_alias="", region="MSK", severity=4))
    return out


def test_root_cause_candidate_is_leader(world):
    df, meta, pairs, store, incidents, runbooks = world
    incident = dict(incident_id="INC-TEST-1", alarms=_follower_alarms(meta))
    case = assemble_case(incident, alarms_history=df, pairs=pairs,
                         maintenance=store, incidents=incidents, runbooks=runbooks)
    assert case["root_cause_candidates"]
    top = case["root_cause_candidates"][0]
    assert top["system"] == meta["butterfly"]["key_a"][1]
    assert top["alert_group"] == meta["butterfly"]["key_a"][2]
    assert top["in_incident"] is False
    assert "BUTTERFLY_LINKED" in [f["code"] for f in case["risk"]["flags"]]


def test_maintenance_flag(world):
    df, meta, pairs, store, incidents, runbooks = world
    w = meta["maintenance_window"]
    wstart = datetime.fromisoformat(w["start"])
    alarms = [dict(ts=wstart + timedelta(minutes=5),
                   integration="oss", system="backup", alert_group="dbBackupStalled",
                   node="db-core-01", node_alias="", region="MSK", severity=3)]
    case = assemble_case(dict(incident_id="INC-TEST-2", alarms=alarms),
                         alarms_history=df, pairs=pairs, maintenance=store,
                         incidents=incidents, runbooks=runbooks)
    assert case["maintenance"]["hit"] is True
    codes = [f["code"] for f in case["risk"]["flags"]]
    assert "MAINTENANCE_OVERLAP" in codes
    assert case["alarms_outside_maintenance"] == 0


def test_similar_incidents_found(world):
    df, meta, pairs, store, incidents, runbooks = world
    alarms = _follower_alarms(meta, n=3)
    alarms += [dict(ts=datetime(2026, 1, 10, 12, 30), integration=meta["butterfly"]["key_a"][0],
                    system=meta["butterfly"]["key_a"][1], alert_group=meta["butterfly"]["key_a"][2],
                    node="ape000.msk", node_alias="", region="MSK", severity=4)]
    case = assemble_case(dict(incident_id="INC-TEST-3", alarms=alarms),
                         alarms_history=df, pairs=pairs, maintenance=store,
                         incidents=incidents, runbooks=runbooks)
    assert case["similar_incidents"]
    assert case["similar_incidents"][0]["similarity"] > 0.3


def test_runbook_match_and_risk(world):
    df, meta, pairs, store, incidents, runbooks = world
    alarms = [dict(ts=datetime(2026, 1, 10, 12, 0),
                   integration="ipcore", system="snmp", alert_group="mplsTunnelRerouted",
                   node="ape001.msk", node_alias="", region="MSK", severity=5)]
    case = assemble_case(dict(incident_id="INC-TEST-4", alarms=alarms),
                         alarms_history=df, pairs=pairs, maintenance=store,
                         incidents=incidents, runbooks=runbooks)
    assert case["runbook"] is not None
    assert case["runbook"]["id"] == "rb-mpls-001"
    assert case["blast"]["max_severity"] == 5
    assert "HIGH_SEVERITY" in [f["code"] for f in case["risk"]["flags"]]
    assert case["risk"]["level"] in ("MEDIUM", "HIGH")


def test_enrichment_completeness_low(world):
    df, meta, pairs, store, incidents, runbooks = world
    alarms = [dict(ts=datetime(2026, 1, 10, 12, 0), integration="", system="sys",
                    alert_group="ag", node="", node_alias="", region="", severity=4)]
    case = assemble_case(dict(incident_id="INC-TEST-5", alarms=alarms),
                         alarms_history=df, pairs=pairs, maintenance=store,
                         incidents=incidents, runbooks=runbooks)
    assert case["enrichment"]["completeness"] < 0.6
    assert "LOW_ENRICHMENT" in [f["code"] for f in case["risk"]["flags"]]

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

fastapi_tc = pytest.importorskip("fastapi.testclient")

from ricochet.config import AppConfig
from ricochet.api.main import create_app


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp(prefix="ricochet-test-")
    root = Path(tmp)
    data = root / "data"
    cfg = AppConfig(
        data_dir=data,
        raw_dir=data / "raw",
        processed_dir=data / "processed",
        state_dir=root / "state",
        alarms_path=data / "processed" / "alarms.parquet",
        synthetic_days=25,
        synthetic_seed=11,
    )
    with fastapi_tc.TestClient(create_app(cfg)) as c:
        yield c


def _wait_job(client, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/butterfly/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(1)
    raise TimeoutError(f"job {job_id} not finished")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["synthetic"] is True
    assert body["alarms"] > 1000


def test_butterfly_run_and_pairs(client):
    r = client.post("/api/v1/butterfly/run", json={})
    assert r.status_code == 200
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job.get("error")

    r = client.get("/api/v1/butterfly/pairs", params={"cls": "BUTTERFLY_CANDIDATE"})
    assert r.status_code == 200
    pairs = r.json()["pairs"]
    assert pairs, "no butterfly candidates found"
    assert any(p["key_a"][2] == "mplsTunnelRerouted" or p["key_b"][2] == "mplsTunnelRerouted"
               for p in pairs)

    r = client.get("/api/v1/butterfly/graph")
    g = r.json()
    assert g["nodes"] and g["edges"]

    r = client.get("/api/v1/butterfly/pairs", params={"cls": "DUPLICATE"})
    dup = r.json()["pairs"]
    assert any({p["key_a"][2], p["key_b"][2]} == {"dryContactAlarm", "externalAlarm"}
               for p in dup)


def test_precheck_incident(client):
    alarms = [
        dict(ts=(datetime(2026, 1, 10, 12, 0) + timedelta(minutes=5 * i)).isoformat(),
             integration="ipcore", system="syslog", alert_group="RPD-ISIS-ADJ",
             node=f"jmx{i:03d}.msk", node_alias="", region="MSK", severity=4)
        for i in range(4)
    ]
    r = client.post("/api/v1/precheck/incident", json=dict(
        incident_id="INC-API-1", region="MSK", alarms=alarms))
    assert r.status_code == 200
    case = r.json()
    assert case["root_cause_candidates"]
    top = case["root_cause_candidates"][0]
    assert top["system"] == "snmp" and top["alert_group"] == "mplsTunnelRerouted"
    assert case["similar_incidents"]
    assert case["risk"]["level"] in ("LOW", "MEDIUM", "HIGH")


def test_maintenance_windows_crud(client):
    r = client.get("/api/v1/maintenance/windows")
    assert r.status_code == 200
    windows = r.json()["windows"]
    assert len(windows) >= 1

    r = client.post("/api/v1/maintenance/windows", json=dict(
        start="2026-02-01T00:00:00", end="2026-02-01T02:00:00",
        region="SPB", source="test", note="planned work"))
    assert r.status_code == 200
    wid = r.json()["id"]

    r = client.delete(f"/api/v1/maintenance/windows/{wid}")
    assert r.status_code == 200


def test_ranker_feedback_and_training(client):
    pytest.importorskip("xgboost")
    r = client.get("/api/v1/ranker/status")
    assert r.status_code == 200
    assert r.json()["feature_count"] > 0

    pairs = client.get("/api/v1/butterfly/pairs").json()["pairs"]
    assert pairs
    assert all("xgboost" in p for p in pairs)
    good = next(p for p in pairs if p.get("class") == "BUTTERFLY_CANDIDATE")
    spam = next(p for p in pairs if p.get("class") == "DUPLICATE")

    for pair, label in ((good, "good"), (spam, "spam")):
        r = client.post("/api/v1/ranker/feedback", json=dict(
            pair_id=pair["id"], label=label, source="test"))
        assert r.status_code == 200

    r = client.post("/api/v1/ranker/train", json=dict(force=True))
    assert r.status_code == 200
    assert r.json()["result"]["trained"] is True
    assert r.json()["status"]["model_kind"] == "xgboost"

    r = client.post("/api/v1/precheck/incident", json=dict(
        incident_id="INC-RANKER", region="MSK", alarms=[]))
    assert r.status_code == 200
    assert "ranker" in r.json()


def test_verify_duplicate_pair(client):
    r = client.post("/api/v1/butterfly/verify", json=dict(
        key_a=dict(integration="nsp", system="nokia", alert_group="dryContactAlarm"),
        key_b=dict(integration="transport", system="alu", alert_group="externalAlarm"),
        lag_bins=0))
    assert r.status_code == 200
    body = r.json()
    assert body["overlap"]["verdict"] in ("SAME_DEVICE", "PARTIAL_OVERLAP")
    assert body["shared_unshared"]["verdict"] == "DUPLICATE"

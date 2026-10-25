from __future__ import annotations

import http.client
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://127.0.0.1:8077"
HOST = "127.0.0.1"
PORT = 8077
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "api_test_results.json"

KNOWN_ALARM = {
    "ts": "2020-04-24T10:15:00",
    "integration": "platform",
    "system": "os_linux",
    "alert_group": "Sent_packets_high",
    "node": "os_linux_01",
    "region": "MSK",
    "severity": 4,
}

UNKNOWN_ALARM = {
    "ts": "2020-04-24T10:20:00",
    "integration": "business",
    "system": "esb",
    "alert_group": "CSF_fail",
    "node": "esb_01",
    "node_alias": "esb-alias",
    "region": "SPB",
    "severity": 3,
}


def call(method: str, path: str, body: dict | list | None = None) -> tuple[int, dict, float]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload, (time.perf_counter() - started) * 1000


def slim(case: dict) -> dict:
    candidates = case.get("root_cause_candidates", [])
    return {
        "incident_id": case.get("incident_id", ""),
        "alarms_total": case.get("alarms_total", 0),
        "alarms_outside_maintenance": case.get("alarms_outside_maintenance", 0),
        "maintenance_hit": case.get("maintenance", {}).get("hit", False),
        "enrichment": case.get("enrichment", {}).get("completeness"),
        "runbook": None if case.get("runbook") is None else case["runbook"].get("id"),
        "blast": case.get("blast", {}),
        "risk": case.get("risk", {}),
        "candidates": len(candidates),
        "top_candidates": [
            {
                "system": item.get("system"),
                "alert_group": item.get("alert_group"),
                "score": item.get("score"),
                "predicts_downstream": item.get("predicts_downstream"),
            }
            for item in candidates[:3]
        ],
        "similar": len(case.get("similar_incidents", [])),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def stats(values: list[float], wall_seconds: float | None = None) -> dict:
    result = {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "p99_ms": round(percentile(values, 0.99), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }
    if wall_seconds:
        result["rps"] = round(len(values) / wall_seconds, 1)
        result["wall_seconds"] = round(wall_seconds, 3)
    return result


def http_post(conn: http.client.HTTPConnection, payload: dict) -> float:
    body = json.dumps(payload, ensure_ascii=False)
    started = time.perf_counter()
    conn.request(
        "POST",
        "/api/v1/precheck/incident",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    response.read()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}")
    return (time.perf_counter() - started) * 1000


def bench_keepalive(n: int, workers: int, payload: dict) -> dict:
    def worker(chunk: int) -> list[float]:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
        try:
            return [http_post(conn, payload) for _ in range(chunk)]
        finally:
            conn.close()

    for _ in range(10):
        conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
        try:
            http_post(conn, payload)
        finally:
            conn.close()

    started = time.perf_counter()
    if workers == 1:
        values = worker(n)
    else:
        base, extra = divmod(n, workers)
        chunks = [base + (1 if i < extra else 0) for i in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = [item for chunk in executor.map(worker, chunks) for item in chunk]
    elapsed = time.perf_counter() - started
    return stats(values, elapsed)


def bench_cold(n: int, payload: dict) -> dict:
    values = []
    started = time.perf_counter()
    for _ in range(n):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            BASE + "/api/v1/precheck/incident",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        tick = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as response:
            response.read()
        values.append((time.perf_counter() - tick) * 1000)
    elapsed = time.perf_counter() - started
    return stats(values, elapsed)


def bench_scaling() -> list[dict]:
    rows = []
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    try:
        for n in (1, 10, 50, 100):
            payload = {
                "incident_id": f"SCALE-{n}",
                "alarms": [
                    {
                        "ts": f"2020-04-24T10:{i % 60:02d}:00",
                        "integration": "platform",
                        "system": "os_linux",
                        "alert_group": "Sent_packets_high",
                        "node": f"os_linux_{i % 20:02d}",
                        "region": ["MSK", "SPB", "KZN"][i % 3],
                        "severity": 4,
                    }
                    for i in range(n)
                ],
            }
            for _ in range(3):
                http_post(conn, payload)
            values = [http_post(conn, payload) for _ in range(10)]
            rows.append({"alarms_per_request": n, **stats(values)})
    finally:
        conn.close()
    return rows


def main() -> None:
    results: dict = {
        "meta": {
            "base_url": BASE,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "method": "http.client keep-alive; warm-up excluded; one local uvicorn worker",
        },
        "health": {},
        "scenarios": {},
        "benchmark": {},
    }

    status, health, _ = call("GET", "/health")
    results["health"] = {"status_code": status, **health}

    status, case, latency_ms = call("POST", "/api/v1/precheck/alarm", KNOWN_ALARM)
    results["scenarios"]["S1_known_platform_alarm"] = {
        "request": KNOWN_ALARM,
        "status_code": status,
        "latency_ms": round(latency_ms, 2),
        "response": slim(case),
    }

    blast_alarms = [
        {
            "ts": f"2020-04-24T10:{i:02d}:00",
            "integration": "platform",
            "system": "os_linux",
            "alert_group": "Sent_packets_high",
            "node": f"os_linux_{i:02d}",
            "region": ["MSK", "SPB", "KZN"][i % 3],
            "severity": 4,
        }
        for i in range(10)
    ]
    status, case, latency_ms = call(
        "POST",
        "/api/v1/precheck/incident",
        {"incident_id": "BLAST-10", "region": "MSK", "alarms": blast_alarms},
    )
    results["scenarios"]["S2_blast_radius"] = {
        "request_alarms": len(blast_alarms),
        "status_code": status,
        "latency_ms": round(latency_ms, 2),
        "response": slim(case),
    }

    status, case, latency_ms = call(
        "POST",
        "/api/v1/precheck/alarm",
        UNKNOWN_ALARM,
    )
    results["scenarios"]["S3_unknown_business_alarm"] = {
        "request": UNKNOWN_ALARM,
        "status_code": status,
        "latency_ms": round(latency_ms, 2),
        "response": slim(case),
    }

    window_id = None
    try:
        status, window, _ = call(
            "POST",
            "/api/v1/maintenance/windows",
            {
                "start": "2020-04-24T09:00:00",
                "end": "2020-04-24T11:00:00",
                "region": "MSK",
                "system": "os_linux",
                "node": "os_linux_05",
                "note": "api test scenario",
            },
        )
        window_id = window.get("id")
        alarm = dict(KNOWN_ALARM, node="os_linux_05")
        status, case, latency_ms = call("POST", "/api/v1/precheck/alarm", alarm)
        results["scenarios"]["S4_maintenance_overlap"] = {
            "window_id": window_id,
            "status_code": status,
            "latency_ms": round(latency_ms, 2),
            "response": slim(case),
        }
    finally:
        if window_id:
            call("DELETE", f"/api/v1/maintenance/windows/{window_id}")

    _, case_a, _ = call("POST", "/api/v1/precheck/alarm", KNOWN_ALARM)
    _, case_b, _ = call("POST", "/api/v1/precheck/alarm", KNOWN_ALARM)
    results["scenarios"]["S5_replay"] = {
        "identical": case_a == case_b,
        "risk_score": case_a.get("risk", {}).get("score"),
    }

    status, error, _ = call(
        "POST",
        "/api/v1/precheck/alarm",
        {"integration": "platform", "system": "os_linux", "alert_group": "X"},
    )
    results["scenarios"]["S6_validation_error"] = {
        "status_code": status,
        "detail_type": error.get("detail", [{}])[0].get("type")
        if isinstance(error.get("detail"), list)
        else None,
    }

    status, error, _ = call("GET", "/api/v1/butterfly/jobs/does-not-exist")
    results["scenarios"]["S7_not_found"] = {
        "status_code": status,
        "detail": error.get("detail"),
    }

    status, pairs, _ = call("GET", "/api/v1/butterfly/pairs?limit=1")
    pair = pairs.get("pairs", [{}])[0] if pairs.get("pairs") else None
    verify = None
    if pair:
        _, verify, _ = call(
            "POST",
            "/api/v1/butterfly/verify",
            {
                "key_a": dict(zip(("integration", "system", "alert_group"), pair["key_a"])),
                "key_b": dict(zip(("integration", "system", "alert_group"), pair["key_b"])),
                "lag_bins": pair.get("lag_bins", 0),
            },
        )
    _, graph, _ = call("GET", "/api/v1/butterfly/graph")
    results["scenarios"]["S8_graph_and_verify"] = {
        "pairs_total": pairs.get("total"),
        "pair": None
        if pair is None
        else {
            "key_a": pair["key_a"],
            "key_b": pair["key_b"],
            "class": pair.get("class"),
        },
        "verify_overlap": None if verify is None else verify.get("overlap"),
        "verify_shared_unshared": None if verify is None else verify.get("shared_unshared"),
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
    }

    single_payload = {"incident_id": "BENCH-1", "alarms": [KNOWN_ALARM]}
    results["benchmark"]["sequential_100"] = bench_keepalive(100, 1, single_payload)
    results["benchmark"]["concurrent_8x100"] = bench_keepalive(100, 8, single_payload)
    results["benchmark"]["cold_connection_100"] = bench_cold(100, single_payload)
    results["benchmark"]["payload_scaling"] = bench_scaling()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

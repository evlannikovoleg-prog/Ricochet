from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..canonical import COLUMNS

BIN_MIN = 5
BUTTERFLY_LAG_BINS = 2
BUTTERFLY_SYSTEMS = ("snmp", "syslog")


def _devices(prefix: str, n: int, regions: list[str], rng) -> list[tuple[str, str]]:
    out = []
    for i in range(n):
        reg = regions[i % len(regions)]
        out.append((f"{prefix}{i:03d}.{reg.lower()}", f"{prefix}{i:03d}"))
    return out


def generate_synthetic(days: int = 45, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n_bins = days * 24 * 60 // BIN_MIN
    t0 = datetime(2026, 1, 1)
    bin_start = [t0 + timedelta(minutes=BIN_MIN * i) for i in range(n_bins)]
    regions = ["MSK", "SPB", "EKB", "NSK"]
    bf_regions = ["MSK", "SPB", "EKB"]
    events: list[tuple] = []

    def emit(bin_idx: int, integration: str, system: str, ag: str,
             devs: list[tuple[str, str]], dev_regions: list[str], region: str, n: int, sev: int = 4):
        if n <= 0 or bin_idx >= n_bins:
            return
        base = bin_start[bin_idx]
        for _ in range(int(n)):
            d, a = devs[rng.integers(0, len(devs))]
            ts = base + timedelta(seconds=float(rng.uniform(0, BIN_MIN * 60)))
            events.append((ts, integration, system, ag, d, a, region, sev))

    dev_a = _devices("ape", 22, bf_regions, rng)
    dev_a_regions = [d[0].split(".")[-1].upper() for d in dev_a]
    dev_b = _devices("jmx", 176, bf_regions, rng)

    background = [
        ("nms-oss", "healthcheck", "nodeUnreachable"),
        ("nms-oss", "healthcheck", "agentDown"),
        ("ipcore", "bgp", "bgpNeighborDown"),
        ("ipcore", "ospf", "ospfNeighborDown"),
        ("transport", "sdh", "losAlarm"),
        ("transport", "sdh", "clockSourceFail"),
        ("ran", "lte", "cellOutage"),
        ("ran", "lte", "rfUnitFail"),
        ("dcim", "power", "rectifierAlarm"),
        ("dcim", "climate", "tempHigh"),
        ("security", "firewall", "sessionLimitHit"),
        ("cloud", "k8s", "podRestart"),
    ]
    bg_devices = {}
    for integ, system, ag in background:
        bg_devices[(integ, system, ag)] = _devices(
            f"bg{system[:3]}{ag[:3]}", rng.integers(30, 80), regions, rng)

    g = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_bins) / 288 - 1.2)
    for i, (integ, system, ag) in enumerate(background):
        devs = bg_devices[(integ, system, ag)]
        lam = 0.5 + 2.5 * g * (0.5 + (i % 5) / 5)
        for b in range(n_bins):
            n = rng.poisson(lam[b])
            if n:
                reg = regions[rng.integers(0, 4)]
                pool = [(d, a) for (d, a) in devs if d.split(".")[-1].upper() == reg]
                if pool:
                    emit(b, integ, system, ag, pool, None, reg, n)

    storms = np.sort(rng.integers(60, n_bins - 60, size=days * 5))
    for s in storms:
        reg = bf_regions[rng.integers(0, 3)]
        pool_a = [da for da, dr in zip(dev_a, dev_a_regions) if dr == reg]
        amp = rng.integers(30, 80)
        for k, w in enumerate([1.0, 0.4, 0.1]):
            emit(int(s) + k, "ipcore", BUTTERFLY_SYSTEMS[0], "mplsTunnelRerouted",
                 pool_a, None, reg, amp * w)
        pool_b = [(d, a) for d, a in dev_b if d.split(".")[-1].upper() == reg]
        amp_b = rng.integers(25, 70)
        for k, w in enumerate([1.0, 0.5, 0.2]):
            emit(int(s) + BUTTERFLY_LAG_BINS + k, "ipcore", BUTTERFLY_SYSTEMS[1], "RPD-ISIS-ADJ",
                 pool_b, None, reg, amp_b * w)

    dev_dup = _devices("dup", 150, regions, rng)
    for b in range(0, n_bins, 12):
        if rng.random() < 0.85:
            n = rng.integers(15, 40)
            for k, w in enumerate([1.0, 0.5]):
                emit(b + k, "nsp", "nokia", "dryContactAlarm", dev_dup, None, "", n * w)
                emit(b + k, "transport", "alu", "externalAlarm", dev_dup, None, "", n * w)

    dev_p = _devices("pwr", 60, regions, rng)
    dev_q = _devices("com", 60, regions, rng)
    for b in range(n_bins):
        level = max(0.0, g[b] - 0.62) / 0.38
        if level > 0:
            n = rng.poisson(12 * level)
            if n:
                emit(b, "dcim", "grid", "voltageDrop", dev_p, None, "", n)
            n = rng.poisson(10 * level)
            if n:
                emit(b, "transport", "link", "linkFlap", dev_q, None, "", n)

    for b in range(n_bins):
        emit(b, "oss", "watchdog", "moiraHeartbeat", [("mon-heartbeat", "")], None, "MSK",
             int(rng.integers(1, 3)), sev=2)

    for s in storms[::2]:
        amp = rng.integers(30, 90)
        emit(int(s), "ran", "enm", "routingTableFail", [("single-ericsson", "single-ericsson")],
             None, "MSK", amp, sev=3)

    maint_bin = n_bins // 2
    for b in range(maint_bin, maint_bin + 36):
        emit(b, "oss", "backup", "dbBackupStalled", [("db-core-01", "")], None, "MSK",
             int(rng.integers(10, 40)), sev=3)
    maintenance_window = dict(
        start=bin_start[maint_bin].isoformat(),
        end=(bin_start[maint_bin + 36]).isoformat(),
        region="MSK",
        system="",
        node="",
        source="synthetic",
        note="planned db maintenance",
    )

    df = pd.DataFrame(events, columns=COLUMNS).sort_values("ts").reset_index(drop=True)

    incidents = []
    for i, s in enumerate(storms):
        if i % 12 == 0:
            reg = bf_regions[i % 3]
            incidents.append(dict(
                id=f"INC-{i:04d}",
                started_at=bin_start[int(s)].isoformat(),
                keys=[
                    {"system": BUTTERFLY_SYSTEMS[0], "alert_group": "mplsTunnelRerouted"},
                    {"system": BUTTERFLY_SYSTEMS[1], "alert_group": "RPD-ISIS-ADJ"},
                ],
                region=reg,
                note="mpls reroute followed by isis adjacency loss",
            ))
    for i in range(8):
        b = (i * 137) % n_bins
        incidents.append(dict(
            id=f"INC-D{i:03d}",
            started_at=bin_start[b].isoformat(),
            keys=[
                {"system": "nokia", "alert_group": "dryContactAlarm"},
                {"system": "alu", "alert_group": "externalAlarm"},
            ],
            region="",
            note="dry contact duplicated by two systems",
        ))

    meta = dict(
        days=days,
        bin_min=BIN_MIN,
        n_events=len(df),
        butterfly=dict(
            key_a=("ipcore", "snmp", "mplsTunnelRerouted"),
            key_b=("ipcore", "syslog", "RPD-ISIS-ADJ"),
            lag_bins=BUTTERFLY_LAG_BINS,
            regions=bf_regions,
        ),
        duplicate=dict(
            key_a=("nsp", "nokia", "dryContactAlarm"),
            key_b=("transport", "alu", "externalAlarm"),
        ),
        common_cause=dict(
            key_a=("dcim", "grid", "voltageDrop"),
            key_b=("transport", "link", "linkFlap"),
        ),
        watchdog=("oss", "watchdog", "moiraHeartbeat"),
        single_device=("ran", "enm", "routingTableFail"),
        maintenance_series=("oss", "backup", "dbBackupStalled"),
        maintenance_window=maintenance_window,
        incidents=incidents,
    )
    return df, meta


def save_synthetic_bundle(out_dir, alarms_path, meta_path, incidents_path, days=45, seed=42):
    from pathlib import Path
    from ..canonical import write_alarms
    df, meta = generate_synthetic(days, seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_alarms(df, alarms_path)
    Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(incidents_path).write_text(
        json.dumps(meta["incidents"], ensure_ascii=False, indent=2), encoding="utf-8")
    return df, meta

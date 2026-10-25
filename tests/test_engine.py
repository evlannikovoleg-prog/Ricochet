from __future__ import annotations

import numpy as np
import pytest

from ricochet.config import EngineConfig
from ricochet.canonical import validate_alarms
from ricochet.ingest.synthetic import generate_synthetic
from ricochet.leadlag import build_bins, detect, run_butterfly, remove_common_factor, transform_matrix, zscore
from ricochet.verdicts import finalize_pairs, overlap_stats
from ricochet.maintenance import MaintenanceStore, MaintenanceWindow


@pytest.fixture(scope="module")
def synthetic():
    df, meta = generate_synthetic(days=20, seed=7)
    return validate_alarms(df), meta


@pytest.fixture(scope="module")
def cfg():
    return EngineConfig(min_events=100, min_bins=30, min_support=10, min_abs_corr=0.3,
                        lags=6, min_devices=3, hub_degree=40)


def test_butterfly_pair_found(synthetic, cfg):
    df, meta = synthetic
    res = run_butterfly(df, cfg, holdout_last=0)
    target = None
    for p in res["pairs"]:
        keys = {p["key_a"], p["key_b"]}
        if {meta["butterfly"]["key_a"], meta["butterfly"]["key_b"]} <= keys:
            target = p
            break
    assert target is not None, f"butterfly pair not found among {len(res['pairs'])} pairs"
    assert abs(target["lag_bins"]) == meta["butterfly"]["lag_bins"]
    assert abs(target["corr"]) >= 0.3
    assert target["days_ok"] >= 0.6 * target["days_total"]


def test_watchdog_and_single_device_filtered(synthetic, cfg):
    df, meta = synthetic
    b = build_bins(df, cfg)
    systems = {k[1] for k in b.keys}
    assert meta["watchdog"][1] not in systems
    assert meta["single_device"][1] not in systems


def test_duplicate_pair_classified(synthetic, cfg):
    df, meta = synthetic
    res = run_butterfly(df, cfg, holdout_last=0)
    pairs = finalize_pairs(df, res["pairs"], cfg)
    dup = [p for p in pairs
           if {p["key_a"][:3], p["key_b"][:3]} == {tuple(meta["duplicate"]["key_a"]),
                                                  tuple(meta["duplicate"]["key_b"])}]
    assert dup, "duplicate pair not detected"
    assert dup[0]["class"] in ("DUPLICATE",)
    assert dup[0]["overlap"]["verdict"] in ("SAME_DEVICE", "PARTIAL_OVERLAP")
    assert dup[0]["shared_unshared"]["corr_shared"] > 0.5


def test_butterfly_verdict_no_overlap(synthetic, cfg):
    df, meta = synthetic
    res = run_butterfly(df, cfg, holdout_last=0)
    pairs = finalize_pairs(df, res["pairs"], cfg)
    bf = [p for p in pairs
          if {p["key_a"][:3], p["key_b"][:3]} == {tuple(meta["butterfly"]["key_a"]),
                                                  tuple(meta["butterfly"]["key_b"])}]
    assert bf
    assert bf[0]["class"] == "BUTTERFLY_CANDIDATE"
    assert bf[0]["overlap"]["common"] == 0
    assert bf[0]["shared_unshared"]["corr_unshared"] is not None


def test_common_factor_removal_reduces_spurious_corr(synthetic, cfg):
    df, meta = synthetic
    b = build_bins(df, cfg)
    if b.M.size == 0:
        pytest.skip("no binned data")
    ka, kb = tuple(meta["common_cause"]["key_a"][:3]), tuple(meta["common_cause"]["key_b"][:3])
    if ka not in b.keys or kb not in b.keys:
        pytest.skip("common cause series filtered out")
    ia, ib = b.keys.index(ka), b.keys.index(kb)
    X = transform_matrix(b.M, cfg.transform)
    Z, _ = zscore(X)
    c_before = abs(float(np.corrcoef(Z[:, ia], Z[:, ib])[0, 1]))
    R = remove_common_factor(X, np.ones(X.shape[1], dtype=bool))
    Rz, _ = zscore(R)
    c_after = abs(float(np.corrcoef(Rz[:, ia], Rz[:, ib])[0, 1]))
    assert c_after < c_before


def test_maintenance_filter_removes_window_events(synthetic):
    df, meta = synthetic
    w = meta["maintenance_window"]
    store = MaintenanceStore(None)
    store.add(MaintenanceWindow(start=w["start"], end=w["end"], region=w["region"]))
    kept, dropped = store.filter_alarms(df)
    assert len(dropped) > 0
    assert (dropped["system"] == meta["maintenance_series"][1]).any()
    assert not (kept["system"] == meta["maintenance_series"][1]).any()
    mid = kept["ts"].between(w["start"], w["end"])
    assert not (mid & (kept["region"] == w["region"])).any()


def test_overlap_stats_token_matching():
    a = {("79.104.254.20", "i-pe2.Krasnodar"), ("i-pe2.Kazan", "i-pe2.Kazan")}
    b = {("i-pe2.Krasnodar", "i-pe2.Krasnodar"), ("mx01.Novosibirsk", "")}
    st = overlap_stats(a, b)
    assert st["common"] == 1
    c = {("node-x", "node-x")}
    d = {("node-y", "node-y")}
    assert overlap_stats(c, d)["verdict"] == "NO_OVERLAP"
    assert overlap_stats(c, c)["verdict"] == "SAME_DEVICE"

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EngineConfig
from .canonical import GROUP_KEY

OVERLAP_SAME = 0.70
OVERLAP_PARTIAL = 0.20
MIN_BINS_FOR_CORR = 30


def _tokens(dev: tuple[str, str]) -> set[str]:
    return {x for x in dev if x}


def shared_devices(a_devs: set[tuple[str, str]], b_devs: set[tuple[str, str]]):
    b_tokens = set().union(*(_tokens(d) for d in b_devs)) if b_devs else set()
    a_tokens = set().union(*(_tokens(d) for d in a_devs)) if a_devs else set()
    shared_a = {d for d in a_devs if _tokens(d) & b_tokens}
    shared_b = {d for d in b_devs if _tokens(d) & a_tokens}
    return shared_a, shared_b, a_devs - shared_a, b_devs - shared_b


def overlap_stats(a_devs: set[tuple[str, str]], b_devs: set[tuple[str, str]]) -> dict:
    sa, sb, ua, ub = shared_devices(a_devs, b_devs)
    common = len(sa)
    denom = min(len(a_devs), len(b_devs))
    frac = common / denom if denom else 0.0
    if common == 0:
        verdict = "NO_OVERLAP"
    elif frac >= OVERLAP_SAME:
        verdict = "SAME_DEVICE"
    elif frac >= OVERLAP_PARTIAL:
        verdict = "PARTIAL_OVERLAP"
    else:
        verdict = "LOW_OVERLAP"
    return dict(
        devices_a=len(a_devs),
        devices_b=len(b_devs),
        common=common,
        shared_a=len(sa),
        shared_b=len(sb),
        unshared_a=len(ua),
        unshared_b=len(ub),
        overlap_frac=round(frac, 4),
        verdict=verdict,
    )


def _series_vectorized(sub: pd.DataFrame, devs: frozenset | None, grid: pd.DatetimeIndex,
                       bin_min: int) -> np.ndarray:
    if sub.empty:
        return np.zeros(len(grid))
    if devs is not None:
        mi = pd.MultiIndex.from_frame(sub[["node", "node_alias"]])
        sub = sub[mi.isin(devs)]
        if sub.empty:
            return np.zeros(len(grid))
    b = sub["ts"].dt.floor(f"{bin_min}min")
    vc = b.value_counts()
    v = vc.reindex(grid).fillna(0).to_numpy(dtype=float)
    return np.log1p(v)


def shared_unshared_corr(df: pd.DataFrame, key_a: tuple, key_b: tuple, lag: int,
                         cfg: EngineConfig) -> dict:
    a0, a1, a2 = key_a
    b0, b1, b2 = key_b
    d = df[
        (df["severity"] > 0)
        & (((df["integration"] == a0) & (df["system"] == a1) & (df["alert_group"] == a2))
           | ((df["integration"] == b0) & (df["system"] == b1) & (df["alert_group"] == b2)))
    ]
    if d.empty:
        return dict(corr_all=None, corr_shared=None, corr_unshared=None, verdict="NO_DATA")
    A = d[(d["integration"] == a0) & (d["system"] == a1) & (d["alert_group"] == a2)]
    B = d[(d["integration"] == b0) & (d["system"] == b1) & (d["alert_group"] == b2)]
    devs_a = frozenset(zip(A["node"].fillna(""), A["node_alias"].fillna("")))
    devs_b = frozenset(zip(B["node"].fillna(""), B["node_alias"].fillna("")))
    sha, shb, ua, ub = shared_devices(devs_a, devs_b)

    freq = f"{cfg.bin_min}min"
    lo = d["ts"].min().floor(freq)
    hi = d["ts"].max().floor(freq)
    grid = pd.date_range(lo, hi, freq=freq)
    gv = np.log1p(d["ts"].dt.floor(freq).value_counts().reindex(grid).fillna(0).to_numpy(dtype=float))

    def demean(x: np.ndarray) -> np.ndarray:
        if gv.std() == 0 or x.std() == 0:
            return x - x.mean()
        Amat = np.column_stack([np.ones(len(x)), (gv - gv.mean()) / gv.std()])
        beta, *_ = np.linalg.lstsq(Amat, x, rcond=None)
        return x - Amat @ beta

    def corr(x: np.ndarray, y: np.ndarray) -> float | None:
        T = len(x)
        if lag >= 0:
            xx, yy = x[: T - lag], y[lag:]
        else:
            xx, yy = x[-lag:], y[: T + lag]
        if len(xx) < MIN_BINS_FOR_CORR or xx.std() == 0 or yy.std() == 0:
            return None
        return float(np.corrcoef(xx, yy)[0, 1])

    xa = demean(_series_vectorized(A, None, grid, cfg.bin_min))
    xb = demean(_series_vectorized(B, None, grid, cfg.bin_min))
    corr_all = corr(xa, xb)

    if not sha and not shb:
        return dict(
            corr_all=None if corr_all is None else round(corr_all, 4),
            corr_shared=None,
            corr_unshared=None if corr_all is None else round(corr_all, 4),
            verdict="NO_COMMON_DEVICES",
        )

    corr_shared = None
    corr_unshared = None
    if sha and shb:
        xs = demean(_series_vectorized(A, frozenset(sha), grid, cfg.bin_min))
        ys = demean(_series_vectorized(B, frozenset(shb), grid, cfg.bin_min))
        corr_shared = corr(xs, ys)
    if ua and ub:
        xu = demean(_series_vectorized(A, frozenset(ua), grid, cfg.bin_min))
        yu = demean(_series_vectorized(B, frozenset(ub), grid, cfg.bin_min))
        corr_unshared = corr(xu, yu)

    cs, cu = corr_shared, corr_unshared
    if cs is None:
        verdict = "AMBIGUOUS"
    elif cs >= 0.3 and (cu is None or abs(cu) < max(0.15, 0.3 * cs)):
        verdict = "DUPLICATE"
    elif cu is not None and abs(cu) >= max(0.2, 0.5 * cs):
        verdict = "CROSS_OBJECT"
    else:
        verdict = "AMBIGUOUS"
    return dict(
        corr_all=None if corr_all is None else round(corr_all, 4),
        corr_shared=None if cs is None else round(cs, 4),
        corr_unshared=None if cu is None else round(cu, 4),
        verdict=verdict,
    )


BUTTERFLY_VERDICTS = {"NO_COMMON_DEVICES", "NO_OVERLAP"}


def finalize_pairs(alarms: pd.DataFrame, pairs: list[dict], cfg: EngineConfig) -> list[dict]:
    dev_sets = {
        k: set(zip(g["node"].fillna(""), g["node_alias"].fillna("")))
        for k, g in alarms[alarms["severity"] > 0].groupby(GROUP_KEY, sort=False)
    }
    for p in pairs:
        da = dev_sets.get(p["key_a"][:3], set())
        db = dev_sets.get(p["key_b"][:3], set())
        p["overlap"] = overlap_stats(da, db)
        su = shared_unshared_corr(alarms, p["key_a"][:3], p["key_b"][:3], p["lag_bins"], cfg)
        p["shared_unshared"] = su
        if su["verdict"] in BUTTERFLY_VERDICTS:
            p["class"] = "BUTTERFLY_CANDIDATE"
        elif su["verdict"] == "DUPLICATE":
            p["class"] = "DUPLICATE"
        else:
            p["class"] = su["verdict"]
        p["score"] = round(
            abs(p.get("corr", 0))
            * (p.get("days_ok", 0) / max(p.get("days_total", 1), 1))
            * (p.get("regions_frac") or 0.5)
            * (1 if p.get("month_stable") else 0.5),
            4,
        )
    return sorted(pairs, key=lambda p: -p["score"])

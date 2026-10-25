from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import EngineConfig
from .canonical import GROUP_KEY, GROUP_KEY_REGION

SYSTEM_FIELDS = {"system", "integration", "both"}


@dataclass
class BinnedData:
    M: np.ndarray
    keys: list[tuple]
    grid: list[datetime]
    region_series: dict[tuple, dict[str, np.ndarray]]
    device_sets: dict[tuple, set]
    events: dict[tuple, int]
    dropped_volume: int = 0
    dropped_devices: int = 0


def transform_matrix(M: np.ndarray, how: str) -> np.ndarray:
    X = M.copy()
    if how == "log1p":
        X = np.log1p(X)
    elif how == "binary":
        X = (X > 0).astype(float)
    return X


def zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0)
    Z = np.zeros_like(X)
    ok = sd > 0
    Z[:, ok] = (X[:, ok] - mu[ok]) / sd[ok]
    return Z, ok


def remove_common_factor(X: np.ndarray, ok: np.ndarray) -> np.ndarray:
    T = X.shape[0]
    g = X.sum(1).astype(float)
    if g.std() == 0:
        return X.copy()
    gz = (g - g.mean()) / g.std()
    A = np.column_stack([np.ones(T), gz])
    R = np.zeros_like(X)
    for j in range(X.shape[1]):
        if not ok[j]:
            continue
        y = X[:, j].astype(float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        R[:, j] = y - A @ beta
    return R


def lag_corr(Z: np.ndarray, lag: int) -> np.ndarray:
    T = Z.shape[0]
    if lag >= 0:
        A, B = Z[: T - lag], Z[lag:]
    else:
        A, B = Z[-lag:], Z[: T + lag]
    return (A.T @ B) / A.shape[0]


def best_lag_matrix(Z: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    N = Z.shape[1]
    best = np.zeros((N, N))
    bl = np.zeros((N, N), dtype=int)
    for lag in range(-lags, lags + 1):
        C = lag_corr(Z, lag)
        m = np.abs(C) > np.abs(best)
        best[m] = C[m]
        bl[m] = lag
    return best, bl


def support_matrix(M: np.ndarray) -> np.ndarray:
    B = (M > 0).astype(np.int32)
    return B.T @ B


def is_cross_system(ka: tuple, kb: tuple, mode: str) -> bool:
    if mode == "system":
        return ka[1] != kb[1]
    if mode == "integration":
        return ka[0] != kb[0]
    return ka[0] != kb[0] and ka[1] != kb[1]


def day_slices(grid: list[datetime]) -> list[np.ndarray]:
    out: dict = defaultdict(list)
    for i, t in enumerate(grid):
        out[t.date()].append(i)
    return [np.array(v) for v in out.values()]


def hour_slices(grid: list[datetime]) -> list[tuple[int, np.ndarray]]:
    out: dict = defaultdict(list)
    for i, t in enumerate(grid):
        out[t.hour].append(i)
    return [(h, np.array(v)) for h, v in sorted(out.items())]


def corr_on_idx(a: np.ndarray, b: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> float:
    if len(ia) < 30:
        return float("nan")
    x, y = a[ia], b[ib]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def stability(a: np.ndarray, b: np.ndarray, lag: int, slices: list[np.ndarray], min_abs: float = 0.3):
    T = len(a)
    vals = []
    for idx in slices:
        ib = idx + lag
        m = (ib >= 0) & (ib < T)
        v = corr_on_idx(a, b, idx[m], ib[m])
        if v == v:
            vals.append(v)
    if not vals:
        return 0, 0, 0.0
    ref = np.sign(np.median(vals))
    ok = sum(1 for v in vals if np.sign(v) == ref and abs(v) >= min_abs)
    return len(vals), ok, float(np.median(vals))


def hour_stability(a: np.ndarray, b: np.ndarray, lag: int, hours: list[tuple[int, np.ndarray]], min_abs: float = 0.2):
    T = len(a)
    vals = []
    for _, idx in hours:
        ib = idx + lag
        m = (ib >= 0) & (ib < T)
        v = corr_on_idx(a, b, idx[m], ib[m])
        if v == v:
            vals.append(v)
    if not vals:
        return 0, 0
    ref = np.sign(np.median(vals))
    ok = sum(1 for v in vals if np.sign(v) == ref and abs(v) >= min_abs)
    return len(vals), ok


def region_recurrence(reg: dict[tuple, dict[str, np.ndarray]], ka: tuple, kb: tuple, lag: int,
                      min_abs: float, ref_sign: float):
    ra, rb = reg.get(ka, {}), reg.get(kb, {})
    det, ok = [], 0
    for r in sorted(set(ra) & set(rb)):
        x, y = np.log1p(ra[r]), np.log1p(rb[r])
        if (x > 0).sum() < 10 or (y > 0).sum() < 10:
            continue
        T = len(x)
        if lag >= 0:
            xx, yy = x[: T - lag], y[lag:]
        else:
            xx, yy = x[-lag:], y[: T + lag]
        if len(xx) < 30 or xx.std() == 0 or yy.std() == 0:
            continue
        c = float(np.corrcoef(xx, yy)[0, 1])
        det.append((r, c))
        if abs(c) >= min_abs and np.sign(c) == ref_sign:
            ok += 1
    return len(det), ok, det


def build_bins(df: pd.DataFrame, cfg: EngineConfig) -> BinnedData:
    if df.empty:
        return BinnedData(np.zeros((0, 0)), [], [], {}, {}, {})
    d = df[df["severity"] > 0].copy()
    d["bucket"] = d["ts"].dt.floor(f"{cfg.bin_min}min")
    key_cols = GROUP_KEY if cfg.drop_region else GROUP_KEY_REGION

    agg = d.groupby(key_cols + ["bucket"], sort=False).size().reset_index(name="cnt")
    dev: dict[tuple, set] = {}
    for k, g in d.groupby(GROUP_KEY, sort=False):
        dev[k] = set(zip(g["node"].fillna(""), g["node_alias"].fillna("")))
    events = agg.groupby(key_cols)["cnt"].sum().to_dict()
    bins_n = agg.groupby(key_cols)["bucket"].nunique().to_dict()

    keep = [k for k in events if events[k] >= cfg.min_events and bins_n[k] >= cfg.min_bins]
    if cfg.min_devices > 0:
        keep2 = [k for k in keep if len(dev.get(k[:3], ())) >= cfg.min_devices]
        cfg_dropped = len(keep) - len(keep2)
        keep = keep2
    else:
        cfg_dropped = 0
    vol_dropped = len(events) - len(keep)

    kept_keys = sorted(keep)
    if not kept_keys:
        return BinnedData(np.zeros((0, 0)), [], [], {}, dev, {}, vol_dropped, cfg_dropped)

    keep_set = set(kept_keys)
    key_tuples = list(zip(*[agg[c] for c in key_cols]))
    sub = agg[[t in keep_set for t in key_tuples]]
    tmin, tmax = sub["bucket"].min(), sub["bucket"].max()
    grid, t = [], tmin
    while t <= tmax:
        grid.append(t)
        t += timedelta(minutes=cfg.bin_min)
    bindex = {b: i for i, b in enumerate(grid)}
    kindex = {k: i for i, k in enumerate(kept_keys)}

    M = np.zeros((len(grid), len(kept_keys)), dtype=np.float64)
    for row in sub.itertuples(index=False):
        M[bindex[row.bucket], kindex[tuple(getattr(row, c) for c in key_cols)]] += row.cnt

    region_series: dict[tuple, dict[str, np.ndarray]] = defaultdict(dict)
    if cfg.drop_region and cfg.region_min_abs > 0:
        rs = d.groupby(GROUP_KEY_REGION + ["bucket"], sort=False).size().reset_index(name="cnt")
        kept_groups = {k[:3] for k in kept_keys}
        for row in rs.itertuples(index=False):
            g = (row.integration, row.system, row.alert_group)
            if g not in kept_groups:
                continue
            i = bindex.get(row.bucket)
            if i is None:
                continue
            arr = region_series[g].get(row.region)
            if arr is None:
                arr = np.zeros(len(grid), dtype=np.float64)
                region_series[g][row.region] = arr
            arr[i] += row.cnt

    return BinnedData(M, kept_keys, grid, dict(region_series), dict(dev), events,
                      vol_dropped, cfg_dropped)


def detect(b: BinnedData, cfg: EngineConfig) -> list[dict]:
    if b.M.size == 0 or not b.keys:
        return []
    X = transform_matrix(b.M, cfg.transform)
    Z, ok = zscore(X)
    if cfg.remove_common_factor:
        R = remove_common_factor(X, ok)
        Rz, okR = zscore(R)
    else:
        Rz, okR = Z, ok

    best, bl = best_lag_matrix(Rz, cfg.lags)
    C0 = lag_corr(Rz, 0)
    sup = support_matrix(b.M)
    duty = (b.M > 0).mean(0)
    N = len(b.keys)

    hot = {i for i in range(N) if duty[i] > cfg.max_duty}
    strong = (np.abs(best) >= cfg.min_abs_corr) & okR[:, None] & okR[None, :]
    degree = strong.sum(1) - 1
    hubs = {i for i in range(N) if degree[i] > cfg.hub_degree}

    days = day_slices(b.grid)
    hours = hour_slices(b.grid)
    iu = np.triu_indices(N, k=1)
    res = []
    for i, j in zip(*iu):
        if not (okR[i] and okR[j]) or i in hubs or j in hubs or i in hot or j in hot:
            continue
        if not is_cross_system(b.keys[i], b.keys[j], cfg.system_field):
            continue
        c = best[i, j]
        if abs(c) < cfg.min_abs_corr or sup[i, j] < cfg.min_support:
            continue
        nd, nok, med = stability(Rz[:, i], Rz[:, j], bl[i, j], days)
        if nd == 0 or nok < cfg.min_days_frac * nd:
            continue
        if cfg.min_hours_frac > 0:
            nh, nhok = hour_stability(Rz[:, i], Rz[:, j], bl[i, j], hours)
            if nh == 0 or nhok < cfg.min_hours_frac * nh:
                continue
        pair = dict(
            key_a=b.keys[i],
            key_b=b.keys[j],
            corr=round(float(c), 4),
            corr0=round(float(C0[i, j]), 4),
            lead=round(abs(float(c)) - abs(float(C0[i, j])), 4),
            lag_bins=int(bl[i, j]),
            lag_min=int(bl[i, j]) * cfg.bin_min,
            support=int(sup[i, j]),
            days_total=int(nd),
            days_ok=int(nok),
            med_corr=round(med, 4),
            duty_a=round(float(duty[i]), 3),
            duty_b=round(float(duty[j]), 3),
            devices_a=len(b.device_sets.get(b.keys[i][:3], ())),
            devices_b=len(b.device_sets.get(b.keys[j][:3], ())),
        )
        if cfg.drop_region and cfg.region_min_abs > 0:
            tot, okr, _ = region_recurrence(b.region_series, b.keys[i], b.keys[j],
                                            bl[i, j], cfg.region_min_abs, np.sign(c))
            pair["regions_total"] = tot
            pair["regions_ok"] = okr
            pair["regions_frac"] = round(okr / tot, 2) if tot else 0.0
        else:
            pair["regions_total"] = None
            pair["regions_ok"] = None
            pair["regions_frac"] = None
        res.append(pair)
    res.sort(key=lambda r: -abs(r["corr"]))
    return res


def pair_id(p: dict) -> str:
    ka, kb = sorted([p["key_a"], p["key_b"]])
    return f"{ka[0]}|{ka[1]}|{ka[2]}<->{kb[0]}|{kb[1]}|{kb[2]}"


def corr_at_lag(b: BinnedData, cfg: EngineConfig, ka: tuple, kb: tuple, lag: int) -> float | None:
    if ka not in b.keys or kb not in b.keys:
        return None
    ia, ib = b.keys.index(ka), b.keys.index(kb)
    X = transform_matrix(b.M[:, [ia, ib]], cfg.transform)
    if cfg.remove_common_factor:
        ok = np.ones(2, dtype=bool)
        R = remove_common_factor(X, ok)
        Z, _ = zscore(R)
    else:
        Z, _ = zscore(X)
    T = Z.shape[0]
    if lag >= 0:
        xx, yy = Z[: T - lag, 0], Z[lag:, 1]
    else:
        xx, yy = Z[-lag:, 0], Z[: T + lag, 1]
    if len(xx) < 30 or xx.std() == 0 or yy.std() == 0:
        return None
    return float(np.corrcoef(xx, yy)[0, 1])


def month_of(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def run_butterfly(df: pd.DataFrame, cfg: EngineConfig, holdout_last: int = 1) -> dict:
    months = sorted({month_of(t) for t in df["ts"]})
    if holdout_last > 0 and len(months) >= max(2, holdout_last + 1):
        train_m = months[:-holdout_last]
        hold_m = months[-holdout_last:]
    else:
        train_m, hold_m = months, []
    train = df[df["ts"].map(month_of).isin(set(train_m))]
    b = build_bins(train, cfg)
    pairs = detect(b, cfg)

    hold_stats = {}
    if hold_m:
        hold = df[df["ts"].map(month_of).isin(set(hold_m))]
        hb = build_bins(hold, cfg)
        for p in pairs:
            ch = corr_at_lag(hb, cfg, p["key_a"], p["key_b"], p["lag_bins"])
            okh = ch is not None and np.sign(ch) == np.sign(p["corr"]) and abs(ch) >= cfg.holdout_min_corr_ratio * abs(p["corr"])
            hold_stats[pair_id(p)] = dict(holdout_corr=None if ch is None else round(ch, 4),
                                          holdout_ok=bool(okh), holdout_months=hold_m)
    for p in pairs:
        p["id"] = pair_id(p)
        p["holdout"] = hold_stats.get(p["id"], {"holdout_corr": None, "holdout_ok": None,
                                                "holdout_months": hold_m})
        p["month_stable"] = p["holdout"]["holdout_ok"]

    info = dict(
        months=months,
        train_months=train_m,
        holdout_months=hold_m,
        rows_series=len(b.keys),
        bins=len(b.grid),
        dropped_volume=b.dropped_volume,
        dropped_min_devices=b.dropped_devices,
        pairs_found=len(pairs),
        lead_pairs=sum(1 for p in pairs if p["lead"] >= cfg.min_lead and p["lag_bins"] != 0),
    )
    return dict(pairs=pairs, info=info, config=asdict(cfg))

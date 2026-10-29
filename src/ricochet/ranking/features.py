from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..canonical import GROUP_KEY

EPS = 1e-9
_TOKEN_RE = re.compile(r"[a-z]+|\d+")

# Feature order is part of the persisted model contract: changing it requires
# retraining the model. Keep this list append-only for backwards compatibility.
FEATURE_NAMES: list[str] = [
    "corr_abs",
    "corr_signed",
    "corr0_abs",
    "lead",
    "lead_ratio",
    "lag_abs_min",
    "lag_is_zero",
    "support_log",
    "days_frac",
    "days_total_log",
    "regions_frac",
    "duty_max",
    "duty_min",
    "device_min_log",
    "device_max_log",
    "device_ratio",
    "score_engine",
    "month_stable",
    "holdout_ok",
    "class_butterfly",
    "class_duplicate",
    "class_cross_object",
    "class_ambiguous",
    "overlap_frac",
    "common_devices_log",
    "corr_shared_abs",
    "corr_unshared_abs",
    "same_integration",
    "same_system",
    "graph_degree_log",
    "cooccurrence_lift_log",
    "temporal_jaccard",
    "topology_site_affinity",
    "topology_token_jaccard",
    "topology_region_affinity",
]

def _pair_key(pair: dict) -> tuple[tuple, tuple] | None:
    ka, kb = pair.get("key_a"), pair.get("key_b")
    if not ka or not kb:
        return None
    return tuple(ka[:3]), tuple(kb[:3])


def _canonical_pair(pair: dict) -> tuple[tuple, tuple] | None:
    keys = _pair_key(pair)
    if keys is None:
        return None
    return tuple(sorted(keys))


def _as_float(value, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def _overlap_coeff(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    common = sum(min(a[k], b[k]) for k in set(a) & set(b))
    denom = min(sum(a.values()), sum(b.values()))
    return common / denom if denom else float("nan")


def _sites(devices: set[tuple[str, str]]) -> set[str]:
    out: set[str] = set()
    for node, _alias in devices:
        raw = (node or "").strip().lower()
        if not raw:
            continue
        out.add(raw.rsplit(".", 1)[1] if "." in raw else raw)
    return out


def _tokens(devices: set[tuple[str, str]]) -> set[str]:
    out: set[str] = set()
    for node, alias in devices:
        for raw in (node, alias):
            for token in _TOKEN_RE.findall((raw or "").lower()):
                out.add("#" if token.isdigit() else token)
    return out


@dataclass
class PairFeatureContext:
    """Pair-level facts derived once from the alarm stream.

    The context deliberately lives outside ``leadlag.py``: the statistical engine
    stays unaware of lift/topology, while the ranker can use them as extra signals.
    """

    lift: dict[tuple[tuple, tuple], float] = field(default_factory=dict)
    temporal_jaccard: dict[tuple[tuple, tuple], float] = field(default_factory=dict)
    site_affinity: dict[tuple[tuple, tuple], float] = field(default_factory=dict)
    token_jaccard: dict[tuple[tuple, tuple], float] = field(default_factory=dict)
    region_affinity: dict[tuple[tuple, tuple], float] = field(default_factory=dict)
    degree: dict[tuple, int] = field(default_factory=dict)
    total_bins: int = 0
    bin_min: int = 5

    @property
    def available(self) -> bool:
        return bool(self.lift or self.site_affinity or self.region_affinity)

    @classmethod
    def build(cls, alarms: pd.DataFrame | None, pairs: list[dict], cfg=None) -> "PairFeatureContext":
        ctx = cls(bin_min=int(getattr(cfg, "bin_min", 5) or 5))
        if alarms is None or alarms.empty or not pairs:
            return ctx

        keys = sorted({k for p in pairs for k in _pair_key(p) or ()})
        if not keys:
            return ctx

        d = alarms
        if "severity" in d.columns:
            d = d[d["severity"] > 0]
        if d.empty:
            return ctx

        codes = pd.MultiIndex.from_frame(d[GROUP_KEY])
        d = d[codes.isin(keys)]
        if d.empty:
            return ctx
        d = d.copy()
        d["bucket"] = d["ts"].dt.floor(f"{ctx.bin_min}min")

        grouped = d.groupby(GROUP_KEY + ["bucket"], sort=False).size()
        active: dict[tuple, set] = defaultdict(set)
        for key_parts in grouped.index:
            active[tuple(key_parts[:-1])].add(key_parts[-1])

        ctx.total_bins = int(d["bucket"].nunique())
        if ctx.total_bins <= 0:
            return ctx

        devices: dict[tuple, set[tuple[str, str]]] = {}
        regions: dict[tuple, Counter] = {}
        for key, group in d.groupby(GROUP_KEY, sort=False):
            k = tuple(key)
            devices[k] = set(zip(group["node"].fillna(""), group["node_alias"].fillna("")))
            regions[k] = Counter(r for r in group["region"].fillna("").astype(str) if r)

        sites = {k: _sites(v) for k, v in devices.items()}
        tokens = {k: _tokens(v) for k, v in devices.items()}

        for pair in pairs:
            token = _canonical_pair(pair)
            if token is None:
                continue
            ka, kb = token
            a, b = active.get(ka, set()), active.get(kb, set())
            if not a or not b:
                continue
            both = len(a & b)
            union = len(a | b)
            pa, pb = len(a) / ctx.total_bins, len(b) / ctx.total_bins
            if pa > 0 and pb > 0:
                ctx.lift[token] = (both / ctx.total_bins) / (pa * pb)
                ctx.temporal_jaccard[token] = both / union if union else float("nan")
            ctx.site_affinity[token] = _jaccard(sites.get(ka, set()), sites.get(kb, set()))
            ctx.token_jaccard[token] = _jaccard(tokens.get(ka, set()), tokens.get(kb, set()))
            ctx.region_affinity[token] = _overlap_coeff(
                regions.get(ka, Counter()), regions.get(kb, Counter()))
            ctx.degree[ka] = ctx.degree.get(ka, 0) + 1
            ctx.degree[kb] = ctx.degree.get(kb, 0) + 1
        return ctx


def _class_flag(value: str, expected: str) -> float:
    return 1.0 if (value or "").upper() == expected else 0.0


def _tri_state(value) -> float:
    if value is None:
        return float("nan")
    return 1.0 if bool(value) else 0.0


def extract_features(pair: dict, context: PairFeatureContext | None = None) -> dict[str, float]:
    """Build the model input from a finalized butterfly pair.

    Missing values are intentional: XGBoost handles NaN natively, so the ranker
    can distinguish "feature is absent" from "feature equals zero".
    """

    corr = _as_float(pair.get("corr"), 0.0)
    corr0 = _as_float(pair.get("corr0"), 0.0)
    lead = _as_float(pair.get("lead"), abs(corr) - abs(corr0))
    lag_bins = _as_float(pair.get("lag_bins"), 0.0)
    lag_min = _as_float(pair.get("lag_min"), 0.0)
    support = _as_float(pair.get("support"), 0.0)
    days_total = _as_float(pair.get("days_total"), 0.0)
    days_ok = _as_float(pair.get("days_ok"), 0.0)
    devices_a = _as_float(pair.get("devices_a"), 0.0)
    devices_b = _as_float(pair.get("devices_b"), 0.0)
    duty_a = _as_float(pair.get("duty_a"), 0.0)
    duty_b = _as_float(pair.get("duty_b"), 0.0)

    overlap = pair.get("overlap") or {}
    shared = pair.get("shared_unshared") or {}
    overlap_frac = _as_float(overlap.get("overlap_frac"),
                             _safe_ratio(_as_float(overlap.get("common"), 0.0),
                                         min(devices_a, devices_b)))
    common = _as_float(overlap.get("common"), 0.0)
    corr_shared = _as_float(shared.get("corr_shared"))
    corr_unshared = _as_float(shared.get("corr_unshared"))

    holdout = pair.get("holdout") or {}
    month_stable = _tri_state(pair.get("month_stable", holdout.get("holdout_ok")))
    holdout_ok = _tri_state(holdout.get("holdout_ok"))

    ka, kb = _pair_key(pair) or (("", "", ""), ("", "", ""))
    cls = str(pair.get("class") or "")
    token = _canonical_pair(pair)

    def ctx(name: str) -> float:
        if context is None or token is None:
            return float("nan")
        return _as_float(getattr(context, name).get(token))

    lift = ctx("lift")
    features = {
        "corr_abs": abs(corr),
        "corr_signed": corr,
        "corr0_abs": abs(corr0),
        "lead": lead,
        "lead_ratio": _safe_ratio(lead, abs(corr0) + EPS),
        "lag_abs_min": abs(lag_min),
        "lag_is_zero": 1.0 if lag_bins == 0 else 0.0,
        "support_log": math.log1p(max(support, 0.0)),
        "days_frac": _safe_ratio(days_ok, days_total),
        "days_total_log": math.log1p(max(days_total, 0.0)),
        "regions_frac": _as_float(pair.get("regions_frac")),
        "duty_max": max(duty_a, duty_b),
        "duty_min": min(duty_a, duty_b),
        "device_min_log": math.log1p(max(min(devices_a, devices_b), 0.0)),
        "device_max_log": math.log1p(max(max(devices_a, devices_b), 0.0)),
        "device_ratio": _safe_ratio(min(devices_a, devices_b), max(devices_a, devices_b)),
        "score_engine": _as_float(pair.get("score"), 0.0),
        "month_stable": month_stable,
        "holdout_ok": holdout_ok,
        "class_butterfly": _class_flag(cls, "BUTTERFLY_CANDIDATE"),
        "class_duplicate": _class_flag(cls, "DUPLICATE"),
        "class_cross_object": _class_flag(cls, "CROSS_OBJECT"),
        "class_ambiguous": _class_flag(cls, "AMBIGUOUS"),
        "overlap_frac": overlap_frac,
        "common_devices_log": math.log1p(max(common, 0.0)),
        "corr_shared_abs": abs(corr_shared) if math.isfinite(corr_shared) else float("nan"),
        "corr_unshared_abs": abs(corr_unshared) if math.isfinite(corr_unshared) else float("nan"),
        "same_integration": 1.0 if ka[0] == kb[0] else 0.0,
        "same_system": 1.0 if ka[1] == kb[1] else 0.0,
        "graph_degree_log": math.log1p(
            max(context.degree.get(ka, 0) + context.degree.get(kb, 0), 0)) if context else float("nan"),
        "cooccurrence_lift_log": math.log1p(max(lift, 0.0)) if math.isfinite(lift) else float("nan"),
        "temporal_jaccard": ctx("temporal_jaccard"),
        "topology_site_affinity": ctx("site_affinity"),
        "topology_token_jaccard": ctx("token_jaccard"),
        "topology_region_affinity": ctx("region_affinity"),
    }
    return {name: float(features.get(name, float("nan"))) for name in FEATURE_NAMES}


def featurize_pairs(pairs: list[dict], context: PairFeatureContext | None = None) -> list[dict[str, float]]:
    return [extract_features(p, context) for p in pairs]


def features_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=float)


def json_features(row: dict[str, float]) -> dict[str, float | None]:
    """JSON-safe snapshot used by the feedback store."""
    return {name: (None if not math.isfinite(row.get(name, float("nan"))) else float(row[name]))
            for name in FEATURE_NAMES}


def features_from_json(row: dict) -> dict[str, float]:
    return {name: _as_float(row.get(name)) for name in FEATURE_NAMES}

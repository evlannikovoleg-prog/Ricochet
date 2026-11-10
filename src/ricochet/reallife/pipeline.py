from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..canonical import GROUP_KEY, alarms_from_records
from .schemas import IncidentBundleIn, MaintenanceWindowIn

SeriesKey = tuple[str, str, str]


def series_id(key: SeriesKey) -> str:
    return "|".join(str(part) for part in key[:3])


def _as_bundle(bundle: IncidentBundleIn | dict) -> IncidentBundleIn:
    if isinstance(bundle, IncidentBundleIn):
        return bundle
    return IncidentBundleIn.model_validate(bundle)


def _window_hit(ts: pd.Timestamp, region: str, system: str, node: str,
                windows: list[MaintenanceWindowIn]) -> list[dict]:
    hits = []
    for window in windows:
        if not (window.start <= ts.to_pydatetime() <= window.end):
            continue
        if window.region and window.region != region:
            continue
        if window.system and window.system != system:
            continue
        if window.node and window.node not in {node, ""}:
            continue
        hits.append(window.model_dump())
    return hits


@dataclass
class PreparedIncident:
    incident_id: str
    region: str
    start: datetime
    end: datetime
    step_seconds: int
    ewma_alpha: float
    alarms: pd.DataFrame
    keys: list[SeriesKey]
    grid: pd.DatetimeIndex
    intensity: pd.DataFrame
    ewma: pd.DataFrame
    meta: dict[SeriesKey, dict]
    device_sets: dict[SeriesKey, set[tuple[str, str]]] = field(default_factory=dict)
    maintenance_hits: list[dict] = field(default_factory=list)
    alarms_in_maintenance: int = 0
    warnings: list[str] = field(default_factory=list)
    bundle: IncidentBundleIn | None = None

    def key_of(self, sid: str) -> SeriesKey | None:
        for key in self.keys:
            if series_id(key) == sid:
                return key
        return None

    @property
    def empty(self) -> bool:
        return not self.keys or self.intensity.empty


def _adaptive_step(span_seconds: float, requested_seconds: int, max_bins: int) -> tuple[int, bool]:
    step = max(1, int(requested_seconds))
    if max_bins <= 0:
        return step, False
    if span_seconds / step <= max_bins:
        return step, False
    raw = span_seconds / max_bins
    step = max(step, int(math.ceil(raw / 10.0) * 10))
    return step, True


def prepare_incident(bundle: IncidentBundleIn | dict, *, step_seconds: int = 10,
                     ewma_half_life_seconds: float = 60.0,
                     max_bins: int = 3000, max_series: int = 5000) -> PreparedIncident:
    """Build intensity and EWMA series from one incident bundle.

    This is the single preprocessing path for all realtime ML endpoints. It uses
    local memory now; the production adapter can replace the source with
    ClickHouse without changing the downstream cluster/causal/runbook code.
    """
    bundle = _as_bundle(bundle)
    records = [a.model_dump() for a in bundle.alarms]
    alarms = alarms_from_records(records) if records else alarms_from_records([])

    if alarms.empty:
        now = bundle.received_at or datetime.now()
        empty_index = pd.DatetimeIndex([pd.Timestamp(now)])
        empty_frame = pd.DataFrame(index=empty_index)
        return PreparedIncident(
            incident_id=bundle.incident_id,
            region=bundle.region,
            start=now,
            end=now,
            step_seconds=max(1, int(step_seconds)),
            ewma_alpha=0.0,
            alarms=alarms,
            keys=[],
            grid=empty_index,
            intensity=empty_frame,
            ewma=empty_frame,
            meta={},
            device_sets={},
            warnings=["empty incident: no alarms to analyse"],
            bundle=bundle,
        )

    if bundle.start is not None:
        alarms = alarms[alarms["ts"] >= pd.Timestamp(bundle.start)]
    if bundle.end is not None:
        alarms = alarms[alarms["ts"] <= pd.Timestamp(bundle.end)]
    if alarms.empty:
        return prepare_incident(
            IncidentBundleIn(incident_id=bundle.incident_id, region=bundle.region,
                             received_at=bundle.received_at, alarms=[],
                             maintenance=bundle.maintenance, metadata=bundle.metadata),
            step_seconds=step_seconds, ewma_half_life_seconds=ewma_half_life_seconds,
            max_bins=max_bins, max_series=max_series)

    t_min = pd.Timestamp(bundle.start) if bundle.start is not None else alarms["ts"].min()
    t_max = pd.Timestamp(bundle.end) if bundle.end is not None else alarms["ts"].max()
    t_min = min(t_min, alarms["ts"].min())
    t_max = max(t_max, alarms["ts"].max())
    span_seconds = max(1.0, (t_max - t_min).total_seconds())
    step, adjusted = _adaptive_step(span_seconds, step_seconds, max_bins)

    fmt = f"{step}s"
    work = alarms.copy()
    series_warning = None
    if max_series and work.groupby(GROUP_KEY, sort=False).ngroups > max_series:
        sizes = work.groupby(GROUP_KEY, sort=False).size().sort_values(ascending=False)
        keep = set(sizes.head(int(max_series)).index)
        codes = pd.MultiIndex.from_frame(work[GROUP_KEY])
        work = work[codes.isin(keep)].copy()
        series_warning = (
            f"series limited to {max_series} of {len(sizes)} by event volume; "
            "increase max_series to include rare groups")
    if work.empty:
        return prepare_incident(
            IncidentBundleIn(incident_id=bundle.incident_id, region=bundle.region,
                             received_at=bundle.received_at, alarms=[],
                             maintenance=bundle.maintenance, metadata=bundle.metadata),
            step_seconds=step_seconds, ewma_half_life_seconds=ewma_half_life_seconds,
            max_bins=max_bins, max_series=0)
    work["bucket"] = work["ts"].dt.floor(fmt)
    keys = sorted({tuple(row) for row in work[GROUP_KEY].itertuples(index=False, name=None)})
    key_index = {key: i for i, key in enumerate(keys)}

    start_floor = t_min.floor(fmt)
    end_floor = t_max.floor(fmt)
    grid = pd.date_range(start_floor, end_floor, freq=fmt)
    if len(grid) == 0:
        grid = pd.DatetimeIndex([start_floor])
    bucket_index = {ts: i for i, ts in enumerate(grid)}

    matrix = np.zeros((len(grid), len(keys)), dtype=float)
    events = Counter()
    devices: dict[SeriesKey, set[tuple[str, str]]] = {k: set() for k in keys}
    regions: dict[SeriesKey, Counter] = {k: Counter() for k in keys}
    severity: dict[SeriesKey, int] = {k: 0 for k in keys}
    systems: dict[SeriesKey, set[str]] = {k: set() for k in keys}
    integrations: dict[SeriesKey, set[str]] = {k: set() for k in keys}

    counts = work.groupby(GROUP_KEY + ["bucket"], sort=False).size()
    for index, count in counts.items():
        key = tuple(index[:-1])
        bucket = index[-1]
        row = bucket_index.get(bucket)
        col = key_index.get(key)
        if row is not None and col is not None:
            matrix[row, col] += float(count)

    for key, group in work.groupby(GROUP_KEY, sort=False):
        k = tuple(key)
        events[k] = int(len(group))
        devices[k] = set(zip(group["node"].fillna(""), group["node_alias"].fillna("")))
        regions[k] = Counter(r for r in group["region"].fillna("").astype(str) if r)
        severity[k] = int(group["severity"].max())
        systems[k] = {str(x) for x in group["system"].dropna().unique() if str(x)}
        integrations[k] = {str(x) for x in group["integration"].dropna().unique() if str(x)}

    intensity = pd.DataFrame(matrix, index=grid, columns=[series_id(k) for k in keys])
    half_life = max(float(ewma_half_life_seconds), 1e-6)
    alpha = float(1.0 - math.exp(-math.log(2.0) * step / half_life))
    alpha = min(max(alpha, 1e-4), 0.999)
    ewma = intensity.ewm(alpha=alpha, adjust=False).mean()

    maintenance_hits: list[dict] = []
    alarms_in_maintenance = 0
    if bundle.maintenance:
        for row in work.itertuples(index=False):
            hits = _window_hit(row.ts, str(row.region or ""), str(row.system or ""),
                               str(row.node or ""), bundle.maintenance)
            if hits:
                alarms_in_maintenance += 1
                for hit in hits:
                    if hit not in maintenance_hits:
                        maintenance_hits.append(hit)

    meta: dict[SeriesKey, dict] = {}
    for key in keys:
        active_bins = int((intensity[series_id(key)] > 0).sum())
        ewma_values = ewma[series_id(key)].to_numpy(dtype=float)
        meta[key] = dict(
            id=series_id(key),
            integration=key[0],
            system=key[1],
            alert_group=key[2],
            events=events[key],
            active_bins=active_bins,
            duty=round(active_bins / max(len(grid), 1), 4),
            devices=len(devices[key]),
            regions=sorted(regions[key]),
            systems=sorted(systems[key]),
            integrations=sorted(integrations[key]),
            severity_max=severity[key],
            peak_intensity=round(float(intensity[series_id(key)].max()), 4),
            peak_ewma=round(float(ewma_values.max()), 4),
            mean_ewma=round(float(ewma_values.mean()), 4),
        )

    warnings = []
    if series_warning:
        warnings.append(series_warning)
    if adjusted:
        warnings.append(
            f"step_seconds adjusted to {step}s to keep the signal matrix <= {max_bins} bins")
    if alarms_in_maintenance:
        warnings.append(
            f"{alarms_in_maintenance} alarms overlap maintenance window(s); "
            "they are kept in the graph and flagged for the runbook")
    if len(grid) < 5:
        warnings.append("too few time bins for a stable causal graph")

    return PreparedIncident(
        incident_id=bundle.incident_id,
        region=bundle.region,
        start=grid[0].to_pydatetime(),
        end=grid[-1].to_pydatetime(),
        step_seconds=step,
        ewma_alpha=round(alpha, 6),
        alarms=work,
        keys=keys,
        grid=grid,
        intensity=intensity,
        ewma=ewma,
        meta=meta,
        device_sets=devices,
        maintenance_hits=maintenance_hits,
        alarms_in_maintenance=alarms_in_maintenance,
        warnings=warnings,
        bundle=bundle,
    )


IncidentBundle = IncidentBundleIn

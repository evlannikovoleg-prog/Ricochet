from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pipeline import PreparedIncident, series_id


@dataclass
class ClusterConfig:
    method: str = "hdbscan"
    min_cluster_size: int = 3
    min_samples: int = 2
    n_clusters: int = 3
    top_k: int = 50
    random_state: int = 42


def _series_features(prepared: PreparedIncident) -> pd.DataFrame:
    rows = []
    for key in prepared.keys:
        sid = series_id(key)
        raw = prepared.intensity[sid].to_numpy(dtype=float)
        smooth = prepared.ewma[sid].to_numpy(dtype=float)
        mean = float(raw.mean())
        std = float(raw.std())
        bursts = int((raw > mean + max(std, 1e-9)).sum()) if std > 0 else 0
        rows.append(dict(
            id=sid,
            events=float(prepared.meta[key]["events"]),
            active_bins=float(prepared.meta[key]["active_bins"]),
            active_ratio=float(prepared.meta[key]["duty"]),
            peak_intensity=float(raw.max()),
            mean_intensity=mean,
            std_intensity=std,
            peak_ewma=float(smooth.max()),
            mean_ewma=float(smooth.mean()),
            ewma_area=float(smooth.sum()),
            bursts=float(bursts),
            devices=float(prepared.meta[key]["devices"]),
            severity_max=float(prepared.meta[key]["severity_max"]),
        ))
    return pd.DataFrame(rows).set_index("id")


def _simple_fallback(labels: np.ndarray, n: int, n_clusters: int) -> np.ndarray:
    # Deterministic fallback if scikit-learn is unavailable: split by activity.
    if n == 0:
        return labels
    order = np.argsort(-labels.astype(float))
    k = max(1, min(n_clusters, n))
    out = np.zeros(n, dtype=int)
    for cluster, chunk in enumerate(np.array_split(order, k)):
        out[chunk] = cluster
    return out


def _cluster_labels(features: pd.DataFrame, config: ClusterConfig) -> tuple[np.ndarray, str, list[str]]:
    warnings: list[str] = []
    n = len(features)
    if n == 0:
        return np.array([], dtype=int), "empty", warnings
    if n == 1:
        return np.array([0], dtype=int), "singleton", warnings

    values = features.fillna(0.0).to_numpy(dtype=float)
    spans = values.max(axis=0) - values.min(axis=0)
    values = np.divide(values, np.where(spans > 0, spans, 1.0))

    method = (config.method or "hdbscan").lower()
    try:
        from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
    except Exception as exc:  # pragma: no cover - depends on optional sklearn
        warnings.append(f"scikit-learn unavailable ({exc}); activity fallback used")
        return _simple_fallback(values.sum(axis=1), n, config.n_clusters), "activity-fallback", warnings

    if method in {"hdbscan", "auto"}:
        min_size = max(2, min(int(config.min_cluster_size), n))
        min_samples = max(1, min(int(config.min_samples), min_size))
        try:
            try:
                model = HDBSCAN(min_cluster_size=min_size, min_samples=min_samples,
                                copy=True)
            except TypeError:  # older scikit-learn without the copy keyword
                model = HDBSCAN(min_cluster_size=min_size, min_samples=min_samples)
            labels = model.fit_predict(values)
            if len(set(labels) - {-1}) >= 1:
                return labels.astype(int), "hdbscan", warnings
            warnings.append("hdbscan found no dense cluster; kmeans fallback used")
        except Exception as exc:
            warnings.append(f"hdbscan failed ({type(exc).__name__}); kmeans fallback used")
    elif method == "dbscan":
        try:
            labels = DBSCAN(eps=0.75, min_samples=max(1, int(config.min_samples))).fit_predict(values)
            if len(set(labels) - {-1}) >= 1:
                return labels.astype(int), "dbscan", warnings
            warnings.append("dbscan found no dense cluster; kmeans fallback used")
        except Exception as exc:
            warnings.append(f"dbscan failed ({type(exc).__name__}); kmeans fallback used")

    unique_rows = len(np.unique(np.round(values, 8), axis=0))
    k = max(1, min(int(config.n_clusters), n, unique_rows))
    labels = KMeans(n_clusters=k, n_init=10, random_state=config.random_state).fit_predict(values)
    return labels.astype(int), "kmeans", warnings


def cluster_series(prepared: PreparedIncident, config: ClusterConfig | None = None) -> dict:
    config = config or ClusterConfig()
    if prepared.empty:
        return dict(incident_id=prepared.incident_id, series_total=0, clusters=[],
                    noise=[], method_used="empty", warnings=list(prepared.warnings),
                    window=dict(start=prepared.start, end=prepared.end,
                                step_seconds=prepared.step_seconds))

    features = _series_features(prepared)
    labels, method, warnings = _cluster_labels(features, config)
    features = features.copy()
    features["cluster"] = labels

    # Per-series anomaly score: distance from the population centre in normalised
    # feature space. It is a ranking score, not a calibrated probability.
    values = features.drop(columns=["cluster"]).fillna(0.0).to_numpy(dtype=float)
    spans = values.max(axis=0) - values.min(axis=0)
    values = np.divide(values, np.where(spans > 0, spans, 1.0))
    centre = values.mean(axis=0)
    scale = values.std(axis=0)
    z = np.abs((values - centre) / np.where(scale > 0, scale, 1.0))
    scores = 1.0 - np.exp(-z.mean(axis=1) / 1.5)
    features["anomaly_score"] = np.round(scores, 4)

    clusters = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        members = features[features["cluster"] == label].sort_values(
            ["anomaly_score", "peak_ewma"], ascending=False)
        member_ids = list(members.index)
        keys = [prepared.key_of(sid) for sid in member_ids]
        systems = sorted({k[1] for k in keys if k})
        integrations = sorted({k[0] for k in keys if k})
        representative = member_ids[: min(10, len(member_ids))]
        clusters.append(dict(
            cluster_id=int(label),
            label=f"micro_storm_{label}",
            size=len(member_ids),
            series=representative,
            all_series=member_ids if config.top_k <= 0 else member_ids[:config.top_k],
            systems=systems,
            integrations=integrations,
            peak_ewma=round(float(members["peak_ewma"].max()), 4),
            anomaly_score=round(float(members["anomaly_score"].mean()), 4),
            mean_anomaly_score=round(float(members["anomaly_score"].mean()), 4),
            max_anomaly_score=round(float(members["anomaly_score"].max()), 4),
            events=round(float(members["events"].sum()), 4),
            is_micro_storm=bool(len(member_ids) >= 2),
        ))
    clusters.sort(key=lambda c: (-c["mean_anomaly_score"], -c["size"]))

    noise = []
    noisy = features[features["cluster"] == -1].sort_values("anomaly_score", ascending=False)
    for sid, row in noisy.head(max(config.top_k, 0) or len(noisy)).iterrows():
        key = prepared.key_of(sid)
        noise.append(dict(
            id=sid,
            system=key[1] if key else "",
            alert_group=key[2] if key else "",
            anomaly_score=round(float(row["anomaly_score"]), 4),
            peak_ewma=round(float(row["peak_ewma"]), 4),
            events=round(float(row["events"]), 4),
        ))

    return dict(
        incident_id=prepared.incident_id,
        series_total=len(prepared.keys),
        clusters=clusters,
        noise=noise,
        method_used=method,
        warnings=list(prepared.warnings) + warnings,
        window=dict(start=prepared.start, end=prepared.end,
                    step_seconds=prepared.step_seconds),
        features={sid: {k: round(float(v), 4) for k, v in row.items()}
                  for sid, row in features.iterrows()},
    )

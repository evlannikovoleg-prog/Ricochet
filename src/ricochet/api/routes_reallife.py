from __future__ import annotations

from fastapi import APIRouter, Query, Request

from ..reallife.causal import CausalConfig, build_causal_graph
from ..reallife.clustering import ClusterConfig, cluster_series
from ..reallife.pipeline import prepare_incident
from ..reallife.runbook import RunbookConfig
from ..reallife.schemas import IncidentBundleIn
from ..reallife.service import RealtimeOptions, RealtimeService

router = APIRouter(tags=["realtime"])


def _service(step_seconds: int, ewma_half_life_seconds: float, max_bins: int,
            max_series: int) -> RealtimeService:
    return RealtimeService(RealtimeOptions(
        step_seconds=step_seconds,
        ewma_half_life_seconds=ewma_half_life_seconds,
        max_bins=max_bins,
        max_series=max_series,
    ))


@router.post("/topology/causal-graph")
async def causal_graph(
    bundle: IncidentBundleIn,
    step_seconds: int = Query(10, ge=1, le=3600),
    ewma_half_life_seconds: float = Query(60.0, gt=0, le=86400),
    max_bins: int = Query(3000, ge=10, le=20000),
    max_series: int = Query(5000, ge=1, le=100000),
    max_lag_seconds: int = Query(300, ge=1, le=86400),
    min_abs_corr: float = Query(0.35, ge=0.0, le=1.0),
    min_causal_power: float = Query(0.30, ge=0.0, le=1.0),
    max_nodes: int = Query(30, ge=2, le=500),
    max_cond_size: int = Query(1, ge=0, le=3),
    max_edges: int = Query(200, ge=1, le=5000),
    remove_common_factor: bool = True,
    use_topology: bool = True,
    topology_strict: bool = False,
):
    prepared = prepare_incident(
        bundle, step_seconds=step_seconds,
        ewma_half_life_seconds=ewma_half_life_seconds, max_bins=max_bins,
        max_series=max_series)
    config = CausalConfig(
        max_lag_seconds=max_lag_seconds,
        min_abs_corr=min_abs_corr,
        min_causal_power=min_causal_power,
        max_nodes=max_nodes,
        max_cond_size=max_cond_size,
        max_edges=max_edges,
        remove_common_factor=remove_common_factor,
        use_topology=use_topology,
        topology_strict=topology_strict,
    )
    return build_causal_graph(prepared, config)


@router.post("/anomalies/clusters")
async def anomalies_clusters(
    bundle: IncidentBundleIn,
    step_seconds: int = Query(10, ge=1, le=3600),
    ewma_half_life_seconds: float = Query(60.0, gt=0, le=86400),
    max_bins: int = Query(3000, ge=10, le=20000),
    max_series: int = Query(5000, ge=1, le=100000),
    method: str = Query("hdbscan", pattern="^(hdbscan|dbscan|kmeans|auto)$"),
    min_cluster_size: int = Query(3, ge=2, le=1000),
    min_samples: int = Query(2, ge=1, le=1000),
    n_clusters: int = Query(3, ge=1, le=100),
    top_k: int = Query(50, ge=1, le=1000),
):
    prepared = prepare_incident(
        bundle, step_seconds=step_seconds,
        ewma_half_life_seconds=ewma_half_life_seconds, max_bins=max_bins,
        max_series=max_series)
    config = ClusterConfig(
        method=method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_clusters=n_clusters,
        top_k=top_k,
    )
    return cluster_series(prepared, config)


@router.post("/agent/runbook")
async def agent_runbook(
    request: Request,
    bundle: IncidentBundleIn,
    step_seconds: int = Query(10, ge=1, le=3600),
    ewma_half_life_seconds: float = Query(60.0, gt=0, le=86400),
    max_bins: int = Query(3000, ge=10, le=20000),
    max_series: int = Query(5000, ge=1, le=100000),
    max_lag_seconds: int = Query(300, ge=1, le=86400),
    min_abs_corr: float = Query(0.35, ge=0.0, le=1.0),
    min_causal_power: float = Query(0.30, ge=0.0, le=1.0),
    max_nodes: int = Query(30, ge=2, le=500),
    max_cond_size: int = Query(1, ge=0, le=3),
    top_k: int = Query(5, ge=1, le=50),
    min_confidence: float = Query(0.50, ge=0.0, le=1.0),
    use_topology: bool = True,
    topology_strict: bool = False,
):
    svc = request.app.state.svc
    service = _service(step_seconds, ewma_half_life_seconds, max_bins, max_series)
    prepared = prepare_incident(
        bundle, step_seconds=step_seconds,
        ewma_half_life_seconds=ewma_half_life_seconds, max_bins=max_bins,
        max_series=max_series)
    causal = build_causal_graph(prepared, CausalConfig(
        max_lag_seconds=max_lag_seconds,
        min_abs_corr=min_abs_corr,
        min_causal_power=min_causal_power,
        max_nodes=max_nodes,
        max_cond_size=max_cond_size,
        use_topology=use_topology,
        topology_strict=topology_strict,
    ))
    clusters = cluster_series(prepared, ClusterConfig())
    return service.runbook_agent.generate(
        prepared, causal, clusters,
        ranker=getattr(svc, "ranker", None),
        runbooks=getattr(svc, "runbooks", None),
        incidents=getattr(svc, "incidents", None),
        config=RunbookConfig(top_k=top_k, min_causal_power=min_causal_power,
                             min_confidence=min_confidence),
    )

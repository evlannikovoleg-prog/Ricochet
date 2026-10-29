from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..config import EngineConfig
from ..verdicts import shared_unshared_corr, overlap_stats
from ..canonical import group_device_sets
from .schemas import RunParams, VerifyPairIn

router = APIRouter(tags=["butterfly"])


def _svc(request: Request):
    return request.app.state.svc


@router.post("/butterfly/run")
async def run_butterfly(params: RunParams, request: Request):
    svc = _svc(request)
    payload = params.model_dump()
    job_id = svc.submit_job(lambda: run_pipeline_job(svc, payload), "butterfly_run")
    return dict(job_id=job_id, status=svc.jobs[job_id]["status"])


def run_pipeline_job(svc, params: dict) -> dict:
    from .main import run_pipeline
    return run_pipeline(svc, params)


@router.get("/butterfly/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    svc = _svc(request)
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return job


@router.get("/butterfly/pairs")
async def list_pairs(request: Request, cls: str = "", min_corr: float = 0.0,
                     min_score: float = 0.0, min_probability: float = 0.0,
                     decision: str = "", limit: int = 100):
    svc = _svc(request)
    if svc.butterfly_last is None:
        raise HTTPException(404, "no butterfly run yet: POST /api/v1/butterfly/run first")
    pairs = svc.butterfly_last.get("pairs", [])

    def ranker_of(p: dict) -> dict:
        return p.get("xgboost") or {}

    out = [p for p in pairs
           if (not cls or p.get("class") == cls)
           and abs(p.get("corr", 0)) >= min_corr
           and (p.get("score") or 0) >= min_score
           and (ranker_of(p).get("probability") or 0) >= min_probability
           and (not decision or ranker_of(p).get("decision") == decision)]
    return dict(total=len(pairs), filtered=len(out), pairs=out[:limit])


@router.get("/butterfly/graph")
async def graph(request: Request, min_score: float = 0.0, cls: str = ""):
    svc = _svc(request)
    if svc.butterfly_last is None:
        raise HTTPException(404, "no butterfly run yet")
    pairs = [p for p in svc.butterfly_last.get("pairs", [])
             if (p.get("score") or 0) >= min_score and (not cls or p.get("class") == cls)]
    nodes, seen = [], {}
    for p in pairs:
        for side in ("key_a", "key_b"):
            k = tuple(p[side])
            if k not in seen:
                seen[k] = True
                nodes.append(dict(
                    id=f"{k[0]}|{k[1]}|{k[2]}",
                    integration=k[0], system=k[1], alert_group=k[2],
                    devices=p.get("devices_a" if side == "key_a" else "devices_b"),
                ))
    edges = [dict(
        source=f"{p['key_a'][0]}|{p['key_a'][1]}|{p['key_a'][2]}",
        target=f"{p['key_b'][0]}|{p['key_b'][1]}|{p['key_b'][2]}",
        corr=p.get("corr"), lag_min=p.get("lag_min"), lead=p.get("lead"),
        verdict=p.get("class"), score=p.get("score"),
        days_ok=p.get("days_ok"), days_total=p.get("days_total"),
        regions_ok=p.get("regions_ok"), month_stable=p.get("month_stable"),
    ) for p in pairs]
    return dict(nodes=nodes, edges=edges)


@router.post("/butterfly/verify")
async def verify_pair(body: VerifyPairIn, request: Request):
    svc = _svc(request)
    ka = (body.key_a.integration, body.key_a.system, body.key_a.alert_group)
    kb = (body.key_b.integration, body.key_b.system, body.key_b.alert_group)
    cfg = EngineConfig()
    dev_sets = group_device_sets(svc.alarms)
    if ka not in dev_sets or kb not in dev_sets:
        raise HTTPException(404, "series not found in loaded alarms")
    ov = overlap_stats(dev_sets[ka], dev_sets[kb])
    su = shared_unshared_corr(svc.alarms, ka, kb, body.lag_bins, cfg)
    return dict(overlap=ov, shared_unshared=su)

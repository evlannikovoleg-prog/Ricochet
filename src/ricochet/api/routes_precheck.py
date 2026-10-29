from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..maintenance import MaintenanceWindow
from ..precheck.context import assemble_case
from .schemas import AlarmIn, IncidentIn, WindowIn, RunbookIn

router = APIRouter(tags=["precheck"])


def _svc(request: Request):
    return request.app.state.svc


@router.post("/precheck/incident")
async def precheck_incident(body: IncidentIn, request: Request):
    svc = _svc(request)
    incident = dict(
        incident_id=body.incident_id,
        received_at=body.received_at,
        region=body.region,
        alarms=[a.model_dump() for a in body.alarms],
    )
    pairs = svc.butterfly_last.get("pairs", []) if svc.butterfly_last else []
    last_ranker = (svc.butterfly_last or {}).get("ranker") or {}
    stale = (last_ranker.get("model_kind") != svc.ranker.model_kind
             or last_ranker.get("model_version") != svc.ranker.model_version)
    if pairs and (stale or any(not isinstance(p.get("xgboost"), dict) for p in pairs)):
        pairs, _ = svc.ranker.score_pairs(
            pairs, alarms=svc.alarms, cfg=svc.cfg.engine, include_features=False)
    pairs, ranker_meta = svc.ranker.select_for_precheck(pairs)
    case = assemble_case(
        incident,
        alarms_history=svc.alarms,
        pairs=pairs,
        maintenance=svc.maintenance,
        incidents=svc.incidents,
        runbooks=svc.runbooks,
    )
    case["ranker"] = ranker_meta
    return case


@router.post("/precheck/alarm")
async def precheck_alarm(alarm: AlarmIn, request: Request):
    svc = _svc(request)
    body = IncidentIn(
        incident_id="ad-hoc",
        region=alarm.region,
        alarms=[alarm],
    )
    return await precheck_incident(body, request)


@router.get("/precheck/runbooks")
async def list_runbooks(request: Request):
    svc = _svc(request)
    return dict(runbooks=svc.runbooks.list())


@router.post("/precheck/runbooks")
async def add_runbook(body: RunbookIn, request: Request):
    svc = _svc(request)
    svc.runbooks.add(body.model_dump())
    svc.runbooks.save(svc.cfg.state_dir / "runbooks.json")
    return dict(ok=True)


@router.get("/maintenance/windows")
async def list_windows(request: Request):
    svc = _svc(request)
    return dict(windows=svc.maintenance.list())


@router.post("/maintenance/windows")
async def add_window(body: WindowIn, request: Request):
    svc = _svc(request)
    w = MaintenanceWindow(**body.model_dump())
    svc.maintenance.add(w)
    return w.to_dict()


@router.delete("/maintenance/windows/{window_id}")
async def delete_window(window_id: str, request: Request):
    svc = _svc(request)
    if not svc.maintenance.remove(window_id):
        raise HTTPException(404, f"window {window_id} not found")
    return dict(ok=True)

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..leadlag import pair_id as make_pair_id
from .schemas import FeedbackIn, RankerTrainIn

router = APIRouter(tags=["ranker"])


def _svc(request: Request):
    return request.app.state.svc


def _scored_pairs(svc) -> list[dict]:
    pairs = svc.butterfly_last.get("pairs", []) if svc.butterfly_last else []
    if not pairs:
        return []
    if any(not isinstance(p.get("xgboost"), dict) for p in pairs):
        scored, _ = svc.ranker.score_pairs(
            pairs, alarms=svc.alarms, cfg=svc.cfg.engine, include_features=True)
        return scored
    return pairs


def _find_pair(pairs: list[dict], target: str) -> dict | None:
    for pair in pairs:
        if pair.get("id") == target:
            return pair
        try:
            if make_pair_id(pair) == target:
                return pair
        except KeyError:
            continue
    return None


@router.get("/ranker/status")
async def ranker_status(request: Request):
    return _svc(request).ranker.status()


@router.post("/ranker/feedback")
async def ranker_feedback(body: FeedbackIn, request: Request):
    svc = _svc(request)
    pairs = _scored_pairs(svc)
    pair = _find_pair(pairs, body.pair_id)
    if pair is None:
        raise HTTPException(404, f"pair {body.pair_id} not found in the last butterfly run")
    try:
        result = svc.ranker.add_feedback(
            body.pair_id, body.label, pair=pair,
            source=body.source, comment=body.comment)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result


@router.post("/ranker/train")
async def ranker_train(body: RankerTrainIn, request: Request):
    svc = _svc(request)
    try:
        result = svc.ranker.train(force=body.force)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    return dict(result=result, status=svc.ranker.status(include_features=False))

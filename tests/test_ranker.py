from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ricochet.config import EngineConfig
from ricochet.canonical import validate_alarms
from ricochet.ingest.synthetic import generate_synthetic
from ricochet.leadlag import run_butterfly
from ricochet.ranking import FEATURE_NAMES, PairRanker, RankerConfig
from ricochet.verdicts import finalize_pairs


@pytest.fixture(scope="module")
def world():
    df, meta = generate_synthetic(days=20, seed=7)
    df = validate_alarms(df)
    cfg = EngineConfig(min_events=100, min_bins=30, min_support=10, min_abs_corr=0.3,
                       lags=6, min_devices=3, hub_degree=40)
    res = run_butterfly(df, cfg, holdout_last=0)
    pairs = finalize_pairs(df, res["pairs"], cfg)
    return df, meta, cfg, pairs


def _is_butterfly(pair, meta) -> bool:
    target = {tuple(meta["butterfly"]["key_a"]), tuple(meta["butterfly"]["key_b"])}
    return {tuple(pair["key_a"]), tuple(pair["key_b"])} == target


def test_features_and_cold_start_ranking(world):
    df, meta, cfg, pairs = world
    ranker = PairRanker(RankerConfig(state_dir=Path(tempfile.mkdtemp(prefix="ranker-cold-"))))
    scored, summary = ranker.score_pairs(pairs, alarms=df, cfg=cfg)

    assert summary["model_kind"] == "heuristic"
    assert summary["context_available"] is True
    assert [p["xgboost"]["rank"] for p in scored] == list(range(1, len(scored) + 1))
    assert set(scored[0]["xgboost"]["features"]) == set(FEATURE_NAMES)
    probs = [p["xgboost"]["probability"] for p in scored]
    assert probs == sorted(probs, reverse=True)
    assert any(_is_butterfly(p, meta) for p in scored if p["xgboost"]["decision"] == "pass")

    selected, meta_sel = ranker.select_for_precheck(scored, top_k=5)
    assert len(selected) <= 5
    assert meta_sel["selected"] == len(selected)
    assert meta_sel["candidates"] == len(pairs)


def test_feedback_training_and_persistence(world):
    pytest.importorskip("xgboost")
    df, meta, cfg, pairs = world
    state = Path(tempfile.mkdtemp(prefix="ranker-model-"))
    config = RankerConfig(state_dir=state, auto_train=False, n_estimators=60)
    ranker = PairRanker(config)
    scored, _ = ranker.score_pairs(pairs, alarms=df, cfg=cfg)

    bf = {tuple(meta["butterfly"]["key_a"]), tuple(meta["butterfly"]["key_b"])}
    for pair in scored:
        keys = {tuple(pair["key_a"]), tuple(pair["key_b"])}
        ranker.add_feedback(pair["id"], "good" if keys == bf else "spam",
                            pair=pair, source="test")

    result = ranker.train(force=True)
    assert result["trained"] is True
    assert result["metrics"]["samples"] == len(scored)
    assert result["metrics"]["auc"] is not None

    ranked, _ = ranker.score_pairs(pairs, alarms=df, cfg=cfg)
    assert ranked[0]["xgboost"]["model_kind"] == "xgboost"
    assert _is_butterfly(ranked[0], meta), "xgboost should put the planted butterfly first"

    reloaded = PairRanker(RankerConfig(state_dir=state, auto_train=False))
    assert reloaded.model_available is True
    rescored, _ = reloaded.score_pairs(pairs, alarms=df, cfg=cfg)
    assert rescored[0]["xgboost"]["model_kind"] == "xgboost"
    assert rescored[0]["xgboost"]["probability"] == pytest.approx(
        ranked[0]["xgboost"]["probability"], abs=1e-6)


def test_duplicate_gets_lower_heuristic_than_butterfly(world):
    df, meta, cfg, pairs = world
    ranker = PairRanker(RankerConfig(state_dir=Path(tempfile.mkdtemp(prefix="ranker-dup-"))))
    scored, _ = ranker.score_pairs(pairs, alarms=df, cfg=cfg)
    butterfly = next(p for p in scored if _is_butterfly(p, meta))
    duplicate = next((p for p in scored if p.get("class") == "DUPLICATE"), None)
    if duplicate is None:
        pytest.skip("duplicate pair is not in this candidate set")
    assert butterfly["xgboost"]["probability"] > duplicate["xgboost"]["probability"]

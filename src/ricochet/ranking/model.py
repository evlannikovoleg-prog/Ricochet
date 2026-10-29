from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    FEATURE_NAMES,
    PairFeatureContext,
    extract_features,
    features_frame,
    features_from_json,
    json_features,
)
from .feedback import FeedbackStore

HEURISTIC_VERSION = "heuristic-v1"


class RankerUnavailable(RuntimeError):
    pass


def xgboost_installed() -> bool:
    return importlib.util.find_spec("xgboost") is not None


@dataclass
class RankerConfig:
    state_dir: Path
    model_name: str = "ranker_model.json"
    meta_name: str = "ranker_model.meta.json"
    feedback_name: str = "ranker_feedback.json"
    min_confidence: float = 0.9
    review_confidence: float = 0.5
    precheck_top_k: int = 5
    min_train_samples: int = 20
    min_per_class: int = 2
    retrain_every: int = 10
    auto_train: bool = True
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = 2

    @property
    def model_path(self) -> Path:
        return self.state_dir / self.model_name

    @property
    def meta_path(self) -> Path:
        return self.state_dir / self.meta_name

    @property
    def feedback_path(self) -> Path:
        return self.state_dir / self.feedback_name


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _finite(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def heuristic_probability(features: dict[str, float]) -> float:
    """Cold-start score used until enough expert feedback is collected.

    It is intentionally transparent and conservative: the XGBoost model replaces
    it as soon as labels are available. The score is not a statistically calibrated
    probability; it is a ranking prior.
    """

    def f(name: str, default: float = 0.0) -> float:
        return _finite(features.get(name, float("nan")), default)

    regions = f("regions_frac", 0.5)
    lift = min(f("cooccurrence_lift_log"), 3.0)
    z = (
        -3.0
        + 3.2 * f("corr_abs")
        + 1.4 * max(f("lead"), 0.0)
        + 1.8 * f("days_frac")
        + 0.8 * regions
        + 0.8 * f("month_stable")
        + 0.6 * f("holdout_ok")
        + 0.9 * f("class_butterfly")
        + 0.2 * f("class_cross_object")
        - 0.5 * f("class_ambiguous")
        - 2.5 * f("class_duplicate")
        + 0.7 * f("topology_site_affinity")
        + 0.4 * f("topology_token_jaccard")
        + 0.3 * f("topology_region_affinity")
        + 0.35 * lift
        + 0.7 * f("temporal_jaccard")
        + 0.8 * f("score_engine")
        + 0.25 * f("support_log") / math.log1p(100.0)
        - 2.0 * f("overlap_frac")
        - 1.0 * f("corr_shared_abs", 0.0)
    )
    return float(_sigmoid(z))


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    pos = np.sort(pos)
    neg = np.sort(neg)
    # Mann-Whitney U with ties counted as 0.5.
    greater = np.searchsorted(neg, pos, side="left")
    greater_eq = np.searchsorted(neg, pos, side="right")
    wins = greater.sum() + 0.5 * (greater_eq - greater).sum()
    return float(wins / (len(pos) * len(neg)))


def _version_of(records: list[dict], trained_at: str) -> str:
    payload = json.dumps(
        [{"pair_id": r.get("pair_id"), "label": r.get("label"), "features": r.get("features")}
         for r in records],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha1(f"{payload}|{trained_at}".encode("utf-8")).hexdigest()[:12]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


class PairRanker:
    """XGBoost post-filter over butterfly candidates.

    Responsibilities:
    - turn a pair into a fixed feature vector;
    - score/rank all candidates without touching the statistical engine;
    - collect expert feedback and retrain the local model;
    - select a small high-confidence set for precheck.

    When no model/feedback exists yet, the ranker transparently falls back to a
    deterministic heuristic. This keeps the service useful from the first run.
    """

    def __init__(self, state_dir: str | Path | RankerConfig, config: RankerConfig | None = None):
        if isinstance(state_dir, RankerConfig):
            config = state_dir
            state_dir = config.state_dir
        if config is None:
            config = RankerConfig(state_dir=Path(state_dir))
        self.config = config
        self.config.state_dir = Path(self.config.state_dir)
        self.feedback = FeedbackStore(self.config.feedback_path)
        self._lock = threading.RLock()
        self._model = None
        self._meta: dict | None = None
        self._load_error: str | None = None
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self.config.meta_path.exists():
            try:
                self._meta = json.loads(self.config.meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._meta = None
        if not self.config.model_path.exists():
            return
        model_features = self._meta.get("feature_names") if self._meta else None
        if model_features and list(model_features) != FEATURE_NAMES:
            self._load_error = "feature contract changed: retrain the model"
            return
        if not xgboost_installed():
            self._load_error = "xgboost is not installed"
            return
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(self.config.model_path))
            self._model = model
            self._load_error = None
        except Exception as exc:  # pragma: no cover - depends on external model file
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._model = None

    @property
    def model_available(self) -> bool:
        return self._model is not None

    @property
    def model_kind(self) -> str:
        return "xgboost" if self._model is not None else "heuristic"

    @property
    def model_version(self) -> str:
        if self._model is not None and self._meta:
            return str(self._meta.get("version", "unknown"))
        return HEURISTIC_VERSION

    # ------------------------------------------------------------------ scoring
    def _decision(self, probability: float) -> str:
        if probability >= self.config.min_confidence:
            return "pass"
        if probability >= self.config.review_confidence:
            return "review"
        return "drop"

    def score_pairs(self, pairs: list[dict], alarms: pd.DataFrame | None = None,
                    cfg=None, *, sort: bool = True,
                    include_features: bool = True) -> tuple[list[dict], dict]:
        """Annotate pairs with ``xgboost`` metadata and return a run summary."""
        if not pairs:
            return [], dict(model_kind=self.model_kind, model_version=self.model_version,
                            scored=0, pass_=0, review=0, drop=0,
                            threshold=self.config.min_confidence,
                            features=FEATURE_NAMES)

        context = PairFeatureContext.build(alarms, pairs, cfg)
        rows = [extract_features(pair, context) for pair in pairs]
        probabilities = None
        kind, version = "heuristic", HEURISTIC_VERSION
        if self._model is not None:
            try:
                frame = features_frame(rows)
                with self._lock:
                    probabilities = self._model.predict_proba(frame)[:, 1]
                kind, version = "xgboost", self.model_version
            except Exception as exc:  # corrupted/foreign model must not break detection
                self._load_error = f"prediction failed: {type(exc).__name__}: {exc}"
                probabilities = None
        if probabilities is None:
            probabilities = np.array([heuristic_probability(row) for row in rows], dtype=float)
            kind, version = "heuristic", HEURISTIC_VERSION

        scored: list[dict] = []
        for pair, row, probability in zip(pairs, rows, probabilities):
            item = dict(pair)
            probability = float(min(max(probability, 0.0), 1.0))
            meta = dict(
                probability=round(probability, 4),
                confidence=round(probability, 4),
                decision=self._decision(probability),
                model_kind=kind,
                model_version=version,
                threshold=self.config.min_confidence,
            )
            if include_features:
                meta["features"] = json_features(row)
            item["xgboost"] = meta
            scored.append(item)

        if sort:
            scored.sort(key=lambda p: (-(p["xgboost"].get("probability") or 0.0),
                                       -(p.get("score") or 0.0)))
        for rank, item in enumerate(scored, 1):
            item["xgboost"]["rank"] = rank

        decisions = [p["xgboost"]["decision"] for p in scored]
        summary = dict(
            model_kind=kind,
            model_version=version,
            scored=len(scored),
            pass_=decisions.count("pass"),
            review=decisions.count("review"),
            drop=decisions.count("drop"),
            threshold=self.config.min_confidence,
            max_confidence=max((p["xgboost"]["probability"] for p in scored), default=0.0),
            context_available=context.available,
            features=FEATURE_NAMES,
        )
        return scored, summary

    def select_for_precheck(self, pairs: list[dict], *, min_confidence: float | None = None,
                            top_k: int | None = None, strict: bool = False
                            ) -> tuple[list[dict], dict]:
        """Return the small ranked set that is allowed to reach the AI agent.

        With a trained model the threshold is strict. If no pair crosses it, the
        caller gets an explicit fallback (unless ``strict=True``) so an incident is
        never left without candidates because of a cold/empty model.
        """
        threshold = self.config.min_confidence if min_confidence is None else float(min_confidence)
        top_k = self.config.precheck_top_k if top_k is None else int(top_k)
        scored = [p for p in pairs if isinstance(p.get("xgboost"), dict)
                  and "probability" in p["xgboost"]]
        if len(scored) != len(pairs):
            scored, _ = self.score_pairs(pairs, sort=True, include_features=False)
        else:
            scored = sorted(scored, key=lambda p: -(p["xgboost"].get("probability") or 0.0))

        high = [p for p in scored if (p["xgboost"].get("probability") or 0.0) >= threshold]
        fallback = False
        if high:
            selected = high[:top_k]
        elif strict:
            selected = []
        else:
            selected = scored[:top_k]
            fallback = True
        meta = dict(
            threshold=threshold,
            top_k=top_k,
            candidates=len(pairs),
            high_confidence=len(high),
            selected=len(selected),
            fallback=fallback,
            model_kind=self.model_kind,
            model_version=self.model_version,
        )
        return selected, meta

    # ------------------------------------------------------------------ feedback
    def add_feedback(self, pair_id: str, label, *, pair: dict | None = None,
                     features: dict | None = None, source: str = "ui",
                     comment: str = "") -> dict:
        if features is None and pair and isinstance(pair.get("xgboost"), dict):
            features = pair["xgboost"].get("features")
        if features is None:
            if pair is None:
                raise ValueError("pair or features are required")
            features = json_features(extract_features(pair))
        record = self.feedback.upsert(pair_id, label, features, source=source,
                                      comment=comment, pair=pair)
        auto = None
        if self.config.auto_train:
            summary = self.feedback.summary()
            meta_samples = int(((self._meta or {}).get("metrics") or {}).get("samples", 0))
            enough = (summary["total"] >= self.config.min_train_samples
                      and summary["positive"] >= self.config.min_per_class
                      and summary["negative"] >= self.config.min_per_class)
            due = self._model is None or summary["total"] - meta_samples >= self.config.retrain_every
            if enough and due:
                try:
                    auto = self.train()
                except Exception as exc:  # feedback must be saved even if retraining fails
                    auto = dict(trained=False, reason=f"{type(exc).__name__}: {exc}")
        return dict(feedback=record, auto_train=auto, status=self.status(include_features=False))

    # ------------------------------------------------------------------ training
    def train(self, *, force: bool = False) -> dict:
        if not xgboost_installed():
            raise RankerUnavailable("xgboost is not installed: pip install -r requirements.txt")
        records = self.feedback.records()
        if not records:
            return dict(trained=False, reason="no feedback records")

        rows = [features_from_json(r.get("features", {})) for r in records]
        y = np.array([int(r.get("label", 0)) for r in records], dtype=int)
        positive, negative = int((y == 1).sum()), int((y == 0).sum())
        if not force:
            if len(y) < self.config.min_train_samples:
                return dict(trained=False, reason="not enough feedback",
                            samples=len(y), positive=positive, negative=negative,
                            min_samples=self.config.min_train_samples)
            if positive < self.config.min_per_class or negative < self.config.min_per_class:
                return dict(trained=False, reason="both classes are required",
                            samples=len(y), positive=positive, negative=negative,
                            min_per_class=self.config.min_per_class)
        if positive == 0 or negative == 0:
            return dict(trained=False, reason="both classes are required",
                        samples=len(y), positive=positive, negative=negative)

        frame = features_frame(rows)
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_lambda=self.config.reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        model.fit(frame, y)
        probabilities = model.predict_proba(frame)[:, 1]
        trained_at = datetime.now().isoformat(timespec="seconds")
        version = _version_of(records, trained_at)
        importance = sorted(
            ({"feature": name, "importance": round(float(value), 4)}
             for name, value in zip(FEATURE_NAMES, model.feature_importances_)),
            key=lambda item: -item["importance"],
        )
        metrics = dict(
            samples=len(y),
            positive=positive,
            negative=negative,
            auc=_roc_auc(y, probabilities),
            accuracy=float(((probabilities >= 0.5).astype(int) == y).mean()),
            high_confidence_accuracy=float(
                ((probabilities >= self.config.min_confidence).astype(int) == y)[
                    probabilities >= self.config.min_confidence].mean()
            ) if (probabilities >= self.config.min_confidence).any() else None,
            high_confidence=float((probabilities >= self.config.min_confidence).mean()),
        )
        meta = dict(
            version=version,
            trained_at=trained_at,
            model_kind="xgboost",
            label_source="expert-feedback",
            feature_names=FEATURE_NAMES,
            threshold=self.config.min_confidence,
            xgboost_version=getattr(xgb, "__version__", "unknown"),
            importances=importance,
            metrics=metrics,
        )
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(self.config.model_path))
        _atomic_json(self.config.meta_path, meta)
        with self._lock:
            self._model = model
            self._meta = meta
            self._load_error = None
        return dict(trained=True, version=version, metrics=metrics,
                    top_features=importance[:10], model_path=str(self.config.model_path))

    # ------------------------------------------------------------------ status
    def status(self, *, include_features: bool = True) -> dict:
        feedback = self.feedback.summary()
        payload = dict(
            model_kind=self.model_kind,
            model_version=self.model_version,
            model_available=self.model_available,
            xgboost_installed=xgboost_installed(),
            load_error=self._load_error,
            threshold=self.config.min_confidence,
            review_threshold=self.config.review_confidence,
            precheck_top_k=self.config.precheck_top_k,
            feedback=feedback,
            feature_count=len(FEATURE_NAMES),
            trained_at=(self._meta or {}).get("trained_at"),
            metrics=(self._meta or {}).get("metrics"),
            top_features=(self._meta or {}).get("importances", [])[:10],
        )
        if include_features:
            payload["features"] = FEATURE_NAMES
        return payload

    def reset_model(self) -> None:
        with self._lock:
            self._model = None
            self._meta = None
            self._load_error = None
        for path in (self.config.model_path, self.config.meta_path):
            if path.exists():
                path.unlink()

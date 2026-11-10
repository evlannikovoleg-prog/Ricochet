from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..config import EngineConfig
from ..leadlag import pair_id as make_pair_id
from ..verdicts import overlap_stats
from .pipeline import PreparedIncident, series_id


@dataclass
class RunbookConfig:
    top_k: int = 5
    min_causal_power: float = 0.30
    min_confidence: float = 0.50
    max_steps: int = 8


class RunbookAgent:
    """Small RAG-style orchestrator over causal graph, clusters and XGBoost.

    The default generator is deterministic and local. Pass ``llm`` (a callable
    accepting the prompt and returning text) to replace only the narrative while
    keeping the same structured evidence and safety checks.
    """

    def __init__(self, llm: Callable[[str], str] | None = None):
        self.llm = llm

    def _pseudo_pairs(self, prepared: PreparedIncident, causal: dict) -> list[dict]:
        by_id = {series_id(key): key for key in prepared.keys}
        pairs: list[dict] = []
        for edge in causal.get("edges", []):
            ka, kb = by_id.get(edge["from"]), by_id.get(edge["to"])
            if ka is None or kb is None:
                continue
            ma, mb = prepared.meta[ka], prepared.meta[kb]
            overlap = overlap_stats(prepared.device_sets.get(ka, set()),
                                    prepared.device_sets.get(kb, set()))
            if ka[2] == kb[2] and overlap["common"] > 0:
                cls = "DUPLICATE"
            elif overlap["verdict"] == "NO_OVERLAP":
                cls = "BUTTERFLY_CANDIDATE"
            else:
                cls = "CROSS_OBJECT"
            corr = float(edge.get("corr") or edge.get("causal_power") or 0.0)
            pair = dict(
                id=make_pair_id({"key_a": ka, "key_b": kb}),
                key_a=ka,
                key_b=kb,
                corr=round(corr, 4),
                corr0=0.0,
                lead=round(abs(corr), 4),
                lag_bins=int(edge.get("lag_bins", 1)),
                lag_min=round(float(edge.get("lag_seconds", 0)) / 60.0, 3),
                support=int(min(ma["active_bins"], mb["active_bins"])),
                days_total=1,
                days_ok=1,
                med_corr=round(corr, 4),
                duty_a=ma["duty"],
                duty_b=mb["duty"],
                devices_a=ma["devices"],
                devices_b=mb["devices"],
                regions_total=0,
                regions_ok=0,
                regions_frac=None,
                overlap=overlap,
                shared_unshared=dict(corr_all=round(corr, 4), corr_shared=None,
                                     corr_unshared=round(corr, 4),
                                     verdict="NO_COMMON_DEVICES" if not overlap["common"] else "AMBIGUOUS"),
                class_=cls,
                month_stable=None,
                holdout=dict(holdout_corr=None, holdout_ok=None),
                score=round(float(edge.get("causal_power", 0.0)), 4),
                **{"class": cls},
            )
            pairs.append(pair)
        return pairs

    def _rank(self, prepared: PreparedIncident, causal: dict, ranker) -> tuple[list[dict], dict]:
        pairs = self._pseudo_pairs(prepared, causal)
        if not pairs:
            return [], dict(model_kind="none", model_version=None, threshold=None,
                            scored=0, max_confidence=0.0)
        if ranker is None:
            for pair in pairs:
                pair["xgboost"] = dict(
                    probability=pair["score"], confidence=pair["score"],
                    decision="pass" if pair["score"] >= 0.5 else "review",
                    model_kind="causal-only", model_version="causal_power",
                    threshold=0.5, rank=0)
            pairs.sort(key=lambda p: -p["xgboost"]["probability"])
            for rank, pair in enumerate(pairs, 1):
                pair["xgboost"]["rank"] = rank
            return pairs, dict(model_kind="causal-only", model_version="causal_power",
                               threshold=0.5, scored=len(pairs),
                               max_confidence=pairs[0]["xgboost"]["probability"])
        try:
            cfg = EngineConfig(bin_min=max(1, prepared.step_seconds // 60))
            scored, summary = ranker.score_pairs(
                pairs, alarms=prepared.alarms, cfg=cfg, include_features=False)
            return scored, summary
        except Exception as exc:
            for pair in pairs:
                pair["xgboost"] = dict(
                    probability=pair["score"], confidence=pair["score"],
                    decision="review", model_kind="causal-only",
                    model_version=f"ranker-error:{type(exc).__name__}",
                    threshold=0.5, rank=0)
            pairs.sort(key=lambda p: -p["xgboost"]["probability"])
            for rank, pair in enumerate(pairs, 1):
                pair["xgboost"]["rank"] = rank
            return pairs, dict(model_kind="causal-only", model_version="ranker-error",
                               threshold=0.5, scored=len(pairs),
                               max_confidence=pairs[0]["xgboost"]["probability"],
                               error=str(exc))

    def generate(self, prepared: PreparedIncident, causal: dict, clusters: dict,
                 *, ranker=None, runbooks=None, incidents=None,
                 config: RunbookConfig | None = None) -> dict:
        config = config or RunbookConfig()
        ranked, ranker_summary = self._rank(prepared, causal, ranker)
        top = [pair for pair in ranked
               if pair["score"] >= config.min_causal_power][:config.top_k]
        if not top:
            top = ranked[:config.top_k]

        roots: dict[tuple, dict] = {}
        for pair in top:
            lag = int(pair.get("lag_bins", 0))
            cause_key = pair["key_a"] if lag >= 0 else pair["key_b"]
            effect_key = pair["key_b"] if lag >= 0 else pair["key_a"]
            prob = float(pair["xgboost"].get("probability") or 0.0)
            item = roots.setdefault(cause_key, dict(
                node=series_id(cause_key), system=cause_key[1],
                integration=cause_key[0], alert_group=cause_key[2],
                probability=0.0, effects=[], evidence=[]))
            item["probability"] = max(item["probability"], prob)
            item["effects"].append(series_id(effect_key))
            item["evidence"].append(dict(from_=series_id(cause_key),
                                         to=series_id(effect_key),
                                         probability=prob,
                                         causal_power=pair.get("score"),
                                         lag_seconds=int(pair.get("lag_min", 0) * 60)))
        root_list = sorted(roots.values(), key=lambda r: -r["probability"])[:config.top_k]
        for root in root_list:
            root["probability"] = round(float(root["probability"]), 4)
            root["effects"] = sorted(set(root["effects"]))[:8]

        steps: list[dict] = []
        matched_runbooks: list[dict] = []
        if runbooks is not None:
            for root in root_list[:3]:
                entry = runbooks.match(root["system"], root["alert_group"])
                if not entry:
                    continue
                matched_runbooks.append(dict(id=entry.get("id"), system=entry.get("system"),
                                             alert_group=entry.get("alert_group"),
                                             url=entry.get("url"), owner=entry.get("owner")))
                for action in entry.get("steps", []):
                    steps.append(dict(order=0, action=action, source=f"runbook:{entry.get('id')}",
                                      runbook_id=entry.get("id")))

        for root in root_list[:3]:
            meta = prepared.meta.get((root["integration"], root["system"], root["alert_group"]), {})
            steps.append(dict(
                order=0,
                action=(f"Проверить {root['system']} / {root['alert_group']}: "
                        f"{meta.get('events', 0)} событий, {meta.get('devices', 0)} устройств, "
                        f"peak EWMA {meta.get('peak_ewma', 0)}."),
                source="generated", runbook_id=None))
            steps.append(dict(
                order=0,
                action=(f"Сверить с CMDB-топологией и соседними узлами; "
                        f"проверить, не является ли {root['node']} общим предком."),
                source="generated", runbook_id=None))

        if clusters.get("clusters"):
            top_cluster = clusters["clusters"][0]
            steps.append(dict(
                order=0,
                action=(f"Проверить микро-шторм {top_cluster['label']}: "
                        f"{top_cluster['size']} рядов, системы {', '.join(top_cluster['systems'][:5])}."),
                source="clustering", runbook_id=None))
        if prepared.maintenance_hits:
            steps.append(dict(
                order=0,
                action=(f"Учесть {len(prepared.maintenance_hits)} окно(а) обслуживания: "
                        "часть сигналов может быть плановой."),
                source="maintenance", runbook_id=None))
        steps.append(dict(
            order=0,
            action="Подтвердить гипотезу на следующем окне и исторических инцидентах; "
                   "при подтверждении — зафиксировать feedback для XGBoost.",
            source="generated", runbook_id=None))

        seen, unique_steps = set(), []
        for step in steps:
            key = step["action"]
            if key in seen:
                continue
            seen.add(key)
            step["order"] = len(unique_steps) + 1
            unique_steps.append(step)
            if len(unique_steps) >= config.max_steps:
                break

        similar = []
        if incidents is not None and prepared.keys:
            profile = sorted({(key[1], key[2]) for key in prepared.keys})
            similar = incidents.search(profile, region=prepared.region, top_k=3)

        confidence = max((r["probability"] for r in root_list), default=0.0)
        warnings = list(prepared.warnings) + list(causal.get("warnings", [])) + list(clusters.get("warnings", []))
        if confidence < config.min_confidence:
            warnings.append(
                f"low confidence ({confidence:.2f} < {config.min_confidence}); treat as hypothesis, not diagnosis")
        if not root_list:
            warnings.append("no ranked root causes; extend the window or lower thresholds")

        if root_list:
            summary = (
                f"Наиболее вероятная первопричина: {root_list[0]['system']} / "
                f"{root_list[0]['alert_group']} (p={root_list[0]['probability']:.2f}). "
                f"Далее: " + ", ".join(
                    f"{r['system']}/{r['alert_group']}" for r in root_list[1:4]) + "."
            )
        else:
            summary = "Причинный граф не дал устойчивых лидеров. Нужен более длинный интервал или ручная проверка."

        prompt_context = dict(
            incident_id=prepared.incident_id,
            summary=summary,
            roots=root_list,
            steps=unique_steps,
            evidence=dict(causal=dict(nodes=causal.get("nodes", []), edges=causal.get("edges", [])),
                          clusters=clusters.get("clusters", [])[:3],
                          ranker=ranker_summary),
        )
        llm_text = None
        if self.llm is not None:
            try:
                llm_text = self.llm(
                    "Сформируй краткий runbook для инженера по structured context:\n"
                    + str(prompt_context))
            except Exception as exc:
                warnings.append(f"llm generator failed: {type(exc).__name__}: {exc}")

        return dict(
            incident_id=prepared.incident_id,
            summary=summary,
            confidence=round(float(confidence), 4),
            root_causes=root_list,
            steps=unique_steps,
            runbooks=matched_runbooks,
            similar_incidents=similar,
            evidence=dict(
                causal=dict(nodes=len(causal.get("nodes", [])),
                            edges=len(causal.get("edges", [])),
                            top_edges=causal.get("edges", [])[:5],
                            topological_order=causal.get("topological_order", [])),
                clusters=clusters.get("clusters", [])[:3],
                ranker=ranker_summary,
                maintenance=prepared.maintenance_hits,
            ),
            window=dict(start=prepared.start, end=prepared.end,
                        step_seconds=prepared.step_seconds),
            generated_by="local-rules-v1" if llm_text is None else "llm+local-rules",
            llm_text=llm_text,
            warnings=warnings,
        )

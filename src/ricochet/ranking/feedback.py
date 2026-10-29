from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

LABEL_GOOD = "good"
LABEL_SPAM = "spam"


def normalize_label(value) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "good", "like", "find", "positive", "полезно", "находка"}:
        return 1
    if text in {"0", "false", "no", "spam", "bad", "negative", "спам", "ложное"}:
        return 0
    raise ValueError(f"unknown feedback label: {value!r}")


class FeedbackStore:
    """Small append/upsert store for expert labels.

    Features are snapshotted at feedback time, so a later graph rebuild does not
    invalidate the training set.
    """

    def __init__(self, path: str | Path, max_records: int = 5000):
        self.path = Path(path)
        self.max_records = max_records
        self._lock = threading.RLock()
        self._records: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        records = payload.get("records", []) if isinstance(payload, dict) else payload
        for record in records:
            pair_id = str(record.get("pair_id", ""))
            if pair_id:
                self._records[pair_id] = record

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = list(self._records.values())
        if len(records) > self.max_records:
            records = records[-self.max_records:]
            self._records = {r["pair_id"]: r for r in records}
        payload = dict(updated_at=datetime.now().isoformat(timespec="seconds"),
                       records=records)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def upsert(self, pair_id: str, label, features: dict, *, source: str = "ui",
               comment: str = "", pair: dict | None = None) -> dict:
        with self._lock:
            record = dict(
                pair_id=str(pair_id),
                label=normalize_label(label),
                features=dict(features),
                source=str(source or "ui"),
                comment=str(comment or ""),
                updated_at=datetime.now().isoformat(timespec="seconds"),
            )
            if pair:
                record["pair"] = dict(
                    key_a=pair.get("key_a"),
                    key_b=pair.get("key_b"),
                    corr=pair.get("corr"),
                    lag_min=pair.get("lag_min"),
                    class_=pair.get("class"),
                    score=pair.get("score"),
                )
            self._records[record["pair_id"]] = record
            self._save()
            return record

    def records(self) -> list[dict]:
        with self._lock:
            return list(self._records.values())

    def summary(self) -> dict:
        with self._lock:
            records = list(self._records.values())
        positive = sum(1 for r in records if int(r.get("label", 0)) == 1)
        by_source: dict[str, int] = {}
        for record in records:
            source = str(record.get("source", "ui"))
            by_source[source] = by_source.get(source, 0) + 1
        return dict(
            total=len(records),
            positive=positive,
            negative=len(records) - positive,
            by_source=by_source,
            last_updated=max((r.get("updated_at", "") for r in records), default=None),
        )

    def clear(self) -> None:
        with self._lock:
            self._records = {}
            self._save()

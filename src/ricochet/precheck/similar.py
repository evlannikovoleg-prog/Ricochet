from __future__ import annotations

import json
from pathlib import Path


class HistoricalIncidents:
    def __init__(self, incidents: list[dict] | None = None):
        self.incidents: list[dict] = incidents or []

    @staticmethod
    def from_file(path: str | Path) -> "HistoricalIncidents":
        p = Path(path)
        if not p.exists():
            return HistoricalIncidents([])
        return HistoricalIncidents(json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.incidents, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @staticmethod
    def _keys_set(inc: dict) -> set[tuple[str, str]]:
        return {(k.get("system", ""), k.get("alert_group", "")) for k in inc.get("keys", [])}

    def search(self, keys: list[tuple[str, str]] | set, region: str = "",
               top_k: int = 5, min_sim: float = 0.1) -> list[dict]:
        q = set(keys)
        if not q or not self.incidents:
            return []
        scored = []
        for inc in self.incidents:
            h = self._keys_set(inc)
            if not h:
                continue
            j = len(q & h) / len(q | h)
            bonus = 0.1 if (region and inc.get("region") == region) else 0.0
            s = j + bonus
            if s >= min_sim:
                scored.append(dict(incident=inc, similarity=round(s, 4)))
        scored.sort(key=lambda x: -x["similarity"])
        return scored[:top_k]

from __future__ import annotations

import json
from pathlib import Path


class RunbookRegistry:
    def __init__(self, entries: list[dict] | None = None):
        self.entries = {e["id"]: e for e in (entries or [])}

    @staticmethod
    def from_file(path: str | Path) -> "RunbookRegistry":
        p = Path(path)
        if not p.exists():
            return RunbookRegistry([])
        return RunbookRegistry(json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(list(self.entries.values()), ensure_ascii=False, indent=2),
            encoding="utf-8")

    def add(self, entry: dict) -> dict:
        if "id" not in entry:
            raise ValueError("runbook entry requires id")
        self.entries[entry["id"]] = entry
        return entry

    def match(self, system: str, alert_group: str = "") -> dict | None:
        for e in self.entries.values():
            if e.get("system") == system and e.get("alert_group") == alert_group:
                return e
        for e in self.entries.values():
            if e.get("system") == system and not e.get("alert_group"):
                return e
        return None

    def list(self) -> list[dict]:
        return list(self.entries.values())

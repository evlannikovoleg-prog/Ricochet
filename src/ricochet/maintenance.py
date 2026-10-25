from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class MaintenanceWindow:
    start: datetime
    end: datetime
    region: str = ""
    system: str = ""
    node: str = ""
    source: str = ""
    note: str = ""
    id: str = ""

    def __post_init__(self):
        if isinstance(self.start, str):
            self.start = datetime.fromisoformat(self.start)
        if isinstance(self.end, str):
            self.end = datetime.fromisoformat(self.end)
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if self.end < self.start:
            raise ValueError(f"end before start: {self.start} > {self.end}")

    def matches(self, ts: datetime, region: str = "", system: str = "", node: str = "") -> bool:
        if not (self.start <= ts < self.end):
            return False
        if self.region and self.region != region:
            return False
        if self.system and self.system != system:
            return False
        if self.node and self.node != node:
            return False
        return True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d

    @staticmethod
    def from_dict(d: dict) -> "MaintenanceWindow":
        return MaintenanceWindow(**d)


class MaintenanceStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.windows: list[MaintenanceWindow] = []
        self._load()

    def _load(self) -> None:
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.windows = [MaintenanceWindow.from_dict(d) for d in raw.get("windows", [])]

    def save(self) -> None:
        if not self.path:
            return
        self.path.write_text(
            json.dumps({"windows": [w.to_dict() for w in self.windows]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, w: MaintenanceWindow) -> MaintenanceWindow:
        self.windows.append(w)
        self.save()
        return w

    def remove(self, window_id: str) -> bool:
        before = len(self.windows)
        self.windows = [w for w in self.windows if w.id != window_id]
        removed = len(self.windows) < before
        if removed:
            self.save()
        return removed

    def list(self) -> list[dict]:
        return [w.to_dict() for w in sorted(self.windows, key=lambda w: w.start)]

    def hit(self, ts: datetime, region: str = "", system: str = "", node: str = "") -> list[MaintenanceWindow]:
        return [w for w in self.windows if w.matches(ts, region, system, node)]

    def filter_alarms(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty or not self.windows:
            return df, df.iloc[0:0]
        ts = df["ts"].dt.to_pydatetime() if hasattr(df["ts"], "dt") else df["ts"]
        keep = []
        hit_ids: list[str] = []
        for t, r, s, n in zip(df["ts"], df["region"], df["system"], df["node"]):
            hits = self.hit(t.to_pydatetime() if hasattr(t, "to_pydatetime") else t, r, s, n)
            if hits:
                hit_ids.append(hits[0].id)
                keep.append(False)
            else:
                hit_ids.append("")
                keep.append(True)
        out = df.copy()
        out["_maintenance_id"] = hit_ids
        return out[keep].drop(columns=["_maintenance_id"]), out[~pd.Series(keep, index=df.index)].drop(columns=["_maintenance_id"])

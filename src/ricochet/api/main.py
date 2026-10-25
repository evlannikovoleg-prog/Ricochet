from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import AppConfig, EngineConfig, ensure_dirs
from ..canonical import read_alarms, write_alarms, validate_alarms
from ..ingest.synthetic import generate_synthetic
from ..maintenance import MaintenanceStore
from ..precheck.similar import HistoricalIncidents
from ..precheck.runbooks import RunbookRegistry


class AppState:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        ensure_dirs(cfg)
        self.alarms = read_alarms(cfg.alarms_path)
        self.synthetic = False
        if self.alarms.empty and cfg.use_synthetic_when_empty:
            df, meta = generate_synthetic(cfg.synthetic_days, cfg.synthetic_seed)
            self.alarms = validate_alarms(df)
            write_alarms(self.alarms, cfg.alarms_path)
            (cfg.state_dir / "synthetic_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            (cfg.state_dir / "incidents.json").write_text(
                json.dumps(meta["incidents"], ensure_ascii=False, indent=2), encoding="utf-8")
            self.synthetic = True
        self.maintenance = MaintenanceStore(cfg.state_dir / "maintenance.json")
        if self.synthetic and not self.maintenance.windows:
            meta_path = cfg.state_dir / "synthetic_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                from ..maintenance import MaintenanceWindow
                w = meta["maintenance_window"]
                self.maintenance.add(MaintenanceWindow(
                    start=w["start"], end=w["end"], region=w.get("region", ""),
                    system=w.get("system", ""), node=w.get("node", ""),
                    source=w.get("source", ""), note=w.get("note", "")))
        incidents_path = cfg.state_dir / "incidents.json"
        if not incidents_path.exists():
            incidents_path = cfg.processed_dir / "incidents.json"
        self.incidents = HistoricalIncidents.from_file(incidents_path)
        self.runbooks = RunbookRegistry.from_file(cfg.state_dir / "runbooks.json")
        if self.synthetic and not self.runbooks.entries:
            self.runbooks.add(dict(
                id="rb-mpls-001", system="snmp", alert_group="mplsTunnelRerouted",
                url="http://runbooks.local/mpls-reroute",
                steps=["check LSP path", "verify TE tunnel", "inspect IS-IS adjacencies"],
                owner="ipcore-team"))
            self.runbooks.add(dict(
                id="rb-nokia-dry", system="nokia", alert_group="dryContactAlarm",
                url="http://runbooks.local/dry-contact",
                steps=["verify dry contact on site", "compare with ALU external alarm feed"],
                owner="transport-team"))
            self.runbooks.save(cfg.state_dir / "runbooks.json")
        self.butterfly_last: dict | None = self._load_butterfly()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _load_butterfly(self) -> dict | None:
        p = self.cfg.state_dir / "butterfly_last.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def save_butterfly(self, result: dict) -> None:
        self.butterfly_last = result
        p = self.cfg.state_dir / "butterfly_last.json"
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def submit_job(self, fn, label: str) -> str:
        import uuid
        job_id = uuid.uuid4().hex[:12]
        job = dict(id=job_id, label=label, status="running",
                   started_at=datetime.now().isoformat(), error=None)
        with self._lock:
            self.jobs[job_id] = job

        def _run():
            try:
                res = fn()
                job["status"] = "done"
                job["result"] = res
            except Exception as e:
                job["status"] = "error"
                job["error"] = f"{type(e).__name__}: {e}"

        self.executor.submit(_run)
        return job_id

    def filtered_alarms(self, ts_from=None, ts_to=None) -> pd.DataFrame:
        d = self.alarms
        if ts_from is not None:
            d = d[d["ts"] >= pd.Timestamp(ts_from)]
        if ts_to is not None:
            d = d[d["ts"] < pd.Timestamp(ts_to)]
        return d


def run_pipeline(state: AppState, params: dict) -> dict:
    from ..leadlag import run_butterfly
    from ..verdicts import finalize_pairs
    cfg = EngineConfig(**{k: v for k, v in params.items()
                          if k in EngineConfig.__dataclass_fields__ and v is not None})
    holdout_last = params.get("holdout_last", 1)
    d = state.filtered_alarms(params.get("ts_from"), params.get("ts_to"))
    kept, _ = state.maintenance.filter_alarms(d)
    if kept.empty:
        kept = d
    result = run_butterfly(kept, cfg, holdout_last=holdout_last)
    result["pairs"] = finalize_pairs(kept, result["pairs"], cfg)
    state.save_butterfly(result)
    return dict(info=result["info"], pairs=len(result["pairs"]))


def create_app(cfg: AppConfig | None = None) -> "object":
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from . import routes_butterfly, routes_precheck
    config = cfg or AppConfig()

    @asynccontextmanager
    async def lifespan(app):
        app.state.svc = AppState(config)
        yield

    app = FastAPI(title="Ricochet", version="0.1.0", lifespan=lifespan,
                  description="Butterfly hidden-dependency discovery + incident precheck")

    app.include_router(routes_precheck.router, prefix="/api/v1")
    app.include_router(routes_butterfly.router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        svc = app.state.svc
        months = sorted({f"{t.year:04d}-{t.month:02d}" for t in svc.alarms["ts"]}) if not svc.alarms.empty else []
        return dict(
            status="ok",
            alarms=len(svc.alarms),
            synthetic=svc.synthetic,
            months=months,
            pairs_last=None if svc.butterfly_last is None else len(svc.butterfly_last.get("pairs", [])),
            maintenance_windows=len(svc.maintenance.windows),
            historical_incidents=len(svc.incidents.incidents),
        )

    return app


app = create_app()

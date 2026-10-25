from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CN_DIR_MAP = {
    "业务指标": "business",
    "平台指标": "platform",
    "调用链指标": "trace",
    "AIOps挑战赛数据": "challenge_data",
}
CN_FILE_MAP = {
    "故障整理（预赛）.csv": "faults_preliminary.csv",
    "故障整理(预赛).csv": "faults_preliminary.csv",
}
CN_COL_MAP = {
    "时间戳": "ts",
    "时间": "ts",
    "指标名": "metric",
    "指标值": "value",
    "值": "value",
    "实例名": "instance",
    "实例": "instance",
    "主机名": "host",
    "主机": "host",
    "服务名": "service",
    "接口名": "endpoint",
    "故障类型": "fault_type",
    "故障定位": "fault_location",
    "具体定位": "fault_detail",
    "处理人ID": "handler_id",
    "pod名": "pod",
    "容器名": "container",
    "id": "span_id",
    "pid": "parent_span_id",
    "耗时": "duration_ms",
}
FAULTS_OBJECT_TO_SYSTEM = {
    "docker": "dcos_docker",
    "container": "dcos_container",
    "os": "os_linux",
    "db": "db_oracle_11g",
    "oracle": "db_oracle_11g",
    "redis": "mw_redis",
    "esb": "esb",
}
ALARM_COLUMNS = [
    "ts", "integration", "system", "alert_group", "node", "node_alias", "region", "severity"
]


def _fix_zip_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("gbk")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def _safe_ascii(s: str) -> str:
    return "".join(c if (c.isascii() and (c.isalnum() or c in "._-/ ()")) else "_" for c in s)


def _to_english_path(name: str) -> str:
    parts = _fix_zip_name(name).replace("\\", "/").split("/")
    out = []
    for p in parts:
        if p in CN_DIR_MAP:
            out.append(CN_DIR_MAP[p])
        elif p in CN_FILE_MAP:
            out.append(CN_FILE_MAP[p])
        else:
            out.append(_safe_ascii(p))
    return "/".join(out)


def _read_csv_bytes(data: bytes) -> pd.DataFrame | None:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return None


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: CN_COL_MAP.get(str(c).strip(), str(c).strip()) for c in df.columns})


def inspect_zip(zip_path: str | Path, max_files: int = 30) -> dict:
    zp = Path(zip_path)
    report = {"outer": [], "nested": [], "samples": []}
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            fixed = _fix_zip_name(info.filename)
            report["outer"].append(
                dict(raw=info.filename, fixed=fixed, english=_to_english_path(info.filename),
                     size=info.file_size))
            if fixed.lower().endswith(".zip") and len(report["nested"]) < max_files:
                with zf.open(info) as fh:
                    nested_data = fh.read()
                try:
                    with zipfile.ZipFile(io.BytesIO(nested_data)) as nz:
                        entries = []
                        for ninfo in nz.infolist()[:max_files]:
                            entries.append(dict(
                                fixed=_fix_zip_name(ninfo.filename),
                                english=_to_english_path(ninfo.filename),
                                size=ninfo.file_size,
                            ))
                            if len(report["samples"]) < max_files and ninfo.filename.lower().endswith(".csv"):
                                with nz.open(ninfo) as cf:
                                    df = _read_csv_bytes(cf.read(3_000_000))
                                if df is not None:
                                    report["samples"].append(dict(
                                        file=_to_english_path(ninfo.filename),
                                        columns=list(df.columns),
                                        rows_head=df.head(2).to_dict("records"),
                                    ))
                        report["nested"].append(dict(archive=info.filename, fixed=fixed, entries=entries))
                except zipfile.BadZipFile:
                    report["nested"].append(dict(archive=info.filename, fixed=fixed, error="not a zip"))
    return report


def extract_all(zip_path: str | Path, out_dir: str | Path) -> list[Path]:
    zp = Path(zip_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            data = zf.read(info)
            fixed = _fix_zip_name(info.filename)
            if fixed.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as nz:
                        for ninfo in nz.infolist():
                            if ninfo.is_dir():
                                continue
                            ndata = nz.read(ninfo)
                            dest = out / _to_english_path(fixed).replace(".zip", "") / \
                                _to_english_path(ninfo.filename)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(ndata)
                            written.append(dest)
                except zipfile.BadZipFile:
                    dest = out / _to_english_path(info.filename)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    written.append(dest)
            else:
                dest = out / _to_english_path(info.filename)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                written.append(dest)
    return written


def load_metric_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = _read_csv_bytes(p.read_bytes())
    if df is None:
        df = pd.read_csv(p, encoding="gb18030")
    return _rename_columns(df)


def load_faults(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = None
    for enc in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            df = pd.read_csv(p, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"cannot decode {p}")
    return _rename_columns(df)


def _mad_flags(values: pd.Series, window: int, k: float, min_periods: int):
    med = values.rolling(window, center=True, min_periods=min_periods).median()
    mad = (values - med).abs().rolling(window, center=True, min_periods=min_periods).median()
    spread = k * 1.4826 * mad
    high = values > med + spread
    low = values < med - spread
    return high, low


def _first_runs(mask: pd.Series) -> pd.Series:
    return mask & ~mask.shift(1, fill_value=False)


def platform_to_alarms(df: pd.DataFrame, source: str, k: float = 4.0,
                       min_points: int = 200) -> pd.DataFrame:
    need = {"name", "timestamp", "value", "cmdb_id"}
    if not need <= set(df.columns):
        return pd.DataFrame(columns=ALARM_COLUMNS)
    d = df.dropna(subset=["timestamp", "value"]).copy()
    d["ts"] = pd.to_datetime(d["timestamp"], unit="ms")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value"])
    rows = []
    for (metric, node), g in d.groupby(["name", "cmdb_id"], sort=False):
        if len(g) < min_points:
            continue
        g = g.sort_values("ts")
        high, low = _mad_flags(g["value"], window=120, k=k, min_periods=30)
        for mask, suffix in ((high, "high"), (low, "low")):
            for t in g.loc[_first_runs(mask), "ts"]:
                rows.append((t, "platform", source, f"{metric}_{suffix}", node, "", "", 4))
    return pd.DataFrame(rows, columns=ALARM_COLUMNS)


def business_to_alarms(df: pd.DataFrame, k: float = 4.0, min_points: int = 200,
                       low_success: float = 0.98) -> pd.DataFrame:
    need = {"serviceName", "startTime", "avg_time", "num", "succee_rate"}
    if not need <= set(df.columns):
        return pd.DataFrame(columns=ALARM_COLUMNS)
    d = df.dropna(subset=["startTime"]).copy()
    d["ts"] = pd.to_datetime(d["startTime"], unit="ms")
    rows = []
    for service, g in d.groupby("serviceName", sort=False):
        if len(g) < min_points:
            continue
        g = g.sort_values("ts")
        slow, _ = _mad_flags(g["avg_time"], window=120, k=k, min_periods=30)
        bad = g["succee_rate"] < low_success
        for t in g.loc[bad, "ts"]:
            rows.append((t, "business", "esb", "esbLowSuccess", service, "", "", 4))
        for t in g.loc[_first_runs(slow), "ts"]:
            rows.append((t, "business", "esb", "esbSlowResponse", service, "", "", 3))
    return pd.DataFrame(rows, columns=ALARM_COLUMNS)


def trace_to_alarms(df: pd.DataFrame, k: float = 6.0, min_points: int = 5000,
                    max_alarms_per_group: int = 20000) -> pd.DataFrame:
    need = {"callType", "startTime", "elapsedTime", "success", "cmdb_id", "serviceName"}
    if not need <= set(df.columns):
        return pd.DataFrame(columns=ALARM_COLUMNS)
    d = df.dropna(subset=["startTime"]).copy()
    d["ts"] = pd.to_datetime(d["startTime"], unit="ms")
    d["elapsedTime"] = pd.to_numeric(d["elapsedTime"], errors="coerce")
    if d["success"].dtype == object:
        d["success"] = d["success"].astype(str).str.lower().isin(("true", "1", "1.0"))
    rows = []
    for (service, ctype), g in d.groupby(["serviceName", "callType"], sort=False):
        fails = g[~g["success"]]
        if not fails.empty:
            if len(fails) > max_alarms_per_group:
                fails = fails.sample(max_alarms_per_group, random_state=7)
            for t, node in zip(fails["ts"], fails["cmdb_id"]):
                rows.append((t, "trace", service, f"{ctype}_fail", node, "", "", 4))
        if len(g) >= min_points:
            g = g.sort_values("ts")
            slow, _ = _mad_flags(g["elapsedTime"], window=2000, k=k, min_periods=500)
            slow_first = g.loc[_first_runs(slow)]
            if len(slow_first) > max_alarms_per_group:
                slow_first = slow_first.sample(max_alarms_per_group, random_state=7)
            for t, node in zip(slow_first["ts"], slow_first["cmdb_id"]):
                rows.append((t, "trace", service, f"{ctype}_slow", node, "", "", 3))
    return pd.DataFrame(rows, columns=ALARM_COLUMNS)


def faults_to_incidents(faults: pd.DataFrame, alarms: pd.DataFrame,
                         window_min: int = 60) -> list[dict]:
    if faults.empty or "log_time" not in faults.columns:
        return []
    f = faults.copy()
    f["log_time"] = pd.to_datetime(f["log_time"], format="mixed", dayfirst=False)
    a = alarms.copy()
    out = []
    for i, r in f.iterrows():
        t = r["log_time"]
        system = FAULTS_OBJECT_TO_SYSTEM.get(str(r.get("object", "")).lower(),
                                              str(r.get("object", "")))
        kpi = str(r.get("kpi", "")) or ""
        ag = f"{kpi}_high" if kpi and kpi != "nan" else str(r.get("fault_desrcibtion", ""))
        keys = [{"system": system, "alert_group": ag}]
        near = a[(a["ts"] >= t) & (a["ts"] <= t + pd.Timedelta(minutes=window_min))]
        for s, g2 in near.groupby(["system", "alert_group"]).size().sort_values(
                ascending=False).head(8).reset_index().values:
            keys.append({"system": s, "alert_group": g2})
        out.append(dict(
            id=f"FAULT-{i:04d}",
            started_at=t.isoformat(),
            keys=keys,
            region="",
            fault_type=str(r.get("fault_desrcibtion", "")),
            fault_location=str(r.get("name", "")),
            fault_detail=str(r.get("container", "")),
            duration=str(r.get("duration", "")),
        ))
    return out


def _layer_files(root: Path, layer: str) -> list[Path]:
    return [p for p in sorted(root.rglob("*.csv")) if p.parent.name == layer]


def ingest_metrics_tree(root: str | Path, processed_dir: str | Path, k: float = 4.0) -> dict:
    from ..canonical import write_alarms, validate_alarms
    root = Path(root)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    all_alarms = []
    stats = {}

    for layer in ("platform", "business", "trace"):
        frames = []
        for p in _layer_files(root, layer):
            try:
                df = load_metric_csv(p)
            except Exception:
                continue
            if layer == "platform":
                al = platform_to_alarms(df, source=p.stem, k=k)
            elif layer == "business":
                al = business_to_alarms(df, k=k)
            else:
                al = trace_to_alarms(df)
            if not al.empty:
                frames.append(al)
        if not frames:
            stats[layer] = dict(files=0, alarms=0)
            continue
        al = pd.concat(frames, ignore_index=True)
        stats[layer] = dict(files=len(frames), alarms=len(al))
        all_alarms.append(al)

    faults_path = None
    for cand in root.rglob("faults_preliminary.csv"):
        faults_path = cand
        break
    incidents = []
    faults_df = pd.DataFrame()
    if faults_path is not None:
        faults_df = load_faults(faults_path)
        faults_df.to_csv(processed / "faults.csv", index=False)
    if all_alarms:
        out = validate_alarms(pd.concat(all_alarms, ignore_index=True))
        write_alarms(out, processed / "alarms.parquet")
        write_alarms(out, processed / "alarms.csv")
        if not faults_df.empty:
            incidents = faults_to_incidents(faults_df, out, window_min=60)
    (processed / "incidents.json").write_text(
        json.dumps(incidents, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(stats=stats, incidents=len(incidents), alarms_total=len(out) if all_alarms else 0)

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "ts",
    "integration",
    "system",
    "alert_group",
    "node",
    "node_alias",
    "region",
    "severity",
]
GROUP_KEY = ["integration", "system", "alert_group"]
GROUP_KEY_REGION = GROUP_KEY + ["region"]


def empty_alarms() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in COLUMNS})


def validate_alarms(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    for c in ["integration", "system", "alert_group", "node", "node_alias", "region"]:
        out[c] = out[c].fillna("").astype(str)
    out["severity"] = pd.to_numeric(out["severity"], errors="coerce").fillna(0).astype(int)
    return out[COLUMNS].sort_values("ts", kind="stable").reset_index(drop=True)


def alarms_from_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return empty_alarms()
    return validate_alarms(pd.DataFrame.from_records(records))


def read_alarms(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return empty_alarms()
    if p.suffix == ".parquet":
        return validate_alarms(pd.read_parquet(p))
    return validate_alarms(pd.read_csv(p))


def write_alarms(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def group_device_sets(df: pd.DataFrame) -> dict[tuple, set[tuple[str, str]]]:
    sub = df[GROUP_KEY + ["node", "node_alias"]].copy()
    sub["node"] = sub["node"].fillna("")
    sub["node_alias"] = sub["node_alias"].fillna("")
    return {
        k: set(zip(g["node"], g["node_alias"]))
        for k, g in sub.groupby(GROUP_KEY, sort=False, dropna=False)
    }


def profile_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["system"], df["alert_group"]))

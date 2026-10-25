from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ricochet.canonical import validate_alarms, write_alarms
from ricochet.ingest.synthetic import generate_synthetic
from ricochet.maintenance import MaintenanceStore, MaintenanceWindow


def main() -> None:
    ap = argparse.ArgumentParser(description="generate synthetic alarm stream with planted patterns")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "processed"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = out.parent.parent / "state"
    state.mkdir(parents=True, exist_ok=True)

    df, meta = generate_synthetic(args.days, args.seed)
    df = validate_alarms(df)
    write_alarms(df, out / "alarms.parquet")
    write_alarms(df, out / "alarms.csv")
    (state / "synthetic_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (state / "incidents.json").write_text(
        json.dumps(meta["incidents"], ensure_ascii=False, indent=2), encoding="utf-8")

    store = MaintenanceStore(state / "maintenance.json")
    w = meta["maintenance_window"]
    store.add(MaintenanceWindow(start=w["start"], end=w["end"], region=w.get("region", ""),
                                system=w.get("system", ""), node=w.get("node", ""),
                                source=w.get("source", ""), note=w.get("note", "")))

    print(json.dumps(dict(alarms=len(df), days=args.days, out=str(out),
                          butterfly=meta["butterfly"]["key_a"], lag=meta["butterfly"]["lag_bins"],
                          duplicate=meta["duplicate"], incidents=len(meta["incidents"])),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

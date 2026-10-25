from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np

from ricochet.config import PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="validate butterfly candidates against labeled faults")
    ap.add_argument("--window-min", type=int, default=30)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--tz-shift-hours", type=float, default=8.0,
                    help="fault log_time is local UTC+; alarms are UTC. Default +8 (Beijing)")
    args = ap.parse_args()

    faults = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "faults.csv")
    faults["log_time"] = pd.to_datetime(faults["log_time"], format="mixed")
    faults["log_time"] = faults["log_time"] - pd.Timedelta(hours=args.tz_shift_hours)
    ftimes = faults["log_time"].dropna().sort_values().to_numpy()

    alarms = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "alarms.parquet")
    pairs = json.loads((PROJECT_ROOT / "state" / "butterfly_last.json").read_text(encoding="utf-8"))["pairs"]

    w = np.timedelta64(args.window_min, "m")
    ts = alarms["ts"]

    def near_fault(series: pd.Series) -> float:
        if len(ftimes) == 0 or series.empty:
            return 0.0
        s = series.to_numpy()
        pos = np.searchsorted(ftimes, s, side="left")
        near = ftimes[pos.clip(0, len(ftimes) - 1)]
        prev = ftimes[(pos - 1).clip(0, len(ftimes) - 1)]
        return float(((np.abs(s - near) <= w) | (np.abs(s - prev) <= w)).mean())

    base = near_fault(ts)
    print(f"faults: {len(ftimes)}  window: ±{args.window_min} min")
    print(f"baseline (all alarms near fault): {base:.3f}")
    print(f"pairs: {len(pairs)}")

    rows = []
    for p in pairs:
        ka, kb = p["key_a"][:3], p["key_b"][:3]
        lag = p.get("lag_bins", 0)
        leader = ka if lag >= 0 else kb
        sel = alarms[(alarms["integration"] == leader[0]) &
                      (alarms["system"] == leader[1]) &
                      (alarms["alert_group"] == leader[2])]
        if sel.empty:
            continue
        hit = near_fault(sel["ts"])
        rows.append(dict(score=p.get("score", 0), corr=p["corr"], lag=p.get("lag_min"),
                         hit=hit, lift=hit / base if base else 0.0,
                         leader=f"{leader[1]}|{leader[2]}",
                         follower=f"{(kb if lag >= 0 else ka)[1]}|{(kb if lag >= 0 else ka)[2]}"))
    rows.sort(key=lambda r: -r["lift"])
    print(f"{'lift':>6} {'hit':>6} {'score':>6} {'corr':>7} {'lag':>5}  leader -> follower")
    for r in rows[:args.top]:
        print(f"{r['lift']:6.2f} {r['hit']:6.3f} {r['score']:6.3f} {r['corr']:+7.3f} "
              f"{r['lag']:>4}m  {r['leader']} -> {r['follower']}")
    if rows:
        import statistics
        lifts = [r["lift"] for r in rows]
        print(f"\nlift: median={statistics.median(lifts):.2f} "
              f"share>1.5: {sum(1 for l in lifts if l > 1.5) / len(lifts):.2f}")


if __name__ == "__main__":
    main()

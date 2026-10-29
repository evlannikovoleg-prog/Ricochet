from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ricochet.canonical import read_alarms
from ricochet.config import EngineConfig, PROJECT_ROOT
from ricochet.leadlag import run_butterfly
from ricochet.ranking import PairRanker, RankerConfig
from ricochet.verdicts import finalize_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="run butterfly pipeline on alarms")
    ap.add_argument("--alarms", default=str(PROJECT_ROOT / "data" / "processed" / "alarms.parquet"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "state" / "butterfly_last.json"))
    ap.add_argument("--holdout-last", type=int, default=1)
    ap.add_argument("--top", type=int, default=20)
    for f, t in [("min_events", int), ("min_bins", int), ("min_abs_corr", float),
                 ("min_support", int), ("min_devices", int), ("lags", int),
                 ("max_duty", float), ("hub_degree", int), ("transform", str)]:
        ap.add_argument(f"--{f.replace('_', '-')}", type=t, default=None)
    args = ap.parse_args()

    over = {k.replace("-", "_"): getattr(args, k.replace("-", "_"))
            for k, _ in [("min_events", 0), ("min_bins", 0), ("min_abs_corr", 0),
                         ("min_support", 0), ("min_devices", 0), ("lags", 0),
                         ("max_duty", 0), ("hub_degree", 0), ("transform", 0)]}
    over = {k: v for k, v in over.items() if v is not None}
    cfg = EngineConfig(**over)

    t0 = time.time()
    alarms = read_alarms(args.alarms)
    print(f"alarms: {len(alarms)}  {alarms['ts'].min()} -> {alarms['ts'].max()}")
    print(f"series: {alarms.groupby(['integration', 'system', 'alert_group']).ngroups}")

    res = run_butterfly(alarms, cfg, holdout_last=args.holdout_last)
    print(json.dumps(res["info"], ensure_ascii=False))
    res["pairs"] = finalize_pairs(alarms, res["pairs"], cfg)
    ranker = PairRanker(RankerConfig(state_dir=Path(args.out).parent))
    res["pairs"], ranker_summary = ranker.score_pairs(res["pairs"], alarms=alarms, cfg=cfg)
    res["ranker"] = ranker_summary
    res["info"]["ranker"] = ranker_summary
    print("ranker:", json.dumps({k: v for k, v in ranker_summary.items() if k != "features"},
                                ensure_ascii=False))

    for p in res["pairs"][:args.top]:
        x = p["xgboost"]
        print(f"{p['score']:.3f} rank={x['rank']:>3} p={x['probability']:.3f} "
              f"{p['class']:<20} corr={p['corr']:+.3f} lead={p['lead']:+.3f} "
              f"lag={p['lag_min']:+3d}m days={p['days_ok']}/{p['days_total']} "
              f"regions={p['regions_ok']}/{p['regions_total']} dev={p['devices_a']}/{p['devices_b']} "
              f"[{p['key_a'][0]}|{p['key_a'][1]}|{p['key_a'][2][:36]}] "
              "<-> "
              f" [{p['key_b'][0]}|{p['key_b'][1]}|{p['key_b'][2][:36]}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()

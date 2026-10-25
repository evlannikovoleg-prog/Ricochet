from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ricochet.ingest import aiops2020


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="AIOps Challenge 2020: inspect / extract / ingest")
    ap.add_argument("--zip", default=str(root / "data" / "raw" / "aiops2020.zip"))
    ap.add_argument("--mode", choices=["inspect", "extract", "ingest"], default="inspect")
    ap.add_argument("--extract-dir", default=str(root / "data" / "raw" / "extracted"))
    ap.add_argument("--processed-dir", default=str(root / "data" / "processed"))
    ap.add_argument("--max-files", type=int, default=30)
    ap.add_argument("--k", type=float, default=4.0, help="MAD multiplier for metric->alarm threshold")
    args = ap.parse_args()

    if args.mode == "inspect":
        report = aiops2020.inspect_zip(args.zip, max_files=args.max_files)
        out = root / "data" / "raw" / "inspect_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"outer entries: {len(report['outer'])}")
        for e in report["outer"][:10]:
            print(f"  {e['fixed']}  ->  {e['english']}  ({e['size']} bytes)")
        print(f"nested archives: {len(report['nested'])}")
        for n in report["nested"][:5]:
            print(f"  {n['fixed']}: {len(n.get('entries', []))} entries")
        print(f"samples: {len(report['samples'])}")
        for s in report["samples"][:8]:
            print(f"  {s['file']}: {s['rows']} rows, columns={s['columns']} -> {s['mapped']}")
        print(f"full report: {out}")

    elif args.mode == "extract":
        written = aiops2020.extract_all(args.zip, args.extract_dir)
        print(f"files written: {len(written)}")
        for p in written[:20]:
            print(f"  {p}")

    elif args.mode == "ingest":
        root_dir = Path(args.extract_dir)
        if not root_dir.exists():
            print("nothing extracted yet, run --mode extract first")
            sys.exit(1)
        result = aiops2020.ingest_metrics_tree(root_dir, args.processed_dir, k=args.k)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

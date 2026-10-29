from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ricochet.config import PROJECT_ROOT
from ricochet.ranking import PairRanker, RankerConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="train the XGBoost pair ranker from collected feedback")
    ap.add_argument("--state-dir", default=str(PROJECT_ROOT / "state"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-confidence", type=float, default=0.9)
    args = ap.parse_args()

    ranker = PairRanker(RankerConfig(state_dir=Path(args.state_dir),
                                     min_confidence=args.min_confidence))
    print("feedback:", json.dumps(ranker.feedback.summary(), ensure_ascii=False))
    result = ranker.train(force=args.force)
    print("training:", json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print("status:", json.dumps(ranker.status(include_features=False),
                                ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

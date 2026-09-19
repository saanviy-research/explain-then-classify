"""Build the stratified representative dev set and save it to data/dev_representative.csv.

Deterministic (seeded): re-running always regenerates the same set, so the CSV
does not need to be committed. Strata: label x loneliness-cue presence x
length tercile, allocated proportionally to the full test set.

Usage: python scripts/make_representative_dev.py [target_size]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data import load_dataset
from src.splits import make_representative_dev_set, make_splits


def main():
    target_size = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    df = load_dataset()
    _, _, test_df = make_splits(df)
    rep_df = make_representative_dev_set(test_df, target_size=target_size)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(config.REP_DEV_PATH, index=False)
    print(f"Saved -> {config.REP_DEV_PATH}")


if __name__ == "__main__":
    main()

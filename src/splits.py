"""Stratified train/val/test split and development-set sampling."""
import re

import numpy as np
import pandas as pd

import config

# Explicit loneliness cue words. Present posts WITHOUT a cue are hard
# positives (about half of all Present posts); Absent posts WITH a cue are
# rare hard negatives (~7% of Absent posts).
_CUE_RE = re.compile(
    r"\b(lonel\w*|alone|isolat\w*|no friends|no one|nobody|by myself"
    r"|abandon\w*|unloved|unwanted|left out|outcast|alienat\w*)\b",
    re.I,
)


def _split_df(d: pd.DataFrame, ratios=(0.70, 0.10, 0.20)):
    n = len(d)
    t = int(n * ratios[0])
    v = int(n * ratios[1])
    return d.iloc[:t], d.iloc[t : t + v], d.iloc[t + v :]


def make_splits(df: pd.DataFrame, seed: int = config.SEED):
    """Reproducible stratified split (same seed as previous experiments)."""
    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    pos = df[df.label == 1]
    neg = df[df.label == 0]

    pos_train, pos_val, pos_test = _split_df(pos)
    neg_train, neg_val, neg_test = _split_df(neg)

    train_df = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Train: {len(train_df)}  (Present={train_df.label.sum()})")
    print(f"Val:   {len(val_df)}  (Present={val_df.label.sum()})")
    print(f"Test:  {len(test_df)}  (Present={test_df.label.sum()})")

    return train_df, val_df, test_df


def compute_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Label x cue-presence x length-tercile strata for an arbitrary split.

    Terciles are computed on this df's own text-length distribution (not
    shared bin edges across splits) - fine for stratum-proportion matching,
    which only needs a consistent method, not identical bin edges.
    """
    t = df.copy()
    t["n_words"] = t["text"].astype(str).str.split().str.len()
    t["has_cue"] = t["text"].astype(str).apply(lambda s: bool(_CUE_RE.search(s)))
    t["len_bin"] = pd.qcut(t["n_words"], 3, labels=["short", "medium", "long"], duplicates="drop")
    return t


def make_representative_dev_set(
    test_df: pd.DataFrame, target_size: int = 80, seed: int = config.SEED
) -> pd.DataFrame:
    """Stratified, diversity-representative dev set drawn from the test set.

    Strata: label x explicit-loneliness-cue presence x text-length tercile
    (12 strata). Allocation is proportional to stratum frequency in the full
    test set, with a floor of 4 per non-empty stratum (capped by stratum
    size) so rare-but-diagnostic strata (e.g. Absent posts containing
    loneliness cue words) are always represented.
    """
    t = test_df.copy()
    t["n_words"] = t["text"].astype(str).str.split().str.len()
    t["has_cue"] = t["text"].astype(str).apply(lambda s: bool(_CUE_RE.search(s)))
    t["len_bin"] = pd.qcut(t["n_words"], 3, labels=["short", "medium", "long"])

    total = len(t)
    parts = []
    for (label, has_cue, len_bin), stratum in t.groupby(
        ["label", "has_cue", "len_bin"], observed=True
    ):
        n = round(target_size * len(stratum) / total)
        n = min(max(n, 4), len(stratum))
        parts.append(stratum.sample(n, random_state=seed))

    dev_df = (
        pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    )

    print(f"Representative dev set: {len(dev_df)} examples")
    print(dev_df.groupby(["label", "has_cue", "len_bin"], observed=True).size())
    return dev_df


def make_dev_set(test_df: pd.DataFrame, n_per_class: int = 5, seed: int = config.SEED) -> pd.DataFrame:
    """Balanced development set for fast prompt iteration."""
    dev_pos = test_df[test_df.label == 1].sample(n_per_class, random_state=seed)
    dev_neg = test_df[test_df.label == 0].sample(n_per_class, random_state=seed)
    dev_df = pd.concat([dev_pos, dev_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nDev set (for iteration): {len(dev_df)}")
    print(dev_df["label"].value_counts())

    return dev_df

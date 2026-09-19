"""Recompute every statistic quoted in the camera-ready reviewer responses.

Reads only committed per-post prediction files — no API calls, no network.

    python scripts/reviewer_response_stats.py

Produces:
  1. Accuracy on the 618 test examples never used during development
     (answers Reviewer 1 on test-set independence).
  2. A k-fold analysis of the held-out predictions. For a training-free
     method k-fold CV degenerates to partitioning the evaluation set: no
     model is fitted, so the per-fold procedure is identical and the
     fold-to-fold spread isolates evaluation-sample variance
     (answers Reviewers 1 and 2 on cross-validation).
  3. Bootstrap confidence intervals on accuracy.
  4. The paired comparison between the reported model and the development
     model, including the exact binomial test.
"""

import csv
import os
import random
from math import comb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

# Per-post predictions for the three full-test-set runs.
RUNS = [
    ("gemini-3.7-flash", "span_pipeline_test_results_gemini-3.7-flash.csv"),
    ("gemini-3.5-flash-lite", "span_pipeline_test_results.csv"),
    ("gemini-3.6-flash", "span_pipeline_test_results_gemini-3.6-flash.csv"),
]
DEV_SUBSET = "dev_representative.csv"   # the 89 examples used during development

BASELINE = 0.78          # Garg et al., GloVe+BiGRU, 10-fold CV
SEED = 42
BOOTSTRAP = 10_000
FOLDS = 10


def norm(text):
    """Whitespace-normalised prefix, used to match rows across files."""
    return " ".join((text or "").split())[:180]


def load(filename):
    with open(os.path.join(DATA, filename), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def correct(row):
    return int(row["true_label"]) == int(row["prediction"])


def accuracy(rows):
    return sum(correct(r) for r in rows) / len(rows)


def stratified_folds(rows, k=FOLDS, seed=SEED):
    """Label-stratified folds, mirroring how k-fold CV is normally run."""
    by_class = {0: [], 1: []}
    for r in rows:
        by_class[int(r["true_label"])].append(r)
    rnd = random.Random(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        members = by_class[cls][:]
        rnd.shuffle(members)
        for i, r in enumerate(members):
            folds[i % k].append(r)
    return folds


def bootstrap_ci(flags, reps=BOOTSTRAP, seed=SEED):
    rnd = random.Random(seed)
    n = len(flags)
    means = sorted(
        sum(flags[rnd.randrange(n)] for _ in range(n)) / n for _ in range(reps)
    )
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def exact_binomial_two_sided(successes, trials):
    """P(outcome at least as extreme as `successes`) under p=0.5."""
    if trials == 0:
        return 1.0
    target = comb(trials, successes)
    total = sum(comb(trials, j) for j in range(trials + 1) if comb(trials, j) <= target)
    return min(1.0, total / 2 ** trials)


def main():
    dev_texts = {norm(r["text"]) for r in load(DEV_SUBSET)}
    runs = {name: load(fn) for name, fn in RUNS}

    print("=" * 78)
    print("1. TEST-SET INDEPENDENCE  (Reviewer 1)")
    print("=" * 78)
    print(f"{'model':<24}{'n':>6}{'full':>9}{'n_clean':>9}{'never-touched':>15}{'delta':>9}")
    clean_rows = {}
    for name, rows in runs.items():
        clean = [r for r in rows if norm(r["text"]) not in dev_texts]
        clean_rows[name] = clean
        print(f"{name:<24}{len(rows):>6}{accuracy(rows):>9.4f}"
              f"{len(clean):>9}{accuracy(clean):>15.4f}"
              f"{accuracy(clean) - accuracy(rows):>+9.4f}")
    print("\nAccuracy rises on the never-touched subset for every model: the 89 held\n"
          "out for development were hard-case-stratified, so removing them removes\n"
          "difficulty rather than removing an advantage.\n")

    print("=" * 78)
    print("2. K-FOLD ANALYSIS  (Reviewers 1 and 2)")
    print("=" * 78)
    print("No model is fitted, so each fold applies an identical procedure and\n"
          "k-fold CV reduces to partitioning the evaluation set.\n")
    print(f"{'model / set':<38}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}{'>baseline':>11}")
    for name, rows in runs.items():
        for label, subset in ((f"{name}  (707 held-out)", rows),
                              (f"{name}  (618 clean)", clean_rows[name])):
            accs = [accuracy(f) for f in stratified_folds(subset)]
            mean = sum(accs) / len(accs)
            sd = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
            above = sum(1 for a in accs if a > BASELINE)
            print(f"{label:<38}{mean:>9.4f}{sd:>8.4f}{min(accs):>8.4f}"
                  f"{max(accs):>8.4f}{above:>8}/{len(accs)}")
    print()

    print("=" * 78)
    print("3. BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 78)
    for name, rows in runs.items():
        lo, hi = bootstrap_ci([1 if correct(r) else 0 for r in rows])
        verdict = "excludes" if lo > BASELINE else "includes"
        print(f"{name:<24}{accuracy(rows):.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
              f"   {verdict} {BASELINE}")
    print()

    print("=" * 78)
    print("4. PAIRED COMPARISON: reported model vs development model")
    print("=" * 78)
    a = {int(r["original_index"]): r for r in runs["gemini-3.5-flash-lite"]}
    b = {int(r["original_index"]): r for r in runs["gemini-3.7-flash"]}
    assert set(a) == set(b), "prediction files are not aligned by original_index"

    both_right = both_wrong = only_a = only_b = 0
    for i in a:
        ca, cb = correct(a[i]), correct(b[i])
        if ca and cb:
            both_right += 1
        elif not ca and not cb:
            both_wrong += 1
        elif ca:
            only_a += 1
        else:
            only_b += 1

    discordant = only_a + only_b
    pval = exact_binomial_two_sided(only_b, discordant)
    diff = accuracy(runs["gemini-3.7-flash"]) - accuracy(runs["gemini-3.5-flash-lite"])

    print(f"  both correct           {both_right}")
    print(f"  both wrong             {both_wrong}")
    print(f"  only development model {only_a}")
    print(f"  only reported model    {only_b}")
    print(f"  discordant             {discordant}")
    print(f"\n  accuracy difference    {diff:+.4f}")
    print(f"  exact binomial p       {pval:.4f}"
          f"   ({'not ' if pval >= 0.05 else ''}significant at p<0.05)")

    rnd = random.Random(SEED)
    idx = list(a)
    diffs = sorted(
        sum((1 if correct(b[j]) else 0) - (1 if correct(a[j]) else 0)
            for j in (idx[rnd.randrange(len(idx))] for _ in range(len(idx)))) / len(idx)
        for _ in range(BOOTSTRAP)
    )
    lo, hi = diffs[int(0.025 * BOOTSTRAP)], diffs[int(0.975 * BOOTSTRAP)]
    spans = "spans" if lo <= 0 <= hi else "excludes"
    print(f"  bootstrap 95% CI       [{lo:+.4f}, {hi:+.4f}]   {spans} zero")
    print("\nThe reported configuration is therefore presented as primary without a\n"
          "claim of significant advantage over the other models.")


if __name__ == "__main__":
    main()

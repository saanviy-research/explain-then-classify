"""Idea 5: explanation-first pipeline (extract the span, then judge the span).

Step 1 (kill criterion): run the extractor on 50 train Present posts, compute
token-level recall of the extracted span vs the gold span. If mean recall
< 0.5, the extractor is finding the wrong evidence - stop before stage 2.

Step 2 (only if step 1 clears): run both stages on the rep set, report
accuracy/Macro-F1 (stage 2 verdict) and ROUGE-1-style precision/recall of
stage 1 vs gold spans on rep Present posts.

Step 3 (full 707-example test-set run, gated on explicit user permission):
same pipeline, full held-out test split, checkpointed every 25 posts so an
interruption doesn't lose progress.

Usage: python scripts/run_span_pipeline.py kill
       python scripts/run_span_pipeline.py full
       python scripts/run_span_pipeline.py test [--workers N]

--workers N (test/full only, default 1) runs N posts concurrently via a
thread pool - these are independent, I/O-bound network calls (post A's
extract+judge doesn't depend on post B's), so threads (not processes) are
enough to overlap the network wait. Start modest (3-5): going too wide risks
tripping per-minute rate limits, which would surface as the same persistent-
failure abort this script already guards against.

Set GEMINI_MODEL to run the same pipeline against a different model (see
README.md's model-comparison section) - checkpoint/output filenames are
model-aware, so different models' runs never collide.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import config
from src import llm_client
from src.data import load_dataset
from src.llm_client import LLMPersistentFailure, call_llm
from src.parsing import extract_json
from src.progress import ProgressReporter
from src.span_pipeline import (
    build_extract_examples_block,
    build_extract_prompt,
    build_judge_examples_block,
    build_judge_prompt,
    extract_context_sentence,
    get_calibration_examples,
    token_recall,
)
from src.splits import make_splits

KILL_N = 50
KILL_THRESHOLD = 0.5

_DEFAULT_MODEL = "gemini-3.5-flash-lite"


def _model_slug() -> str:
    return config.MODEL_NAME.replace("/", "_").replace(":", "_")


def _rep_out_path():
    """Model-aware output path so a model-comparison run never overwrites another
    model's rep-set results. The default model keeps its original filename for
    backward compatibility with existing Idea 5 results/docs."""
    if config.MODEL_NAME == _DEFAULT_MODEL:
        return config.DATA_DIR / "span_pipeline_rep_results.csv"
    return config.DATA_DIR / f"span_pipeline_rep_results_{_model_slug()}.csv"


def _test_checkpoint_path():
    """Model-aware checkpoint path - same reasoning as _rep_out_path(). Critical for
    full-707 runs: without this, a second model's run would see the default model's
    already-complete checkpoint and skip every row instead of actually running."""
    if config.MODEL_NAME == _DEFAULT_MODEL:
        return config.DATA_DIR / "span_pipeline_test_results.csv"
    return config.DATA_DIR / f"span_pipeline_test_results_{_model_slug()}.csv"


def run_extractor(text: str, examples_block: str) -> str:
    prompt = build_extract_prompt(text, examples_block)
    raw = call_llm(prompt, temperature=0.0)
    parsed = extract_json(raw)
    span = str(parsed.get("span", "NONE")).strip()
    return span


def _process_post(idx, row, extract_examples: str, judge_examples: str) -> dict:
    """One post's full two-stage pipeline (extract, then judge). The two calls
    are inherently sequential (the judge needs the extractor's span) - but
    different posts are fully independent, which is what makes them safe to
    run concurrently across threads (see --workers)."""
    text = str(row["text"])
    true_label = int(row["label"])

    span = run_extractor(text, extract_examples)
    context = extract_context_sentence(text, span)

    judge_prompt = build_judge_prompt(span, context, judge_examples)
    raw = call_llm(judge_prompt, temperature=0.0)
    parsed = extract_json(raw)
    decision = str(parsed.get("decision", "Absent")).strip().lower()
    pred = 1 if "present" in decision else 0

    return {
        "original_index": idx,
        "true_label": true_label,
        "prediction": pred,
        "correct": int(true_label == pred),
        "span": span,
        "text": text[:300],
    }


def run_kill_check():
    llm_client.reset_counters()
    df = load_dataset()
    train_df, _, _ = make_splits(df)
    examples_block = build_extract_examples_block(train_df)

    calib_ids = set(get_calibration_examples(train_df).index)
    present_train = train_df[train_df["label"] == 1].dropna(subset=["explanation"])
    present_train = present_train[~present_train.index.isin(calib_ids)]
    sample = present_train.sample(KILL_N, random_state=1)  # different seed from calibration set

    recalls = []
    progress = ProgressReporter(len(sample), desc="Extracting", every=10)
    for _, r in tqdm(sample.iterrows(), total=len(sample), desc="Extracting"):
        span = run_extractor(str(r["text"]), examples_block)
        recalls.append(token_recall(span, str(r["explanation"])))
        progress.update()

    mean_recall = sum(recalls) / len(recalls)
    print(f"\nMean token recall of extracted span vs gold span ({KILL_N} posts): {mean_recall:.4f}")
    print(f"Call reliability: {llm_client.error_summary()}")
    if mean_recall < KILL_THRESHOLD:
        print(f"KILL CRITERION HIT: recall < {KILL_THRESHOLD}. Extractor finds the wrong evidence. Stop.")
    else:
        print("Kill criterion cleared - proceed to the full pipeline (`run_span_pipeline.py full`).")


def run_full():
    llm_client.reset_counters()
    df = load_dataset()
    train_df, _, _ = make_splits(df)
    rep_df = pd.read_csv(config.REP_DEV_PATH)

    extract_examples = build_extract_examples_block(train_df)
    judge_examples = build_judge_examples_block(train_df)

    records = []
    progress = ProgressReporter(len(rep_df), desc="Span pipeline", every=10)
    for idx, row in tqdm(rep_df.iterrows(), total=len(rep_df), desc="Span pipeline"):
        text = str(row["text"])
        true_label = int(row["label"])

        span = run_extractor(text, extract_examples)
        context = extract_context_sentence(text, span)

        judge_prompt = build_judge_prompt(span, context, judge_examples)
        raw = call_llm(judge_prompt, temperature=0.0)
        parsed = extract_json(raw)
        decision = str(parsed.get("decision", "Absent")).strip().lower()
        pred = 1 if "present" in decision else 0

        records.append({"id": idx, "true_label": true_label, "prediction": pred, "span": span, "text": text[:300]})
        progress.update()

    results = pd.DataFrame(records)
    out_path = _rep_out_path()
    results.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    acc = accuracy_score(results["true_label"], results["prediction"])
    f1 = f1_score(results["true_label"], results["prediction"], average="macro")
    print(f"\nModel: {config.MODEL_NAME}")
    print(f"Stage-2 verdict accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print(f"Call reliability: {llm_client.error_summary()}")
    if llm_client.error_count:
        print("WARNING: some calls failed even after retries - accuracy above may be distorted "
              "by NONE/Absent defaults. Treat as unreliable until re-run cleanly.")

    present_results = results[results["true_label"] == 1].join(rep_df[["explanation"]])
    recalls = [token_recall(r["span"], str(r["explanation"])) for _, r in present_results.iterrows()
               if pd.notna(r["explanation"])]
    if recalls:
        print(f"Stage-1 span recall vs gold (rep Present posts): {sum(recalls) / len(recalls):.4f}")


def run_test_set(save_every: int = 25, workers: int = 1):
    """Full 707-example held-out test-set run, checkpointed. workers>1 runs
    posts concurrently via a thread pool (see module docstring)."""
    llm_client.reset_counters()
    df = load_dataset()
    train_df, _, test_df = make_splits(df)

    extract_examples = build_extract_examples_block(train_df)
    judge_examples = build_judge_examples_block(train_df)

    checkpoint_path = _test_checkpoint_path()
    print(f"Model: {config.MODEL_NAME}  ->  checkpoint {checkpoint_path.name}  (workers={workers})")
    if checkpoint_path.exists():
        done = pd.read_csv(checkpoint_path)
        done_ids = set(done["original_index"].tolist())
        print(f"Resuming from checkpoint - {len(done_ids)} already done")
    else:
        done = pd.DataFrame()
        done_ids = set()
        print("Starting fresh full-test-set run")

    pending = [(idx, row) for idx, row in test_df.iterrows() if idx not in done_ids]
    new_rows = []
    progress = ProgressReporter(len(pending), desc="Span pipeline (test set)", every=10)

    def _checkpoint_if_due():
        if len(new_rows) % save_every == 0:
            partial = pd.DataFrame(new_rows)
            combined = pd.concat([done, partial], ignore_index=True)
            combined.to_csv(checkpoint_path, index=False)
            print(f"  checkpoint - {len(combined)} total")

    try:
        if workers <= 1:
            for idx, row in tqdm(pending, total=len(pending), desc="Span pipeline (test set)"):
                new_rows.append(_process_post(idx, row, extract_examples, judge_examples))
                progress.update()
                _checkpoint_if_due()
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_post, idx, row, extract_examples, judge_examples): idx
                    for idx, row in pending
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Span pipeline (test set, parallel)"):
                    new_rows.append(future.result())
                    progress.update()
                    _checkpoint_if_due()
    except LLMPersistentFailure as e:
        if new_rows:
            combined = pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_csv(checkpoint_path, index=False)
            print(f"Saved partial checkpoint ({len(combined)} rows) before aborting -> {checkpoint_path}")
        print(f"\nABORTED: {e}")
        print(f"Call reliability at abort: {llm_client.error_summary()}")
        raise

    if new_rows:
        combined = pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(checkpoint_path, index=False)
    else:
        combined = done

    print(f"Saved -> {checkpoint_path}")
    acc = accuracy_score(combined["true_label"], combined["prediction"])
    f1 = f1_score(combined["true_label"], combined["prediction"], average="macro")
    print(f"\n=== FULL TEST SET (model={config.MODEL_NAME}, n={len(combined)}) ===")
    print(f"Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print(f"Call reliability: {llm_client.error_summary()}")
    if llm_client.error_count:
        print("WARNING: some calls failed even after retries - accuracy above may be distorted "
              "by NONE/Absent defaults. Treat as unreliable until re-run cleanly.")
    print("Idea 5 baseline (gemini-3.5-flash-lite): Accuracy 0.8218, Macro-F1 0.8216")
    print("Paper's GloVe+BiGRU baseline: Accuracy 0.78, F1 ~0.79")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("kill", "full", "test"):
        print(__doc__)
        sys.exit(1)

    workers = 1
    if "--workers" in sys.argv:
        i = sys.argv.index("--workers")
        workers = int(sys.argv[i + 1])

    try:
        if sys.argv[1] == "kill":
            run_kill_check()
        elif sys.argv[1] == "full":
            run_full()
        else:
            run_test_set(workers=workers)
    except LLMPersistentFailure:
        sys.exit(1)

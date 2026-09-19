# Explain-Then-Classify

An evidence-bottlenecked two-stage LLM pipeline for detecting *lonesomeness*
(loneliness as a symptom of mental disturbance) in Reddit posts, evaluated on
the [LonXplain dataset](https://arxiv.org/abs/2305.18736) (Garg et al., 2023).

**Method.** Stage 1 has an LLM extract the minimal verbatim span of a post
that would justify a lonesomeness judgment (or `NONE`). Stage 2 decides
Present/Absent from *only that span and its containing sentence* — never the
full post. Both stages use one commercial LLM at temperature 0; the pipeline
is model-agnostic.

## Setup

```bash
python -m venv venv

# Windows:
venv\Scripts\pip.exe install -r requirements.txt
# macOS/Linux:
venv/bin/pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=<your key>
```

If `python`/`pip` don't resolve on your system, invoke your venv's interpreter
directly (e.g. `venv\Scripts\python.exe` on Windows).

## The dataset

`data/Lonesomenessdataset.csv` is **not** committed here. It belongs to the
LonXplain authors and is fetched from their repository automatically the first
time you run anything (`config.DATASET_URL` →
[`drmuskangarg/lonesomeness_dataset`](https://github.com/drmuskangarg/lonesomeness_dataset)).
The MIT licence in this repository covers this code, not that dataset.

## Reproducing the results

Every statistic reported in the paper can be recomputed from the committed
per-post prediction files, with **no API calls and no network access**:

```bash
python scripts/reviewer_response_stats.py
```

That prints accuracy on the 618 test posts never used during development, a
10-fold analysis of the held-out predictions, bootstrap confidence intervals,
and the paired comparison between the reported and development models.

To re-run the pipeline itself:

```bash
python scripts/run_span_pipeline.py kill   # stage-1 recall check on 50 train posts
python scripts/run_span_pipeline.py full   # 89-example subset run
python scripts/run_span_pipeline.py test   # full 707-example test-set run, checkpointed
```

**Note on `test` mode:** `data/span_pipeline_test_results.csv` already
contains all 707 rows from a completed run. Running `test` again will **not**
make any new API calls — it detects the complete checkpoint, reloads it, and
reprints accuracy/Macro-F1. `src/llm_client.py` configures the Gemini client
(and validates the API key) at *import time*, so `.env` still needs *some*
value in `GEMINI_API_KEY` — even a placeholder — just to get past that import.

**Note on `full` mode:** re-running this overwrites
`data/span_pipeline_rep_results.csv` with a fresh run's numbers, which may
differ slightly due to the API's non-determinism at temperature 0.

**Model comparison:** set `GEMINI_MODEL` to run the identical pipeline
against a different model — output/checkpoint filenames are model-aware, so
this never overwrites the default model's results.

```bash
GEMINI_MODEL=gemini-3.7-flash python scripts/run_span_pipeline.py test
GEMINI_MODEL=gemini-3.6-flash python scripts/run_span_pipeline.py test
```

Add `--workers N` to process posts concurrently via a thread pool (each post
is independent, so this is safe).

## File layout

```
config.py                  # paths, dataset URL/path, model name, .env loading
src/
  data.py                  # dataset download + column normalization
  llm_client.py             # Gemini API wrapper - retries transient failures with
                              #   backoff, aborts loudly on repeated back-to-back
                              #   failures instead of silently corrupting results
  parsing.py                # robust JSON extraction from LLM output
  progress.py                # progress logging for long LLM-calling loops
  splits.py                  # reproducible train/val/test split (seed 42)
  span_pipeline.py           # core method: prompt builders, calibration sampling,
                              #   context-sentence extraction, token P/R/F1 metrics
scripts/
  run_span_pipeline.py       # entry point: kill / full / test modes; test supports
                              #   --workers N for concurrent posts, and GEMINI_MODEL
                              #   for model comparisons
  make_representative_dev.py # documents how data/dev_representative.csv was built
  reviewer_response_stats.py # recomputes every reported statistic offline
prompts/
  span_extract.txt           # stage-1 extractor prompt
  span_judge.txt              # stage-2 judge prompt
data/
  Lonesomenessdataset.csv    # raw LonXplain dataset (3,522 posts) - NOT tracked;
                             #   downloaded from the source repository on first run
  dev_representative.csv     # 89-example stratified evaluation subset
  span_pipeline_rep_results.csv       # 89-example subset run
  span_pipeline_rep_results_run1.csv  # 89-example subset run (replicate)
  span_pipeline_test_results.csv               # full 707-example test-set results
  span_pipeline_test_results_gemini-3.7-flash.csv
  span_pipeline_test_results_gemini-3.6-flash.csv
```

## Citation

If you use this code or the method, please cite:

```bibtex
@inproceedings{yellanki2026explainthenclassify,
  title     = {Explain-Then-Classify: Evidence-Bottlenecked Classification
               for Explainable Lonesomeness Detection},
  author    = {Yellanki, Saanvi and Msvpj, Sathvik},
  booktitle = {IEEE Annual Information Technology, Electronics and Mobile
               Communication Conference (IEMCON)},
  year      = {2026}
}
```


## License

MIT — see [LICENSE](LICENSE).



"""Gemini client wrapper. The API key is never referenced directly here -
it comes only from config.require_api_key(), which reads it from the environment."""
import threading
import time

import google.generativeai as genai

import config

genai.configure(api_key=config.require_api_key())
_model = genai.GenerativeModel(config.MODEL_NAME)
_model_cache = {config.MODEL_NAME: _model}
_model_cache_lock = threading.Lock()

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 3.0
# A quota/billing wall (as opposed to a transient blip) fails every call, back to
# back. Abort after this many consecutive failures instead of silently grinding
# through hundreds of guaranteed-to-fail calls - without this, a quota wall can
# silently default every failed call to NONE/Absent and produce a fake-looking
# accuracy number instead of an obvious error.
_MAX_CONSECUTIVE_FAILURES = 8

error_count = 0
call_count = 0
_consecutive_failures = 0
_counters_lock = threading.Lock()


def get_model(model_name: str = None):
    """Returns a cached GenerativeModel for model_name (default: config.MODEL_NAME).
    Thread-safe so concurrent callers (see run_span_pipeline.py's --workers) don't
    race to create duplicate entries."""
    name = model_name or config.MODEL_NAME
    with _model_cache_lock:
        if name not in _model_cache:
            _model_cache[name] = genai.GenerativeModel(name)
        return _model_cache[name]


def reset_counters():
    """Call at the start of a run so error_summary() reflects just that run."""
    global error_count, call_count, _consecutive_failures
    with _counters_lock:
        error_count = 0
        call_count = 0
        _consecutive_failures = 0


def error_summary() -> str:
    rate = 100 * error_count / call_count if call_count else 0
    return f"{error_count}/{call_count} calls failed after retries ({rate:.1f}%)"


class LLMPersistentFailure(RuntimeError):
    """Raised when calls keep failing back-to-back - almost always a quota/billing
    wall, not a transient blip worth retrying through."""


def call_llm(prompt: str, temperature: float = 0.2, model_name: str = None) -> str:
    """Call the LLM and return raw text. model_name overrides config.MODEL_NAME
    for this call only (used for model-comparison runs). Retries transient
    failures with backoff; raises LLMPersistentFailure if failures keep coming
    back to back (see _MAX_CONSECUTIVE_FAILURES). Safe to call from multiple
    threads concurrently - counters are lock-protected, and each call still
    gets its own retry loop independent of other in-flight calls."""
    global error_count, call_count, _consecutive_failures
    with _counters_lock:
        call_count += 1
    model = get_model(model_name)
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            with _counters_lock:
                _consecutive_failures = 0
            return response.text
        except Exception as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                print(f"LLM error (attempt {attempt + 1}/{_MAX_RETRIES}, retrying in {delay:.0f}s): {e}")
                time.sleep(delay)

    with _counters_lock:
        error_count += 1
        _consecutive_failures += 1
        failures_now = _consecutive_failures
    print(f"LLM error (all {_MAX_RETRIES} attempts failed): {last_exc}")
    if failures_now >= _MAX_CONSECUTIVE_FAILURES:
        raise LLMPersistentFailure(
            f"{failures_now} consecutive calls failed - stopping instead of "
            f"grinding through what looks like a quota/billing wall. Last error: {last_exc}"
        )
    return ""

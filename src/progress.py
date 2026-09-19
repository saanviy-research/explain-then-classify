"""Periodic flushed progress prints for long LLM-calling loops.

tqdm's carriage-return bar doesn't render well in captured/non-interactive
logs, so this prints a discrete line every `every` items (and always on the
last item) with elapsed/ETA - visible in any log viewer, not just a live
terminal.
"""
import threading
import time


class ProgressReporter:
    def __init__(self, total: int, desc: str = "", every: int = 10):
        self.total = total
        self.desc = desc
        self.every = max(1, every)
        self.start = time.time()
        self.count = 0
        self._lock = threading.Lock()

    def update(self, n: int = 1):
        """Thread-safe: multiple worker threads (see run_span_pipeline.py's
        --workers) may call this concurrently."""
        with self._lock:
            self.count += n
            count = self.count
        if count % self.every == 0 or count == self.total:
            elapsed = time.time() - self.start
            rate = count / elapsed if elapsed > 0 else 0
            remaining = (self.total - count) / rate if rate > 0 else 0
            pct = 100 * count / self.total if self.total else 100
            prefix = f"{self.desc}: " if self.desc else ""
            print(
                f"{prefix}{count}/{self.total} ({pct:.0f}%) "
                f"- {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining",
                flush=True,
            )

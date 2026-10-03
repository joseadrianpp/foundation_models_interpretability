"""tqdm progress bars for the long loops (model forward passes, IG, Leiden,
GO3 pairs, g:Profiler queries).

In a terminal they are ordinary tqdm bars. When stderr goes to a file (the
logs of scripts/run_full.sh), or when several GPU workers share one output,
every refresh is written on its own line every LOG_INTERVAL seconds instead
of being redrawn with a carriage return, so the log stays readable.
"""

import os
import sys

from tqdm import tqdm

# Seconds between two progress lines in line mode (PROGRESS_INTERVAL overrides it)
LOG_INTERVAL = float(os.environ.get("PROGRESS_INTERVAL", 30))

_line_mode = not sys.stderr.isatty()


def use_line_mode() -> None:
    """One progress line per refresh (run_tasks' GPU workers share one output)."""
    global _line_mode
    _line_mode = True


class _LineStream:
    """File-like wrapper that turns tqdm's carriage-return redraws into lines."""

    def __init__(self, stream):
        self._stream = stream
        self._last = None

    def write(self, text: str) -> None:
        line = text.replace("\r", "").rstrip()
        # Closing a finished bar redraws it: skip a line whose count (the part
        # before the [elapsed<remaining, rate] stats) did not change
        count = line.split(" [")[0]
        if line and count != self._last:
            self._stream.write(line + "\n")
            self._last = count

    def flush(self) -> None:
        self._stream.flush()


def progress(iterable=None, **kwargs) -> tqdm:
    """tqdm over `iterable`, or a manual bar (`total=`, then `.update(n)`)."""
    if _line_mode:
        kwargs.setdefault("file", _LineStream(sys.stderr))
        kwargs.setdefault("mininterval", LOG_INTERVAL)
        kwargs.setdefault("maxinterval", LOG_INTERVAL)
    return tqdm(iterable, **kwargs)

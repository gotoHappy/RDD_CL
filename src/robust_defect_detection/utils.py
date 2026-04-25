import datetime
from pathlib import Path
import sys


def get_project_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return get_project_root() / path


def get_utc_time(fmt="%Y-%m-%d.%H-%M-%S"):
    return datetime.datetime.now(datetime.timezone.utc).strftime(fmt)


class ProgressTimer:
    def __init__(self, bar_length=30, prefix="", verbose=True):
        self.bar_length = bar_length
        self.prefix = prefix
        self.verbose = verbose
        self.total_items = 0
        self.items = 0
        self.start_time = None
        self.end_time = None
        self._last_line_length = 0

    def _write(self, message):
        if self.verbose:
            sys.stdout.write(message)
            sys.stdout.flush()

    @staticmethod
    def _format_seconds(seconds):
        seconds = max(float(seconds), 0.0)
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _render(self, postfix=""):
        if not self.verbose:
            return
        percent = 0.0 if self.total_items == 0 else 100.0 * self.items / self.total_items
        percent = max(0.0, min(100.0, round(percent, 1)))
        filled = int(self.bar_length * percent / 100.0)
        empty = self.bar_length - filled
        now = self.end_time or datetime.datetime.now(datetime.timezone.utc)
        start = self.start_time or now
        elapsed = (now - start).total_seconds()
        rate = self.items / elapsed if elapsed > 0 else 0.0
        remaining_items = max(self.total_items - self.items, 0)
        eta = remaining_items / rate if rate > 0 else 0.0
        bar = f"[{'=' * filled}{'.' * empty}]"
        line = (
            f"\r{self.prefix}{bar} "
            f"{percent:5.1f}% "
            f"{self.items}/{self.total_items} "
            f"| {rate:5.2f} it/s "
            f"| elapsed {self._format_seconds(elapsed)} "
            f"| eta {self._format_seconds(eta)}"
        )
        if postfix:
            line += f" | {postfix}"

        padding = max(self._last_line_length - len(line), 0)
        self._write(line + (" " * padding))
        self._last_line_length = len(line)

    def tic(self, total_items):
        self.total_items = total_items
        self.items = 0
        now = datetime.datetime.now(datetime.timezone.utc)
        self.start_time = now
        self.end_time = now
        self._last_line_length = 0
        self._render()

    def toc(self, add=1, postfix=""):
        self.items += add
        self.end_time = datetime.datetime.now(datetime.timezone.utc)
        self._render(postfix=postfix)

    def close(self):
        if self.verbose:
            self._write("\n")

    @property
    def total_seconds(self):
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

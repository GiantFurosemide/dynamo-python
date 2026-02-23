"""Runtime helpers: logging, error-file output, and progress display."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def _resolve_log_path(args=None, config: Optional[dict] = None) -> Optional[str]:
    """Resolve log path from CLI args/config."""
    if args is not None and getattr(args, "log_file", None):
        return str(getattr(args, "log_file"))
    if config:
        if config.get("error_log_file"):
            return str(config.get("error_log_file"))
        if config.get("log_file"):
            return str(config.get("log_file"))
    return None


def configure_logging(args=None, config: Optional[dict] = None, logger_name: Optional[str] = None) -> logging.Logger:
    """Configure root logger (stdout + optional file)."""
    level_name = "info"
    if args is not None and getattr(args, "log_level", None):
        level_name = str(getattr(args, "log_level"))
    elif config and config.get("log_level"):
        level_name = str(config.get("log_level"))
    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(stream_handler)

    log_path = _resolve_log_path(args=args, config=config)
    if log_path:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(lp), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
        root.addHandler(file_handler)

    return logging.getLogger(logger_name) if logger_name else root


def write_error(msg: str, args=None, config: Optional[dict] = None) -> None:
    """Append error message to resolved error log file."""
    path = _resolve_log_path(args=args, config=config)
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR {msg}\n")


def progress_iter(iterable: Iterable[T], total: Optional[int] = None, desc: str = "") -> Iterator[T]:
    """Simple terminal progress bar without external deps."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = 0

    count = 0
    width = 24
    prefix = f"{desc} " if desc else ""
    for item in iterable:
        count += 1
        if total > 0:
            frac = count / total
            done = int(width * frac)
            bar = "#" * done + "-" * (width - done)
            sys.stderr.write(f"\r{prefix}[{bar}] {count}/{total}")
        else:
            sys.stderr.write(f"\r{prefix}{count}")
        sys.stderr.flush()
        yield item
    if count > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()

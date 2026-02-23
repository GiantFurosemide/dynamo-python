"""Runtime helpers: logging, error-file output, and progress display."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def _resolve_path(raw_path: Optional[str], config_path: Optional[str]) -> Optional[str]:
    """Resolve raw path. Relative path is resolved against config file directory."""
    if not raw_path:
        return None
    p = Path(str(raw_path))
    if p.is_absolute() or not config_path:
        return str(p)
    return str((Path(config_path).resolve().parent / p).resolve())


def _default_log_paths(config_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Default log/error file paths in same directory as config."""
    if not config_path:
        return None, None
    cp = Path(config_path).resolve()
    return (
        str(cp.with_suffix(".log")),
        str(cp.with_suffix(".error.log")),
    )


def resolve_log_paths(args=None, config: Optional[dict] = None, config_path: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve (log_path, error_log_path).
    Priority:
      1) explicit args/config
      2) default path next to YAML config
    """
    default_log, default_err = _default_log_paths(config_path)
    cfg_log = _resolve_path(config.get("log_file") if config else None, config_path)
    cfg_err = _resolve_path(config.get("error_log_file") if config else None, config_path)
    arg_log = _resolve_path(getattr(args, "log_file", None) if args is not None else None, config_path)

    log_path = arg_log or cfg_log or default_log
    err_path = cfg_err or default_err
    return log_path, err_path


def configure_logging(args=None, config: Optional[dict] = None, logger_name: Optional[str] = None, config_path: Optional[str] = None) -> logging.Logger:
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

    log_path, _ = resolve_log_paths(args=args, config=config, config_path=config_path)
    if log_path:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(lp), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
        root.addHandler(file_handler)

    return logging.getLogger(logger_name) if logger_name else root


def write_error(msg: str, args=None, config: Optional[dict] = None, config_path: Optional[str] = None) -> None:
    """Append error message to resolved error log file."""
    _, path = resolve_log_paths(args=args, config=config, config_path=config_path)
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR {msg}\n")


def progress_iter(iterable: Iterable[T], total: Optional[int] = None, desc: str = "") -> Iterator[T]:
    """Progress iterator (silent by default)."""
    del total, desc
    for item in iterable:
        yield item

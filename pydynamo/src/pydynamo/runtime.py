"""Runtime helpers: logging, error-file output, and progress display."""
from __future__ import annotations

import logging
import os
import resource
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar

import mrcfile
import numpy as np
import yaml

T = TypeVar("T")
_RSS_TRACKERS: dict[int, dict] = {}


def _resolve_path(raw_path: Optional[str], config_path: Optional[str]) -> Optional[str]:
    """Resolve raw path. Relative path is resolved against config file directory."""
    if not raw_path:
        return None
    p = Path(str(raw_path))
    if p.is_absolute() or not config_path:
        return str(p)
    return str((Path(config_path).resolve().parent / p).resolve())


def resolve_path(raw_path: Optional[str], config_path: Optional[str]) -> Optional[str]:
    """Public path resolver for config-relative file paths."""
    return _resolve_path(raw_path, config_path)


def resolve_cpu_workers(config_value, default: int = 1) -> int:
    """
    Resolve CPU worker count for alignment/classification/reconstruction.
    default: when key missing or None.
    <=0: use max(1, cpu_count - 1).
    """
    if config_value is None:
        return default
    try:
        n = int(config_value)
    except (TypeError, ValueError):
        return default
    if n <= 0:
        return max(1, (os.cpu_count() or 1) - 1)
    return max(1, n)


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


def _format_duration(seconds: float) -> str:
    """Format duration seconds as HH:MM:SS."""
    s = int(max(0, round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _read_current_rss_mb() -> Optional[float]:
    """Best-effort current RSS (MB) from /proc/self/status on Linux."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Example: VmRSS:	  123456 kB
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = float(parts[1])
                        return kb / 1024.0
    except Exception:
        return None
    return None


def _read_peak_rss_mb() -> Optional[float]:
    """Best-effort peak RSS (MB) from ru_maxrss."""
    try:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if peak <= 0:
            return None
        # Linux ru_maxrss is KiB; macOS/BSD can be bytes.
        if os.name == "posix" and peak > 1024 * 1024:
            return peak / (1024.0 * 1024.0)
        return peak / 1024.0
    except Exception:
        return None


def _rss_observability_text(start_time: float, processed: int, total: int) -> str:
    """Track and format rss_cur/rss_avg/rss_peak for a progress stage."""
    key = int(round(float(start_time) * 1000))
    st = _RSS_TRACKERS.get(key)
    if st is None:
        st = {"samples": 0, "sum_cur_mb": 0.0, "max_cur_mb": 0.0}
        _RSS_TRACKERS[key] = st

    cur_mb = _read_current_rss_mb()
    if cur_mb is not None:
        st["samples"] += 1
        st["sum_cur_mb"] += float(cur_mb)
        if cur_mb > st["max_cur_mb"]:
            st["max_cur_mb"] = float(cur_mb)

    avg_mb = None
    if st["samples"] > 0:
        avg_mb = st["sum_cur_mb"] / st["samples"]
    peak_mb = _read_peak_rss_mb()

    cur_s = f"{cur_mb:.1f}MB" if cur_mb is not None else "unknown"
    avg_s = f"{avg_mb:.1f}MB" if avg_mb is not None else "unknown"
    peak_s = f"{peak_mb:.1f}MB" if peak_mb is not None else "unknown"

    if total > 0 and processed >= total:
        _RSS_TRACKERS.pop(key, None)
    return f"rss_cur={cur_s} rss_avg={avg_s} rss_peak={peak_s}"


def progress_timing_text(start_time: float, processed: int, total: int) -> str:
    """
    Build timing text for progress logs:
    - elapsed: elapsed wall time
    - eta: estimated remaining duration from average per-item cost
    - eta_at: estimated wall-clock finish time
    """
    now = time.time()
    elapsed = max(0.0, now - float(start_time))
    if processed <= 0 or total <= 0 or processed > total:
        timing_text = f"elapsed={_format_duration(elapsed)} eta=unknown eta_at=unknown"
        return f"{timing_text} {_rss_observability_text(start_time, processed, total)}"
    remaining = max(0, int(total) - int(processed))
    avg_per_item = elapsed / max(1, int(processed))
    eta_seconds = remaining * avg_per_item
    eta_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + eta_seconds))
    timing_text = (
        f"elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta_seconds)} "
        f"eta_at={eta_at}"
    )
    return f"{timing_text} {_rss_observability_text(start_time, processed, total)}"


def load_realspace_mask(
    mask_path: Optional[str],
    config_path: Optional[str] = None,
    expected_shape: Optional[tuple[int, int, int]] = None,
) -> Optional[np.ndarray]:
    """Load real-space mask as bool array; returns None when path is unset."""
    resolved = _resolve_path(mask_path, config_path)
    if not resolved:
        return None
    with mrcfile.open(resolved, mode="r", permissive=True) as mrc:
        mask = np.asarray(mrc.data, dtype=np.float32).copy()
    if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"nmask shape {tuple(mask.shape)} != expected {tuple(expected_shape)}"
        )
    mask_bool = mask > 0
    if not np.any(mask_bool):
        raise ValueError("nmask contains no positive voxels")
    return mask_bool


def log_command_inputs(
    logger: logging.Logger,
    command_name: str,
    config: Optional[dict] = None,
    config_path: Optional[str] = None,
    args=None,
    rest: Optional[list] = None,
) -> None:
    """Log effective command inputs (primarily YAML config) at command start."""
    payload = {
        "command": command_name,
        "config_path": config_path,
        "config": config or {},
    }
    if args is not None:
        payload["cli"] = {
            "log_level": getattr(args, "log_level", None),
            "log_file": getattr(args, "log_file", None),
            "json_errors": getattr(args, "json_errors", None),
        }
    if rest:
        payload["extra_args"] = list(rest)
    try:
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False).rstrip()
    except Exception:
        text = repr(payload)
    logger.info("Command inputs:\n%s", text)

"""pydynamo crop — extract subtomograms from tomograms."""
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mrcfile
import pandas as pd
import starfile
import yaml

from ..config_loader import load_config
from ..core.crop import crop_volume, save_subtomo
from ..io import read_dynamo_tbl, read_vll_to_df, dynamo_df_to_relion
from ..runtime import configure_logging, log_command_inputs, progress_iter, progress_timing_text, write_error

logger = logging.getLogger(__name__)


def _crop_one_with_volume(vol, x, y, z, sidelength, fill, output_dir, tag, row_dict):
    """Crop one particle using already-loaded tomogram volume."""
    position = (z, y, x)
    subtomo, _ = crop_volume(vol, sidelength, position, fill=fill)
    if subtomo is None:
        return None, False
    out_path = Path(output_dir) / f"particle_{tag:06d}.mrc"
    save_subtomo(subtomo, str(out_path))
    out_row = dict(row_dict)
    out_row["rlnImageName"] = out_path.name
    out_row["tag"] = tag
    out_row["x"] = x
    out_row["y"] = y
    out_row["z"] = z
    return out_row, True


def _process_tomo_group(tomo_path, group_tasks, num_workers: int, progress_log_every: int):
    """
    Process all particles for a single tomogram.
    Loads tomogram once (mmap view), then crops group tasks.
    """
    out_rows = []
    processed = 0
    failed = 0
    total = len(group_tasks)
    progress_start = time.time()
    try:
        with mrcfile.open(str(tomo_path), mode="r", permissive=True) as mrc:
            vol = mrc.data  # mmap-like view: avoid full copy for each task
            if num_workers > 1 and total > 1:
                with ThreadPoolExecutor(max_workers=min(num_workers, total)) as ex:
                    futures = {
                        ex.submit(
                            _crop_one_with_volume,
                            vol,
                            t["x"],
                            t["y"],
                            t["z"],
                            t["sidelength"],
                            t["fill"],
                            t["output_dir"],
                            t["tag"],
                            t["row_dict"],
                        ): t
                        for t in group_tasks
                    }
                    for f in progress_iter(as_completed(futures), total=len(futures), desc="crop"):
                        processed += 1
                        try:
                            out_row, ok = f.result()
                            if ok:
                                out_rows.append(out_row)
                            else:
                                failed += 1
                        except Exception:
                            failed += 1
                        if processed % progress_log_every == 0 or processed == total:
                            logger.info(
                                "Crop progress %d/%d (success=%d failed=%d, %s)",
                                processed, total, len(out_rows), failed,
                                progress_timing_text(progress_start, processed, total),
                            )
            else:
                for t in progress_iter(group_tasks, total=len(group_tasks), desc="crop"):
                    processed += 1
                    out_row, ok = _crop_one_with_volume(
                        vol,
                        t["x"],
                        t["y"],
                        t["z"],
                        t["sidelength"],
                        t["fill"],
                        t["output_dir"],
                        t["tag"],
                        t["row_dict"],
                    )
                    if ok:
                        out_rows.append(out_row)
                    else:
                        failed += 1
                    if processed % progress_log_every == 0 or processed == total:
                        logger.info(
                            "Crop progress %d/%d (success=%d failed=%d, %s)",
                            processed, total, len(out_rows), failed,
                            progress_timing_text(progress_start, processed, total),
                        )
    except Exception as e:
        logger.warning("Failed loading tomogram %s: %s", tomo_path, e)
        failed = total
    return out_rows, processed, failed


def run(config_path: str, rest: list, args) -> int:
    """Run crop command. Returns exit code."""
    try:
        config = load_config(config_path, "crop")
    except FileNotFoundError as e:
        _err(str(e), args)
    config["log_level"] = getattr(args, "log_level", config.get("log_level", "info"))
    configure_logging(args, config, __name__, config_path=config_path)
    log_command_inputs(logger, "crop", config=config, config_path=config_path, args=args, rest=rest)

    particles_in = config.get("particles")
    tomograms = config.get("tomograms")
    vll_path = config.get("vll") or config.get("vll_path")
    sidelength = int(config.get("sidelength"))
    output_star = config.get("output_star")
    output_dir = config.get("output_dir", "subtomos")
    fill = int(config.get("fill", -1))
    pixel_size = config.get("pixel_size")
    tomogram_size = config.get("tomogram_size")
    progress_log_every = max(1, int(config.get("progress_log_every", 10)))

    if not particles_in or not sidelength or not output_star:
        _err("Missing required: particles, sidelength, output_star", args, config=config, config_path=config_path)
        return 1

    if sidelength <= 0 or sidelength % 2 != 0:
        _err("sidelength must be positive and even", args, config=config, config_path=config_path)
        return 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = _resolve_num_workers(config.get("num_workers", 0))

    # Load particle table
    source_is_tbl = False
    if isinstance(particles_in, str):
        if particles_in.endswith(".star"):
            df = starfile.read(particles_in, always_dict=False)
            if isinstance(df, dict):
                df = df.get("particles", list(df.values())[0])
            if "rlnCoordinateX" in df.columns:
                x_col, y_col, z_col = "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"
            elif "rlnCenteredCoordinateXAngst" in df.columns:
                if not pixel_size or not tomogram_size:
                    _err("star with rlnCenteredCoordinate requires pixel_size and tomogram_size", args, config=config, config_path=config_path)
                ts = tuple(tomogram_size)
                df = _star_centered_to_absolute(df, pixel_size, ts)
                x_col, y_col, z_col = "x", "y", "z"
            else:
                _err("Star file must have rlnCoordinateX/Y/Z or rlnCenteredCoordinateXAngst/Y/Z", args, config=config, config_path=config_path)
            tomo_col = "rlnMicrographName" if "rlnMicrographName" in df.columns else "rlnTomoName"
            if tomo_col not in df.columns:
                _err(f"Star file must have {tomo_col}", args, config=config, config_path=config_path)
        else:
            source_is_tbl = True
            if not vll_path and tomograms is None:
                _err("tbl requires vll or tomograms for tomogram paths", args, config=config, config_path=config_path)
            df = read_dynamo_tbl(particles_in, vll_path=vll_path)
            x_col, y_col, z_col = "x", "y", "z"
            tomo_col = "rlnMicrographName" if "rlnMicrographName" in df.columns else "tomo"
    else:
        _err("particles must be path to .tbl or .star", args, config=config, config_path=config_path)

    vll_df = None
    if vll_path:
        vll_df = read_vll_to_df(vll_path)
    tomo_paths = _resolve_tomogram_paths(tomograms, vll_df)

    # Filter aligned if present
    if "aligned" in df.columns:
        df = df[df["aligned"].fillna(1) == 1].reset_index(drop=True)
    if "rlnImageName" not in df.columns:
        df["rlnImageName"] = None

    # Build task list
    tasks = []
    for idx, row in df.iterrows():
        tag = int(row.get("tag", idx + 1))
        x, y, z = float(row[x_col]), float(row[y_col]), float(row[z_col])
        tomo_key = row[tomo_col]
        tomo_path = tomo_paths.get(tomo_key) if tomo_paths else None
        if vll_df is not None and tomo_path is None:
            tid = row.get("tomo", tomo_key)
            if isinstance(tid, (int, float)):
                tid = int(tid)
                if 1 <= tid <= len(vll_df):
                    tomo_path = vll_df["tomo_path"].iloc[tid - 1]
            else:
                mask = vll_df["rlnMicrographName"] == str(tomo_key)
                if mask.any():
                    tomo_path = vll_df.loc[mask, "tomo_path"].iloc[0]
        if tomo_path is None:
            logger.warning("No tomogram path for particle %d (tomo=%s), skipping", tag, tomo_key)
            continue
        row_dict = row.to_dict()
        row_dict[tomo_col] = tomo_key
        tasks.append(
            {
                "tomo_path": str(tomo_path),
                "x": x,
                "y": y,
                "z": z,
                "sidelength": sidelength,
                "fill": fill,
                "output_dir": str(output_dir),
                "tag": tag,
                "row_dict": row_dict,
            }
        )

    out_rows = []
    total_tasks = len(tasks)
    processed = 0
    failed = 0
    run_progress_start = time.time()

    # Group by tomogram so each tomogram is loaded once.
    grouped = defaultdict(list)
    for t in tasks:
        grouped[t["tomo_path"]].append(t)

    tomo_groups = list(grouped.items())
    if len(tomo_groups) == 1:
        # Single tomogram: thread parallelism shares one loaded volume.
        rows, p, f = _process_tomo_group(
            tomo_groups[0][0],
            tomo_groups[0][1],
            num_workers=num_workers,
            progress_log_every=progress_log_every,
        )
        out_rows.extend(rows)
        processed += p
        failed += f
    else:
        # Multiple tomograms: parallelize by tomogram group.
        if num_workers <= 1 or len(tomo_groups) == 1:
            for tomo_path, group_tasks in tomo_groups:
                rows, p, f = _process_tomo_group(
                    tomo_path,
                    group_tasks,
                    num_workers=1,
                    progress_log_every=progress_log_every,
                )
                out_rows.extend(rows)
                processed += p
                failed += f
        else:
            max_group_workers = min(num_workers, len(tomo_groups))
            with ThreadPoolExecutor(max_workers=max_group_workers) as ex:
                futures = {
                    ex.submit(
                        _process_tomo_group,
                        tomo_path,
                        group_tasks,
                        1,
                        progress_log_every,
                    ): (tomo_path, len(group_tasks))
                    for tomo_path, group_tasks in tomo_groups
                }
                for f in progress_iter(as_completed(futures), total=len(futures), desc="crop"):
                    try:
                        rows, p, fail_cnt = f.result()
                        out_rows.extend(rows)
                        processed += p
                        failed += fail_cnt
                    except Exception as e:
                        _tomo_path, group_n = futures[f]
                        logger.warning("Crop tomogram-group failed: %s", e)
                        processed += group_n
                        failed += group_n

    if processed % progress_log_every != 0:
        logger.info(
            "Crop progress %d/%d (success=%d failed=%d, %s)",
            processed, total_tasks, len(out_rows), failed,
            progress_timing_text(run_progress_start, processed, total_tasks),
        )

    out_df = pd.DataFrame(out_rows)
    if not out_rows:
        _err("No particles were cropped", args, config=config, config_path=config_path)
        return 1

    # Write star
    out_df = _build_output_star_df(
        out_df,
        source_is_tbl=source_is_tbl,
        pixel_size=pixel_size,
        tomogram_size=tomogram_size,
    )
    out_star_path = Path(output_star)
    out_star_path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(out_df, str(out_star_path), overwrite=True)

    logger.info("Cropped %d particles to %s", len(out_rows), output_dir)
    logger.info("Output star: %s", output_star)
    return 0


def _resolve_num_workers(num_workers_value) -> int:
    """Resolve crop workers. <=0 means use all detected CPUs."""
    try:
        n = int(num_workers_value)
    except Exception:
        n = 0
    if n <= 0:
        return max(1, int(os.cpu_count() or 1))
    return n


def _build_output_star_df(df: pd.DataFrame, source_is_tbl: bool, pixel_size=None, tomogram_size=None) -> pd.DataFrame:
    """
    Build STAR output with RELION-style fields.
    - tbl input: convert from Dynamo columns to RELION columns.
    - star input: preserve RELION columns from input and ensure rlnImageName is updated.
    """
    if source_is_tbl:
        output_centered = (pixel_size is not None and tomogram_size is not None)
        relion_df = dynamo_df_to_relion(
            df,
            pixel_size=pixel_size,
            tomogram_size=tomogram_size,
            output_centered=output_centered,
        )
        if "rlnImageName" in df.columns:
            relion_df["rlnImageName"] = df["rlnImageName"].values
        if "tag" in df.columns:
            relion_df["rlnTomoParticleId"] = df["tag"].astype(int).values
        return relion_df

    # Source was star: keep relion-style fields only, plus updated image names.
    preferred_cols = [
        "rlnImageName",
        "rlnMicrographName",
        "rlnTomoName",
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnCoordinateZ",
        "rlnCenteredCoordinateXAngst",
        "rlnCenteredCoordinateYAngst",
        "rlnCenteredCoordinateZAngst",
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
        "rlnOriginXAngst",
        "rlnOriginYAngst",
        "rlnOriginZAngst",
    ]
    existing = [c for c in preferred_cols if c in df.columns]
    if existing:
        return df[existing].copy()
    return df.copy()


def _resolve_tomogram_paths(tomograms, vll_df):
    """Build tomo_key -> path mapping."""
    if tomograms is None:
        return {}
    paths = {}
    if isinstance(tomograms, str):
        if tomograms.endswith(".vll"):
            vll = read_vll_to_df(tomograms)
            for i, r in vll.iterrows():
                paths[r["rlnMicrographName"]] = r["tomo_path"]
                paths[i + 1] = r["tomo_path"]
        else:
            paths["1"] = tomograms
    elif isinstance(tomograms, list):
        for i, p in enumerate(tomograms):
            paths[i + 1] = p
            paths[str(i + 1)] = p
    return paths


def _star_centered_to_absolute(df, pixel_size, tomogram_size):
    """Convert rlnCenteredCoordinate to absolute pixels for cropping."""
    import numpy as np
    ts = np.array(tomogram_size)
    center = ts / 2.0
    x = df["rlnCenteredCoordinateXAngst"].values / pixel_size + center[0]
    y = df["rlnCenteredCoordinateYAngst"].values / pixel_size + center[1]
    z = df["rlnCenteredCoordinateZAngst"].values / pixel_size + center[2]
    df = df.copy()
    df["x"] = x
    df["y"] = y
    df["z"] = z
    return df




def _err(msg: str, args, code: int = 1, config=None, config_path=None):
    write_error(msg, args=args, config=config, config_path=config_path)
    if getattr(args, "json_errors", False):
        import json
        print(json.dumps({"error": msg, "code": code}), file=sys.stderr)
    else:
        print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)

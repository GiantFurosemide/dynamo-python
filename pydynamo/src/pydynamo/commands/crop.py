"""pydynamo crop — extract subtomograms from tomograms."""
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import starfile
import yaml

from ..config_loader import load_config
from ..core.crop import crop_volume, load_tomogram, save_subtomo
from ..io import read_dynamo_tbl, read_vll_to_df, dynamo_df_to_relion
from ..runtime import configure_logging, progress_iter, write_error

logger = logging.getLogger(__name__)


def _crop_one(tomo_path, x, y, z, sidelength, fill, output_dir, tag, row_dict):
    """Worker: load tomo, crop, save. Returns (out_row, True) or (None, False)."""
    try:
        vol = load_tomogram(tomo_path)
    except Exception:
        return None, False
    position = (z, y, x)
    subtomo, report = crop_volume(vol, sidelength, position, fill=fill)
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


def run(config_path: str, rest: list, args) -> int:
    """Run crop command. Returns exit code."""
    try:
        config = load_config(config_path, "crop")
    except FileNotFoundError as e:
        _err(str(e), args)
    config["log_level"] = getattr(args, "log_level", config.get("log_level", "info"))
    configure_logging(args, config, __name__, config_path=config_path)

    particles_in = config.get("particles")
    tomograms = config.get("tomograms")
    vll_path = config.get("vll") or config.get("vll_path")
    sidelength = int(config.get("sidelength"))
    output_star = config.get("output_star")
    output_dir = config.get("output_dir", "subtomos")
    fill = int(config.get("fill", -1))
    pixel_size = config.get("pixel_size")
    tomogram_size = config.get("tomogram_size")

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

    # Build task list: (tomo_path, x, y, z, tag, row_dict)
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
        tasks.append((tomo_path, x, y, z, sidelength, fill, str(output_dir), tag, row_dict))

    out_rows = []
    if num_workers <= 1:
        for t in progress_iter(tasks, total=len(tasks), desc="crop"):
            tomo_path, x, y, z, sl, fl, od, tag, rd = t
            out_row, ok = _crop_one(tomo_path, x, y, z, sl, fl, od, tag, rd)
            if ok:
                out_rows.append(out_row)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_crop_one, *t): t for t in tasks}
            for f in progress_iter(as_completed(futures), total=len(futures), desc="crop"):
                try:
                    out_row, ok = f.result()
                    if ok:
                        out_rows.append(out_row)
                except Exception as e:
                    logger.warning("Crop task failed: %s", e)

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

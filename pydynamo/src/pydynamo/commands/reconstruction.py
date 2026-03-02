"""pydynamo reconstruction — average subtomograms to produce density map."""
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
import yaml

from ..core.average import apply_inverse_transform, apply_symmetry
from ..core.wedge import get_wedge_mask
from ..io import read_dynamo_tbl
from ..runtime import (
    configure_logging,
    load_realspace_mask,
    log_command_inputs,
    progress_iter,
    progress_timing_text,
    resolve_cpu_workers,
    write_error,
)

logger = logging.getLogger(__name__)


def _resolve_particle_path(p_path, base_dir: Path, subtomograms) -> Path:
    """Resolve particle path with priority: absolute -> subtomograms dir -> particles dir."""
    p = Path(p_path)
    if p.is_absolute():
        return p
    if isinstance(subtomograms, str):
        sub_dir = Path(subtomograms)
        if sub_dir.is_dir():
            cand = sub_dir / p
            if cand.exists():
                return cand
    return base_dir / p


def _reconstruction_chunk_worker(payload: dict):
    """
    Process one chunk of particles: load, transform, accumulate.
    Chunk is processed in a streaming way: read one particle, transform, add to
    accumulator, discard (no full-chunk load in memory; jg_015 §3.2).
    payload: base_dir, subtomograms, paths_chunk, angles_chunk, shifts_chunk,
             sidelength, mask_path, config_path, apply_wedge, wedge_*, fcompensate.
    Returns (acc_or_fft_sum, n_acc) — acc is float64 for realspace, complex for Fourier.
    """
    from ..runtime import load_realspace_mask, resolve_path

    base_dir = Path(payload["base_dir"])
    subtomograms = payload["subtomograms"]
    paths_chunk = payload["paths_chunk"]
    angles_chunk = payload["angles_chunk"]
    shifts_chunk = payload["shifts_chunk"]
    sidelength = int(payload["sidelength"])
    config_path = payload["config_path"]
    apply_wedge = payload.get("apply_wedge", False)
    fcompensate = payload.get("fcompensate", False)

    real_mask = None
    if payload.get("mask_path") is not None:
        real_mask = load_realspace_mask(
            payload["mask_path"],
            config_path=config_path,
            expected_shape=(sidelength, sidelength, sidelength),
        )
    wedge_mask = None
    if apply_wedge:
        wedge_mask = get_wedge_mask(
            (sidelength, sidelength, sidelength),
            ftype=int(payload.get("wedge_ftype", 1)),
            ymintilt=float(payload.get("wedge_ymin", -48)),
            ymaxtilt=float(payload.get("wedge_ymax", 48)),
            xmintilt=float(payload.get("wedge_xmin", -60)),
            xmaxtilt=float(payload.get("wedge_xmax", 60)),
        )

    n_acc = 0
    if wedge_mask is not None:
        from scipy.fft import fftn, ifftn
        fft_sum = np.zeros((sidelength, sidelength, sidelength), dtype=np.complex128)
        for k, p_path in enumerate(paths_chunk):
            full_path = _resolve_particle_path(p_path, base_dir, subtomograms)
            try:
                with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                    vol = np.asarray(mrc.data, dtype=np.float32, copy=False)
            except Exception:
                continue
            if vol.shape[0] != sidelength or vol.shape[1] != sidelength or vol.shape[2] != sidelength:
                continue
            ang = angles_chunk[k]
            sh = shifts_chunk[k]
            tr = apply_inverse_transform(vol, ang[0], ang[1], ang[2], sh[0], sh[1], sh[2])
            if real_mask is not None:
                tr = tr * real_mask
            f = np.fft.fftshift(fftn(tr))
            f *= wedge_mask
            fft_sum += f
            n_acc += 1
        return (fft_sum, n_acc)
    else:
        acc = np.zeros((sidelength, sidelength, sidelength), dtype=np.float64)
        for k, p_path in enumerate(paths_chunk):
            full_path = _resolve_particle_path(p_path, base_dir, subtomograms)
            try:
                with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                    vol = np.asarray(mrc.data, dtype=np.float32, copy=False)
            except Exception:
                continue
            if vol.shape[0] != sidelength or vol.shape[1] != sidelength or vol.shape[2] != sidelength:
                continue
            ang = angles_chunk[k]
            sh = shifts_chunk[k]
            tr = apply_inverse_transform(vol, ang[0], ang[1], ang[2], sh[0], sh[1], sh[2])
            if real_mask is not None:
                tr = tr * real_mask
            acc += tr
            n_acc += 1
        return (acc, n_acc)


def run(config_path: str, rest: list, args) -> int:
    """Run reconstruction command. Returns exit code."""
    config = _load_config(config_path, args)
    configure_logging(args, config, __name__, config_path=config_path)
    log_command_inputs(logger, "reconstruction", config=config, config_path=config_path, args=args, rest=rest)

    particles = config.get("particles")
    subtomograms = config.get("subtomograms")
    output = config.get("output")
    sidelength = config.get("sidelength")
    symmetry = config.get("symmetry", "c1")
    tags = config.get("tags")  # optional filter
    vll_path = config.get("vll") or config.get("vll_path")
    apply_wedge = config.get("apply_wedge", False)
    wedge_ftype = int(config.get("wedge_ftype", 1))
    wedge_ymin = float(config.get("wedge_ymin", -48))
    wedge_ymax = float(config.get("wedge_ymax", 48))
    wedge_xmin = float(config.get("wedge_xmin", -60))
    wedge_xmax = float(config.get("wedge_xmax", 60))
    fcompensate = config.get("fcompensate", False)
    progress_log_every = max(1, int(config.get("progress_log_every", 10)))
    recon_workers = resolve_cpu_workers(config.get("recon_workers"), default=1)
    mask_path = config.get("nmask")
    mask_consistency_min_fraction = float(config.get("mask_consistency_min_fraction", 0.01))

    if not all([particles, subtomograms, output, sidelength]):
        _err("Missing required: particles, subtomograms, output, sidelength", args, config=config, config_path=config_path)
        return 1

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load table
    pixel_size = config.get("pixel_size", 1.0)
    if isinstance(particles, str):
        if particles.endswith(".star"):
            data = starfile.read(particles, always_dict=False)
            if isinstance(data, dict):
                tbl_df = data.get("particles", list(data.values())[0])
            else:
                tbl_df = data
            if "rlnAngleRot" in tbl_df.columns:
                from ..io import convert_euler
                ang_zyz = tbl_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].values
                ang_zxz = convert_euler(ang_zyz, src_convention="relion", dst_convention="dynamo", degrees=True)
                angles = np.atleast_2d(ang_zxz)
            elif "tdrot" in tbl_df.columns:
                angles = np.column_stack([
                    tbl_df["tdrot"].values, tbl_df["tilt"].values, tbl_df["narot"].values
                ])
            else:
                angles = np.zeros((len(tbl_df), 3))
            if "rlnOriginXAngst" in tbl_df.columns:
                shifts = np.column_stack([
                    -tbl_df["rlnOriginXAngst"].values / pixel_size,
                    -tbl_df["rlnOriginYAngst"].values / pixel_size,
                    -tbl_df["rlnOriginZAngst"].values / pixel_size,
                ])
            else:
                shifts = np.zeros((len(tbl_df), 3))
            paths_list = tbl_df["rlnImageName"].tolist() if "rlnImageName" in tbl_df.columns else None
        else:
            tbl_df = read_dynamo_tbl(particles, vll_path=vll_path)
            angles = np.column_stack([
                tbl_df["tdrot"].values if "tdrot" in tbl_df.columns else np.zeros(len(tbl_df)),
                tbl_df["tilt"].values if "tilt" in tbl_df.columns else np.zeros(len(tbl_df)),
                tbl_df["narot"].values if "narot" in tbl_df.columns else np.zeros(len(tbl_df)),
            ])
            shifts = np.column_stack([
                tbl_df["dx"].values if "dx" in tbl_df.columns else np.zeros(len(tbl_df)),
                tbl_df["dy"].values if "dy" in tbl_df.columns else np.zeros(len(tbl_df)),
                tbl_df["dz"].values if "dz" in tbl_df.columns else np.zeros(len(tbl_df)),
            ])
            paths_list = None
    else:
        _err("particles must be path to .tbl or .star", args)

    # Resolve subtomogram paths
    if paths_list is None:
        if isinstance(subtomograms, str):
            sub_dir = Path(subtomograms)
            if sub_dir.is_dir():
                paths_list = sorted(sub_dir.glob("*.mrc")) + sorted(sub_dir.glob("*.mrcs"))
                paths_list = [str(p) for p in paths_list]
            else:
                paths_list = [subtomograms] * len(tbl_df)
        elif isinstance(subtomograms, list):
            paths_list = subtomograms
        else:
            _err("subtomograms must be path or list; or particles star must have rlnImageName", args, config=config, config_path=config_path)

    # Filter by tags / averaged
    if "averaged" in tbl_df.columns:
        mask = tbl_df["averaged"].fillna(0) == 1
        tbl_df = tbl_df[mask]
        angles = angles[mask]
        shifts = shifts[mask]
        paths_list = [p for p, m in zip(paths_list, mask) if m]
    if tags is not None:
        tag_col = "tag" if "tag" in tbl_df.columns else tbl_df.columns[0]
        mask = tbl_df[tag_col].isin(tags)
        tbl_df = tbl_df[mask]
        angles = angles[mask]
        shifts = shifts[mask]
        paths_list = [p for p, m in zip(paths_list, mask) if m]

    n = len(tbl_df)
    if n == 0:
        _err("No particles to average", args, config=config, config_path=config_path)
        return 1
    if len(paths_list) < n:
        paths_list = (paths_list * (n // len(paths_list) + 1))[:n]

    try:
        real_mask = load_realspace_mask(
            mask_path,
            config_path=config_path,
            expected_shape=(int(sidelength), int(sidelength), int(sidelength)),
        )
    except Exception as e:
        _err(f"Failed to load nmask: {e}", args, config=config, config_path=config_path)
        return 1
    if real_mask is not None:
        frac = float(np.mean(real_mask))
        logger.info("Reconstruction mask coverage: %.4f", frac)
        if frac < mask_consistency_min_fraction:
            logger.warning(
                "Reconstruction mask coverage %.4f < threshold %.4f; average stability may degrade",
                frac,
                mask_consistency_min_fraction,
            )

    # Load and transform
    base_dir = Path(particles).parent if isinstance(particles, str) else Path(".")
    wedge_mask = None
    if apply_wedge:
        wedge_mask = get_wedge_mask(
            (sidelength, sidelength, sidelength),
            ftype=wedge_ftype,
            ymintilt=wedge_ymin,
            ymaxtilt=wedge_ymax,
            xmintilt=wedge_xmin,
            xmaxtilt=wedge_xmax,
        )
        logger.info(
            "Wedge ftype=%d y=[%.0f,%.0f] x=[%.0f,%.0f]; fcompensate=%s",
            wedge_ftype, wedge_ymin, wedge_ymax, wedge_xmin, wedge_xmax, fcompensate,
        )

    if wedge_mask is not None:
        # Fourier-space average with wedge
        from scipy.fft import fftn, ifftn
        progress_start = time.time()
        n_acc = 0
        n_fail = 0
        fft_sum = np.zeros((sidelength, sidelength, sidelength), dtype=np.complex128)
        if recon_workers > 1 and len(paths_list) > 1:
            chunk_size = max(1, (len(paths_list) + recon_workers - 1) // recon_workers)
            payloads = []
            for start in range(0, len(paths_list), chunk_size):
                end = min(start + chunk_size, len(paths_list))
                paths_chunk = paths_list[start:end]
                angles_chunk = [tuple(angles[i]) for i in range(start, end)]
                shifts_chunk = [tuple(shifts[i]) for i in range(start, end)]
                payloads.append({
                    "base_dir": str(base_dir),
                    "subtomograms": subtomograms,
                    "paths_chunk": paths_chunk,
                    "angles_chunk": angles_chunk,
                    "shifts_chunk": shifts_chunk,
                    "sidelength": sidelength,
                    "mask_path": mask_path,
                    "config_path": config_path,
                    "apply_wedge": True,
                    "wedge_ftype": wedge_ftype,
                    "wedge_ymin": wedge_ymin,
                    "wedge_ymax": wedge_ymax,
                    "wedge_xmin": wedge_xmin,
                    "wedge_xmax": wedge_xmax,
                    "fcompensate": fcompensate,
                })
            logger.info("Reconstruction Fourier path with recon_workers=%d", recon_workers)
            with ProcessPoolExecutor(max_workers=recon_workers) as ex:
                futures = [ex.submit(_reconstruction_chunk_worker, p) for p in payloads]
                for f in progress_iter(as_completed(futures), total=len(futures), desc="recon"):
                    part_fft, part_n = f.result()
                    fft_sum += part_fft
                    n_acc += part_n
        else:
            n_proc = 0
            for i, p_path in progress_iter(list(enumerate(paths_list)), total=len(paths_list), desc="recon"):
                full_path = _resolve_particle_path(p_path, base_dir, subtomograms)
                n_proc += 1
                # Read-only view + copy only when not writeable (P2 particle I/O strategy).
                try:
                    with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                        vol = np.asarray(mrc.data, dtype=np.float32, copy=False)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", full_path, e)
                    n_fail += 1
                    if n_proc % progress_log_every == 0 or n_proc == len(paths_list):
                        logger.info(
                            "Recon progress %d/%d (used=%d failed=%d, %s)",
                            n_proc, len(paths_list), n_acc, n_fail,
                            progress_timing_text(progress_start, n_proc, len(paths_list)),
                        )
                    continue
                if vol.shape[0] != sidelength or vol.shape[1] != sidelength or vol.shape[2] != sidelength:
                    logger.warning("Particle %s size %s != sidelength %s", full_path, vol.shape, sidelength)
                    n_fail += 1
                    if n_proc % progress_log_every == 0 or n_proc == len(paths_list):
                        logger.info(
                            "Recon progress %d/%d (used=%d failed=%d, %s)",
                            n_proc, len(paths_list), n_acc, n_fail,
                            progress_timing_text(progress_start, n_proc, len(paths_list)),
                        )
                    continue
                tr = apply_inverse_transform(
                    vol,
                    angles[i, 0], angles[i, 1], angles[i, 2],
                    shifts[i, 0], shifts[i, 1], shifts[i, 2],
                )
                if real_mask is not None:
                    tr = tr * real_mask
                f = np.fft.fftshift(fftn(tr))
                f *= wedge_mask
                fft_sum += f
                n_acc += 1
                if n_proc % progress_log_every == 0 or n_proc == len(paths_list):
                    logger.info(
                        "Recon progress %d/%d (used=%d failed=%d, %s)",
                        n_proc, len(paths_list), n_acc, n_fail,
                        progress_timing_text(progress_start, n_proc, len(paths_list)),
                    )
        if n_acc == 0:
            _err("No valid particles loaded", args, config=config, config_path=config_path)
            return 1
        denom = wedge_mask * n_acc if fcompensate else (np.ones_like(wedge_mask) * n_acc)
        denom = np.maximum(denom, 1e-12)
        avg = np.real(ifftn(np.fft.ifftshift(fft_sum / denom))).astype(np.float32)
    else:
        # Stream-friendly accumulation: avoid storing all transformed particles.
        progress_start = time.time()
        n_acc = 0
        n_fail = 0
        acc = np.zeros((sidelength, sidelength, sidelength), dtype=np.float64)
        if recon_workers > 1 and len(paths_list) > 1:
            chunk_size = max(1, (len(paths_list) + recon_workers - 1) // recon_workers)
            payloads = []
            for start in range(0, len(paths_list), chunk_size):
                end = min(start + chunk_size, len(paths_list))
                paths_chunk = paths_list[start:end]
                angles_chunk = [tuple(angles[i]) for i in range(start, end)]
                shifts_chunk = [tuple(shifts[i]) for i in range(start, end)]
                payloads.append({
                    "base_dir": str(base_dir),
                    "subtomograms": subtomograms,
                    "paths_chunk": paths_chunk,
                    "angles_chunk": angles_chunk,
                    "shifts_chunk": shifts_chunk,
                    "sidelength": sidelength,
                    "mask_path": mask_path,
                    "config_path": config_path,
                    "apply_wedge": False,
                    "fcompensate": fcompensate,
                })
            logger.info("Reconstruction realspace path with recon_workers=%d", recon_workers)
            with ProcessPoolExecutor(max_workers=recon_workers) as ex:
                futures = [ex.submit(_reconstruction_chunk_worker, p) for p in payloads]
                for f in progress_iter(as_completed(futures), total=len(futures), desc="recon"):
                    part_acc, part_n = f.result()
                    acc += part_acc
                    n_acc += part_n
        else:
            n_proc = 0
            for i, p_path in progress_iter(list(enumerate(paths_list)), total=len(paths_list), desc="recon"):
                full_path = _resolve_particle_path(p_path, base_dir, subtomograms)
                n_proc += 1
                # Read-only view + copy only when not writeable (P2 particle I/O strategy).
                try:
                    with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                        vol = np.asarray(mrc.data, dtype=np.float32, copy=False)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", full_path, e)
                    n_fail += 1
                    if n_proc % progress_log_every == 0 or n_proc == len(paths_list):
                        logger.info(
                            "Recon progress %d/%d (used=%d failed=%d, %s)",
                            n_proc, len(paths_list), n_acc, n_fail,
                            progress_timing_text(progress_start, n_proc, len(paths_list)),
                        )
                    continue
                if vol.shape[0] != sidelength or vol.shape[1] != sidelength or vol.shape[2] != sidelength:
                    logger.warning("Particle %s size %s != sidelength %s", full_path, vol.shape, sidelength)
                    n_fail += 1
                    if n_proc % progress_log_every == 0 or n_proc == len(paths_list):
                        logger.info(
                            "Recon progress %d/%d (used=%d failed=%d, %s)",
                            n_proc, len(paths_list), n_acc, n_fail,
                            progress_timing_text(progress_start, n_proc, len(paths_list)),
                        )
                    continue
                tr = apply_inverse_transform(
                    vol,
                    angles[i, 0], angles[i, 1], angles[i, 2],
                    shifts[i, 0], shifts[i, 1], shifts[i, 2],
                )
                if real_mask is not None:
                    tr = tr * real_mask
                acc += tr
                n_acc += 1
                if n_proc % progress_log_every == 0 or n_proc == len(paths_list):
                    logger.info(
                        "Recon progress %d/%d (used=%d failed=%d, %s)",
                        n_proc, len(paths_list), n_acc, n_fail,
                        progress_timing_text(progress_start, n_proc, len(paths_list)),
                    )
        if n_acc == 0:
            _err("No valid particles loaded", args, config=config, config_path=config_path)
            return 1
        avg = (acc / n_acc).astype(np.float32)

    avg = apply_symmetry(avg, symmetry)

    with mrcfile.new(str(output), overwrite=True) as mrc:
        mrc.set_data(avg.astype(np.float32))
        try:
            mrc.voxel_size = float(pixel_size)
        except Exception:
            logger.warning("Failed to set output voxel_size from pixel_size=%s", pixel_size)

    n_avg = n_acc
    logger.info("Averaged %d particles to %s", n_avg, output)
    return 0


def _load_config(path: str, args=None) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        if args:
            _err(f"Config file not found: {path}", args, config_path=path)
        raise


def _err(msg: str, args, code: int = 1, config=None, config_path=None):
    write_error(msg, args=args, config=config, config_path=config_path)
    if getattr(args, "json_errors", False):
        import json
        print(json.dumps({"error": msg, "code": code}), file=sys.stderr)
    else:
        print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)

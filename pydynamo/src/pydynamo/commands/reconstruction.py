"""pydynamo reconstruction — average subtomograms to produce density map."""
import logging
import sys
from pathlib import Path

import numpy as np
import mrcfile
import pandas as pd
import starfile
import yaml

from ..core.average import apply_inverse_transform, apply_symmetry
from ..core.wedge import get_wedge_mask
from ..io import read_dynamo_tbl

logger = logging.getLogger(__name__)


def run(config_path: str, rest: list, args) -> int:
    """Run reconstruction command. Returns exit code."""
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    config = _load_config(config_path, args)

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

    if not all([particles, subtomograms, output, sidelength]):
        _err("Missing required: particles, subtomograms, output, sidelength", args)
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
                angles = ang_zxz
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
            _err("subtomograms must be path or list; or particles star must have rlnImageName", args)

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
        _err("No particles to average", args)
        return 1
    if len(paths_list) < n:
        paths_list = (paths_list * (n // len(paths_list) + 1))[:n]

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
        n_acc = 0
        fft_sum = np.zeros((sidelength, sidelength, sidelength), dtype=np.complex128)
        for i, p_path in enumerate(paths_list):
            full_path = Path(p_path)
            if not full_path.is_absolute():
                full_path = base_dir / p_path
            try:
                with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                    vol = mrc.data.copy()
            except Exception as e:
                logger.warning("Failed to load %s: %s", full_path, e)
                continue
            if vol.shape[0] != sidelength or vol.shape[1] != sidelength or vol.shape[2] != sidelength:
                logger.warning("Particle %s size %s != sidelength %s", full_path, vol.shape, sidelength)
                continue
            tr = apply_inverse_transform(
                vol,
                angles[i, 0], angles[i, 1], angles[i, 2],
                shifts[i, 0], shifts[i, 1], shifts[i, 2],
            )
            f = np.fft.fftshift(fftn(tr))
            f *= wedge_mask
            fft_sum += f
            n_acc += 1
        if n_acc == 0:
            _err("No valid particles loaded", args)
            return 1
        denom = wedge_mask * n_acc if fcompensate else (np.ones_like(wedge_mask) * n_acc)
        denom = np.maximum(denom, 1e-12)
        avg = np.real(ifftn(np.fft.ifftshift(fft_sum / denom))).astype(np.float32)
    else:
        particles_data = []
        for i, p_path in enumerate(paths_list):
            full_path = Path(p_path)
            if not full_path.is_absolute():
                full_path = base_dir / p_path
            try:
                with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                    vol = mrc.data.copy()
            except Exception as e:
                logger.warning("Failed to load %s: %s", full_path, e)
                continue
            if vol.shape[0] != sidelength or vol.shape[1] != sidelength or vol.shape[2] != sidelength:
                logger.warning("Particle %s size %s != sidelength %s", full_path, vol.shape, sidelength)
                continue
            tr = apply_inverse_transform(
                vol,
                angles[i, 0], angles[i, 1], angles[i, 2],
                shifts[i, 0], shifts[i, 1], shifts[i, 2],
            )
            particles_data.append(tr)
        if not particles_data:
            _err("No valid particles loaded", args)
            return 1
        avg = sum(particles_data) / len(particles_data)

    avg = apply_symmetry(avg, symmetry)

    with mrcfile.new(str(output), overwrite=True) as mrc:
        mrc.set_data(avg.astype(np.float32))

    n_avg = n_acc if wedge_mask is not None else len(particles_data)
    logger.info("Averaged %d particles to %s", n_avg, output)
    return 0


def _load_config(path: str, args=None) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        if args:
            _err(f"Config file not found: {path}", args)
        raise


def _err(msg: str, args, code: int = 1):
    if getattr(args, "json_errors", False):
        import json
        print(json.dumps({"error": msg, "code": code}), file=sys.stderr)
    else:
        print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)

"""pydynamo classification — MRA: multireference alignment and classification."""
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
import yaml

from ..core.align import align_one_particle
from ..core.average import apply_inverse_transform, apply_symmetry
from ..io import create_dynamo_table, convert_euler, read_dynamo_tbl
from ..runtime import configure_logging, progress_iter, write_error

logger = logging.getLogger(__name__)


def run(config_path: str, rest: list, args) -> int:
    """Run classification (MRA) command. Returns exit code."""
    config = _load_config(config_path, args)
    configure_logging(args, config, __name__)

    particles = config.get("particles")
    subtomograms = config.get("subtomograms")
    references = config.get("references")
    tables = config.get("tables")
    output_dir = Path(config.get("output_dir", "mra_output"))
    max_iterations = int(config.get("max_iterations", 5))
    swap = config.get("swap", True)
    cone_step = float(config.get("cone_step", 15))
    cone_range = tuple(config.get("cone_range", [0, 180]))
    inplane_step = float(config.get("inplane_step", 15))
    inplane_range = tuple(config.get("inplane_range", [0, 360]))
    shift_search = int(config.get("shift_search", 3))
    lowpass = config.get("lowpass")
    pixel_size = float(config.get("pixel_size", 1.0))
    multigrid_levels = int(config.get("multigrid_levels", 1))
    device = str(config.get("device", "auto"))
    device_id = config.get("device_id")
    gpu_ids = config.get("gpu_ids")

    if not all([particles, subtomograms, references, output_dir]):
        _err("Missing required: particles, subtomograms, references, output_dir", args, config=config)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    refs = references if isinstance(references, list) else [references]
    nref = len(refs)

    # Load particle paths
    if particles.endswith(".star"):
        tbl_df = starfile.read(particles, always_dict=False)
        if isinstance(tbl_df, dict):
            tbl_df = tbl_df.get("particles", list(tbl_df.values())[0])
        paths = tbl_df["rlnImageName"].tolist() if "rlnImageName" in tbl_df.columns else None
    else:
        tbl_df = read_dynamo_tbl(particles, vll_path=config.get("vll"))
        paths = None

    if paths is None:
        sub_dir = Path(subtomograms)
        paths = sorted(sub_dir.glob("*.mrc")) + sorted(sub_dir.glob("*.mrcs"))
        paths = [str(p) for p in paths]
    base_dir = Path(particles).parent if isinstance(particles, str) else Path(".")

    # Load references
    ref_vols = []
    for r in refs:
        with mrcfile.open(r, mode="r", permissive=True) as mrc:
            ref_vols.append(mrc.data.copy().astype(np.float32))
    sidelength = ref_vols[0].shape[0]

    # Initial per-ref tables (swap: each particle in all refs)
    if tables and isinstance(tables, list) and len(tables) == nref:
        ref_tables = [read_dynamo_tbl(t) for t in tables]
    else:
        ref_tables = [tbl_df.copy() for _ in range(nref)]

    resolved_device, gpu_ids = _resolve_execution_devices(device, device_id, gpu_ids)

    def _align_particle_task(pidx, p_path):
        full_path = base_dir / p_path if not Path(p_path).is_absolute() else Path(p_path)
        try:
            with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                part = mrc.data.copy().astype(np.float32)
        except Exception:
            return None, None
        if part.shape != ref_vols[0].shape:
            return None, None

        best_cc = -2
        best_ref = 0
        best_row = None
        refs_to_align = range(nref) if swap else [max(0, min(nref - 1, int(ref_tables[0].iloc[pidx].get("ref", 1)) - 1))]
        local_device_id = None
        if resolved_device == "cuda" and gpu_ids:
            local_device_id = gpu_ids[pidx % len(gpu_ids)]

        for r in refs_to_align:
            tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
                part, ref_vols[r],
                cone_step=cone_step, cone_range=cone_range,
                inplane_step=inplane_step, inplane_range=inplane_range,
                shift_search=shift_search,
                lowpass_angstrom=lowpass, pixel_size=pixel_size,
                multigrid_levels=multigrid_levels, device=resolved_device, device_id=local_device_id,
            )
            row = tbl_df.iloc[pidx].to_dict() if pidx < len(tbl_df) else {}
            row.update({
                "tag": pidx + 1,
                "tdrot": tdrot, "tilt": tilt, "narot": narot,
                "dx": dx, "dy": dy, "dz": dz,
                "cc": cc, "cc2": cc, "ref": r + 1,
                "aligned": 1, "averaged": 1,
                "x": row.get("x", 0), "y": row.get("y", 0), "z": row.get("z", 0),
                "rlnImageName": str(full_path),
                "particle_idx": pidx,
            })
            if cc > best_cc:
                best_cc = cc
                best_ref = r
                best_row = row
        return best_ref, best_row

    for ite in range(max_iterations):
        logger.info("MRA iteration %d/%d", ite + 1, max_iterations)
        # Compute: align each particle against each ref (swap) or assigned ref only
        refined_per_ref = [[] for _ in range(nref)]
        if resolved_device == "cuda" and len(gpu_ids) > 1 and len(paths) > 1:
            logger.info("Classification multi-GPU scheduling on devices: %s", gpu_ids)
            with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
                futures = [ex.submit(_align_particle_task, pidx, p_path) for pidx, p_path in enumerate(paths)]
                for f in progress_iter(as_completed(futures), total=len(futures), desc=f"classify ite{ite+1}"):
                    best_ref, best_row = f.result()
                    if best_row is not None:
                        best_row["ref"] = best_ref + 1
                        best_row["grep_average"] = 1
                        refined_per_ref[best_ref].append(best_row)
        else:
            for pidx, p_path in progress_iter(list(enumerate(paths)), total=len(paths), desc=f"classify ite{ite+1}"):
                best_ref, best_row = _align_particle_task(pidx, p_path)
                if best_row is not None:
                    best_row["ref"] = best_ref + 1
                    best_row["grep_average"] = 1
                    refined_per_ref[best_ref].append(best_row)

        # Assemble + MRA: re-average per ref
        for r in range(nref):
            if not refined_per_ref[r]:
                continue
            sub_df = pd.DataFrame(refined_per_ref[r])
            # Average particles in this ref
            angles = np.column_stack([sub_df["tdrot"], sub_df["tilt"], sub_df["narot"]])
            shifts = np.column_stack([sub_df["dx"], sub_df["dy"], sub_df["dz"]])

            particles_data = []
            for i in range(len(sub_df)):
                p_path = sub_df.iloc[i].get("rlnImageName") or paths[min(sub_df.iloc[i].get("particle_idx", i), len(paths) - 1)]
                full = Path(p_path) if Path(p_path).is_absolute() else base_dir / p_path
                try:
                    with mrcfile.open(str(full), mode="r", permissive=True) as mrc:
                        vol = mrc.data.copy()
                except Exception:
                    continue
                tr = apply_inverse_transform(
                    vol,
                    angles[i, 0], angles[i, 1], angles[i, 2],
                    shifts[i, 0], shifts[i, 1], shifts[i, 2],
                )
                particles_data.append(tr)

            if particles_data:
                avg = sum(particles_data) / len(particles_data)
                avg_path = output_dir / f"ite_{ite+1:03d}" / f"average_ref_{r+1:03d}.mrc"
                avg_path.parent.mkdir(parents=True, exist_ok=True)
                with mrcfile.new(str(avg_path), overwrite=True) as mrc:
                    mrc.set_data(avg.astype(np.float32))
                ref_vols[r] = avg

            tbl_path = output_dir / f"ite_{ite+1:03d}" / f"refined_table_ref_{r+1:03d}.tbl"
            tbl_path.parent.mkdir(parents=True, exist_ok=True)
            sub = pd.DataFrame(refined_per_ref[r])
            if len(sub) > 0:
                coords = sub[["x", "y", "z"]].values if all(c in sub.columns for c in ["x", "y", "z"]) else np.zeros((len(sub), 3))
                ang_zxz = np.column_stack([sub["tdrot"], sub["tilt"], sub["narot"]])
                ang_zyz = convert_euler(ang_zxz, src_convention="dynamo", dst_convention="relion", degrees=True)
                ang_zyz = np.atleast_2d(ang_zyz)
                create_dynamo_table(
                    coords,
                    angles_zyz=ang_zyz,
                    micrograph_names=sub["rlnMicrographName"].tolist() if "rlnMicrographName" in sub.columns else None,
                    origins=sub[["dx", "dy", "dz"]].values,
                    output_file=str(tbl_path),
                    ref=r + 1,
                )

    logger.info("MRA complete: %d iterations, output in %s", max_iterations, output_dir)
    return 0


def _load_config(path: str, args=None) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        if args:
            _err(f"Config file not found: {path}", args)
        raise


def _resolve_execution_devices(device: str, device_id=None, gpu_ids=None):
    """
    Resolve execution device and GPU ids.
    Defaults:
      - cpu: no GPU ids
      - auto/cuda with CUDA available and no explicit ids -> all detected GPUs
    """
    if device == "cpu":
        return "cpu", []
    cuda_ok, n_gpu = _get_cuda_info()

    if device == "auto":
        if not cuda_ok or n_gpu == 0:
            return "cpu", []
        if device_id is not None:
            return "cuda", [int(device_id)]
        if gpu_ids:
            return "cuda", [int(i) for i in gpu_ids]
        return "cuda", list(range(n_gpu))

    # device == "cuda"
    if not cuda_ok or n_gpu == 0:
        logger.warning("CUDA requested but unavailable, fallback to CPU")
        return "cpu", []
    if device_id is not None:
        return "cuda", [int(device_id)]
    if gpu_ids:
        return "cuda", [int(i) for i in gpu_ids]
    return "cuda", list(range(n_gpu))


def _get_cuda_info():
    """Return (cuda_available, gpu_count)."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        n_gpu = int(torch.cuda.device_count()) if cuda_ok else 0
    except Exception:
        cuda_ok = False
        n_gpu = 0
    return cuda_ok, n_gpu


def _err(msg: str, args, code: int = 1, config=None):
    write_error(msg, args=args, config=config)
    if getattr(args, "json_errors", False):
        import json
        print(json.dumps({"error": msg, "code": code}), file=sys.stderr)
    else:
        print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)

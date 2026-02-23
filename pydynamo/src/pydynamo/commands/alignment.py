"""pydynamo alignment — align subtomograms against reference(s)."""
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
from ..io import read_dynamo_tbl
from ..runtime import configure_logging, progress_iter, write_error

logger = logging.getLogger(__name__)


def run(config_path: str, rest: list, args) -> int:
    """Run alignment command. Returns exit code."""
    config = _load_config(config_path, args)
    configure_logging(args, config, __name__)

    particles = config.get("particles")
    subtomograms = config.get("subtomograms")
    reference = config.get("reference")
    output_table = config.get("output_table")
    output_star = config.get("output_star") or output_table
    vll_path = config.get("vll") or config.get("vll_path")
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

    if not all([particles, subtomograms, reference, output_table]):
        _err("Missing required: particles, subtomograms, reference, output_table", args, config=config)
        return 1

    # Load reference
    with mrcfile.open(reference, mode="r", permissive=True) as mrc:
        ref_vol = mrc.data.copy().astype(np.float32)

    # Load particle table
    if particles.endswith(".star"):
        tbl_df = starfile.read(particles, always_dict=False)
        if isinstance(tbl_df, dict):
            tbl_df = tbl_df.get("particles", list(tbl_df.values())[0])
        paths_col = "rlnImageName" if "rlnImageName" in tbl_df.columns else None
    else:
        tbl_df = read_dynamo_tbl(particles, vll_path=vll_path)
        paths_col = None

    if paths_col and paths_col in tbl_df.columns:
        paths_list = tbl_df[paths_col].tolist()
    else:
        sub_path = Path(subtomograms)
        paths_list = sorted(sub_path.glob("*.mrc")) + sorted(sub_path.glob("*.mrcs"))
        paths_list = [str(p) for p in paths_list]
        if len(paths_list) < len(tbl_df):
            paths_list = (paths_list * (len(tbl_df) // len(paths_list) + 1))[: len(tbl_df)]

    base_dir = Path(particles).parent if isinstance(particles, str) else Path(".")
    tasks = []
    for i, p_path in enumerate(paths_list):
        full_path = base_dir / p_path if not Path(p_path).is_absolute() else Path(p_path)
        try:
            with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                part = mrc.data.copy().astype(np.float32)
        except Exception as e:
            logger.warning("Skip %s: %s", full_path, e)
            continue
        if part.shape != ref_vol.shape:
            logger.warning("Shape mismatch %s vs ref", part.shape)
            continue
        tasks.append((i, str(p_path), part))

    resolved_device, gpu_ids = _resolve_execution_devices(device, device_id, gpu_ids)

    def _run_one(task_tuple):
        i, p_path_local, part_local = task_tuple
        local_device_id = None
        if resolved_device == "cuda" and gpu_ids:
            local_device_id = gpu_ids[i % len(gpu_ids)]
        tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
            part_local, ref_vol,
            cone_step=cone_step, cone_range=cone_range,
            inplane_step=inplane_step, inplane_range=inplane_range,
            shift_search=shift_search, lowpass_angstrom=lowpass, pixel_size=pixel_size,
            multigrid_levels=multigrid_levels, device=resolved_device, device_id=local_device_id,
        )
        row = tbl_df.iloc[i].to_dict() if i < len(tbl_df) else {}
        row.update({
            "tag": int(row.get("tag", i + 1)),
            "tdrot": tdrot, "tilt": tilt, "narot": narot,
            "dx": dx, "dy": dy, "dz": dz,
            "cc": cc, "cc2": cc,
            "aligned": 1, "averaged": 1,
            "ref": int(row.get("ref", 1)),
        })
        if paths_col:
            row["rlnImageName"] = str(p_path_local)
        return i, row

    rows_pairs = []
    if resolved_device == "cuda" and len(gpu_ids) > 1 and len(tasks) > 1:
        logger.info("Alignment multi-GPU scheduling on devices: %s", gpu_ids)
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
            futures = [ex.submit(_run_one, t) for t in tasks]
            for f in progress_iter(as_completed(futures), total=len(futures), desc="alignment"):
                try:
                    rows_pairs.append(f.result())
                except Exception as e:
                    logger.warning("Alignment task failed: %s", e)
    else:
        for t in progress_iter(tasks, total=len(tasks), desc="alignment"):
            rows_pairs.append(_run_one(t))

    rows_pairs.sort(key=lambda x: x[0])
    rows = [r for _, r in rows_pairs]

    out_df = pd.DataFrame(rows)
    out_path = Path(output_star)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if str(out_path).endswith(".star"):
        starfile.write(out_df, str(out_path))
    else:
        from ..io import create_dynamo_table, convert_euler
        coords = out_df[["x", "y", "z"]].values if "x" in out_df.columns else np.zeros((len(out_df), 3))
        ang_zxz = np.column_stack([out_df["tdrot"], out_df["tilt"], out_df["narot"]])
        ang_zyz = convert_euler(ang_zxz, src_convention="dynamo", dst_convention="relion", degrees=True)
        create_dynamo_table(
            coords,
            angles_zyz=ang_zyz,
            micrograph_names=out_df["rlnMicrographName"].tolist() if "rlnMicrographName" in out_df.columns else None,
            origins=out_df[["dx", "dy", "dz"]].values,
            output_file=str(out_path),
        )

    logger.info("Aligned %d particles, output: %s", len(rows), output_star)
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

"""pydynamo alignment — align subtomograms against reference(s)."""
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
import yaml

from ..core.align import align_one_particle
from ..core.average import apply_inverse_transform, apply_symmetry
from ..core.wedge import get_wedge_mask
from ..io import convert_euler, dynamo_df_to_relion, read_dynamo_tbl
from ..runtime import (
    configure_logging,
    load_realspace_mask,
    log_command_inputs,
    progress_iter,
    progress_timing_text,
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


def run(config_path: str, rest: list, args) -> int:
    """Run alignment command. Returns exit code."""
    config = _load_config(config_path, args)
    configure_logging(args, config, __name__, config_path=config_path)
    log_command_inputs(logger, "alignment", config=config, config_path=config_path, args=args, rest=rest)

    particles = config.get("particles")
    subtomograms = config.get("subtomograms")
    reference = config.get("reference")
    output_table = config.get("output_table")
    output_star_cfg = config.get("output_star")
    output_star = output_star_cfg
    output_average = config.get("output_average")
    average_symmetry = str(config.get("average_symmetry", "c1"))
    vll_path = config.get("vll") or config.get("vll_path")
    cone_step = float(config.get("cone_step", 15))
    tdrot_step = float(config.get("tdrot_step", cone_step))
    tdrot_range = tuple(config.get("tdrot_range", [0, 360]))
    cone_range = tuple(config.get("cone_range", [0, 180]))
    inplane_step = float(config.get("inplane_step", 15))
    inplane_range = tuple(config.get("inplane_range", [0, 360]))
    shift_search = int(config.get("shift_search", 3))
    shift_mode = str(config.get("shift_mode", "cube"))
    subpixel = bool(config.get("subpixel", True))
    cc_mode = str(config.get("cc_mode", "ncc"))
    cc_local_window = int(config.get("cc_local_window", 5))
    cc_local_eps = float(config.get("cc_local_eps", 1e-8))
    angle_sampling_mode = str(config.get("angle_sampling_mode", "dynamo"))
    lowpass = config.get("lowpass")
    pixel_size = float(config.get("pixel_size", 1.0))
    tomogram_size = config.get("tomogram_size")
    multigrid_levels = int(config.get("multigrid_levels", 1))
    device = str(config.get("device", "auto"))
    device_id = config.get("device_id")
    gpu_ids = config.get("gpu_ids")
    progress_log_every = max(1, int(config.get("progress_log_every", 10)))
    mask_path = config.get("nmask")
    mask_consistency_min_fraction = float(config.get("mask_consistency_min_fraction", 0.01))
    apply_wedge_scoring = bool(config.get("apply_wedge_scoring", False))
    fsampling_mode = str(config.get("fsampling_mode", "none"))
    wedge_apply_to = str(config.get("wedge_apply_to", "auto"))
    subpixel_method = str(config.get("subpixel_method", "auto"))
    wedge_ftype = int(config.get("wedge_ftype", 1))
    wedge_ymin = float(config.get("wedge_ymin", -48))
    wedge_ymax = float(config.get("wedge_ymax", 48))
    wedge_xmin = float(config.get("wedge_xmin", -60))
    wedge_xmax = float(config.get("wedge_xmax", 60))

    if not all([particles, subtomograms, reference, output_table]):
        _err("Missing required: particles, subtomograms, reference, output_table", args, config=config, config_path=config_path)
        return 1

    # Load reference
    with mrcfile.open(reference, mode="r", permissive=True) as mrc:
        ref_vol = mrc.data.copy().astype(np.float32)
    try:
        align_mask = load_realspace_mask(mask_path, config_path=config_path, expected_shape=ref_vol.shape)
    except Exception as e:
        _err(f"Failed to load nmask: {e}", args, config=config, config_path=config_path)
        return 1
    if align_mask is not None:
        frac = float(np.mean(align_mask))
        logger.info("Alignment mask coverage: %.4f", frac)
        if frac < mask_consistency_min_fraction:
            logger.warning(
                "Alignment mask coverage %.4f < threshold %.4f; score/reconstruction consistency may degrade",
                frac,
                mask_consistency_min_fraction,
            )
    wedge_mask = None
    if apply_wedge_scoring:
        wedge_mask = get_wedge_mask(
            ref_vol.shape,
            ftype=wedge_ftype,
            ymintilt=wedge_ymin,
            ymaxtilt=wedge_ymax,
            xmintilt=wedge_xmin,
            xmaxtilt=wedge_xmax,
        ).astype(np.float32)
        logger.info(
            "Alignment wedge-aware scoring enabled: ftype=%d y=[%.0f,%.0f] x=[%.0f,%.0f]",
            wedge_ftype,
            wedge_ymin,
            wedge_ymax,
            wedge_xmin,
            wedge_xmax,
        )
    logger.info(
        "Alignment shape context: ref_shape=%s multigrid_levels=%d wedge_enabled=%s wedge_shape=%s",
        tuple(ref_vol.shape),
        multigrid_levels,
        bool(apply_wedge_scoring or (fsampling_mode.lower() == "table")),
        None if wedge_mask is None else tuple(wedge_mask.shape),
    )

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
        full_path = _resolve_particle_path(p_path, base_dir, subtomograms)
        if not full_path.exists():
            logger.warning("Skip missing particle file: %s", full_path)
            continue
        tasks.append((i, str(p_path), str(full_path)))

    resolved_device, gpu_ids = _resolve_execution_devices(device, device_id, gpu_ids)

    def _run_one(task_tuple):
        i, p_path_local, full_path_local = task_tuple
        try:
            with mrcfile.open(str(full_path_local), mode="r", permissive=True) as mrc:
                part_local = mrc.data.copy().astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"failed loading particle {full_path_local}: {e}")
        if part_local.shape != ref_vol.shape:
            raise RuntimeError(f"shape mismatch {part_local.shape} vs ref {ref_vol.shape}")
        seed = (0.0, 0.0, 0.0)
        fsampling = None
        if i < len(tbl_df):
            row0 = tbl_df.iloc[i]
            seed = (
                float(row0.get("tdrot", 0.0)),
                float(row0.get("tilt", 0.0)),
                float(row0.get("narot", 0.0)),
            )
            fsampling = {
                "ftype": row0.get("ftype", wedge_ftype),
                "ymintilt": row0.get("ymintilt", wedge_ymin),
                "ymaxtilt": row0.get("ymaxtilt", wedge_ymax),
                "xmintilt": row0.get("xmintilt", wedge_xmin),
                "xmaxtilt": row0.get("xmaxtilt", wedge_xmax),
                "fs1": row0.get("fs1", np.nan),
                "fs2": row0.get("fs2", np.nan),
            }
        local_device_id = None
        if resolved_device == "cuda" and gpu_ids:
            local_device_id = gpu_ids[i % len(gpu_ids)]
        tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
            part_local, ref_vol,
            mask=align_mask,
            cone_step=cone_step, tdrot_step=tdrot_step, tdrot_range=tdrot_range, cone_range=cone_range,
            inplane_step=inplane_step, inplane_range=inplane_range,
            shift_search=shift_search, lowpass_angstrom=lowpass, pixel_size=pixel_size,
            multigrid_levels=multigrid_levels, shift_mode=shift_mode, subpixel=subpixel,
            cc_mode=cc_mode,
            cc_local_window=cc_local_window,
            cc_local_eps=cc_local_eps,
            angle_sampling_mode=angle_sampling_mode,
            old_angles=seed,
            wedge_mask=wedge_mask,
            wedge_apply_to=wedge_apply_to,
            fsampling=fsampling,
            fsampling_mode=fsampling_mode,
            subpixel_method=subpixel_method,
            device=resolved_device, device_id=local_device_id,
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
        row["rlnImageName"] = str(p_path_local)
        tr = apply_inverse_transform(
            part_local,
            float(tdrot),
            float(tilt),
            float(narot),
            float(dx),
            float(dy),
            float(dz),
        )
        if align_mask is not None:
            tr = tr * align_mask
        return i, row, tr

    table_path = Path(output_table)
    table_suffix = table_path.suffix.lower()
    write_tbl_only_stream = (table_suffix == ".tbl" and not output_star_cfg)

    rows_pairs = []
    processed = 0
    failed = 0
    avg_acc = np.zeros_like(ref_vol, dtype=np.float64)
    avg_used = 0
    tbl_stream_fh = None
    tomo_name_to_id = {}
    if write_tbl_only_stream:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        tbl_stream_fh = open(table_path, "w", encoding="utf-8")
    progress_start = time.time()
    if resolved_device == "cuda" and len(gpu_ids) > 1 and len(tasks) > 1:
        logger.info("Alignment multi-GPU scheduling on devices: %s", gpu_ids)
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
            futures = {ex.submit(_run_one, t): t for t in tasks}
            for f in progress_iter(as_completed(futures), total=len(futures), desc="alignment"):
                processed += 1
                try:
                    i, row, tr = f.result()
                    if write_tbl_only_stream:
                        tbl_stream_fh.write(_format_tbl_row(_row_to_tbl_vector(row, tomo_name_to_id)))
                    else:
                        rows_pairs.append((i, row))
                    avg_acc += tr
                    avg_used += 1
                except Exception as e:
                    t = futures.get(f, ("?", "?", "?"))
                    logger.warning(
                        "Alignment task failed: %s (particle=%s full_path=%s ref_shape=%s multigrid_levels=%d wedge_enabled=%s)",
                        e,
                        t[1],
                        t[2],
                        tuple(ref_vol.shape),
                        multigrid_levels,
                        bool(apply_wedge_scoring or (fsampling_mode.lower() == "table")),
                    )
                    failed += 1
                if processed % progress_log_every == 0 or processed == len(tasks):
                    success_cnt = avg_used if write_tbl_only_stream else len(rows_pairs)
                    logger.info(
                        "Alignment progress %d/%d (success=%d failed=%d, %s)",
                        processed, len(tasks), success_cnt, failed,
                        progress_timing_text(progress_start, processed, len(tasks)),
                    )
    else:
        for t in progress_iter(tasks, total=len(tasks), desc="alignment"):
            processed += 1
            try:
                i, row, tr = _run_one(t)
                if write_tbl_only_stream:
                    tbl_stream_fh.write(_format_tbl_row(_row_to_tbl_vector(row, tomo_name_to_id)))
                else:
                    rows_pairs.append((i, row))
                avg_acc += tr
                avg_used += 1
            except Exception as e:
                logger.warning(
                    "Alignment task failed: %s (particle=%s full_path=%s ref_shape=%s multigrid_levels=%d wedge_enabled=%s)",
                    e,
                    t[1],
                    t[2],
                    tuple(ref_vol.shape),
                    multigrid_levels,
                    bool(apply_wedge_scoring or (fsampling_mode.lower() == "table")),
                )
                failed += 1
            if processed % progress_log_every == 0 or processed == len(tasks):
                success_cnt = avg_used if write_tbl_only_stream else len(rows_pairs)
                logger.info(
                    "Alignment progress %d/%d (success=%d failed=%d, %s)",
                    processed, len(tasks), success_cnt, failed,
                    progress_timing_text(progress_start, processed, len(tasks)),
                )

    if tbl_stream_fh is not None:
        tbl_stream_fh.close()

    rows = []
    out_df = pd.DataFrame()
    source_is_tbl = not particles.endswith(".star")
    if not write_tbl_only_stream:
        rows_pairs.sort(key=lambda x: x[0])
        rows = [r for _, r in rows_pairs]
        out_df = pd.DataFrame(rows)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        # Always honor output_table first: this is the primary refined-parameter output.
        if table_suffix == ".tbl":
            _write_refined_dynamo_tbl(out_df, table_path)
        elif table_suffix == ".star":
            out_star_df = _build_output_star_df(
                out_df,
                source_is_tbl=source_is_tbl,
                pixel_size=pixel_size,
                tomogram_size=tomogram_size,
            )
            starfile.write(out_star_df, str(table_path))
        else:
            _err(
                f"Unsupported output_table extension: {table_path.suffix}. Use .tbl or .star",
                args,
                config=config,
                config_path=config_path,
            )
            return 1

    # Optional secondary star output (RELION schema), useful when output_table is .tbl.
    if output_star_cfg:
        star_path = Path(output_star)
        same_as_table = star_path.resolve() == table_path.resolve()
        if not same_as_table:
            star_path.parent.mkdir(parents=True, exist_ok=True)
            out_star_df = _build_output_star_df(
                out_df,
                source_is_tbl=source_is_tbl,
                pixel_size=pixel_size,
                tomogram_size=tomogram_size,
            )
            starfile.write(out_star_df, str(star_path))

    # Build aligned average directly from alignment results.
    if avg_used > 0:
        avg = (avg_acc / avg_used).astype(np.float32)
        avg = apply_symmetry(avg, average_symmetry)
        avg_path = Path(output_average) if output_average else (table_path.parent / "average.mrc")
        avg_path.parent.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(str(avg_path), overwrite=True) as mrc:
            mrc.set_data(avg)
            try:
                mrc.voxel_size = float(pixel_size)
            except Exception:
                logger.warning("Failed to set average voxel_size from pixel_size=%s", pixel_size)
        logger.info("Reconstructed average %s from %d aligned particles", avg_path, avg_used)
    else:
        logger.warning("Skip average reconstruction: no aligned rows")

    aligned_n = avg_used if write_tbl_only_stream else len(rows)
    if output_star_cfg:
        logger.info("Aligned %d particles, output_table: %s, output_star: %s", aligned_n, output_table, output_star)
    else:
        logger.info("Aligned %d particles, output: %s", aligned_n, output_table)
    return 0


def _load_config(path: str, args=None) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        if args:
            _err(f"Config file not found: {path}", args, config_path=path)
        raise


def _build_output_star_df(
    df: pd.DataFrame,
    source_is_tbl: bool,
    pixel_size: float,
    tomogram_size=None,
) -> pd.DataFrame:
    """
    Build RELION-style STAR output from internal alignment rows.
    This intentionally avoids writing Dynamo tbl columns directly into STAR.
    """
    if source_is_tbl:
        # tbl source -> explicit Dynamo->RELION conversion.
        output_centered = tomogram_size is not None
        relion_df = dynamo_df_to_relion(
            df,
            pixel_size=pixel_size,
            tomogram_size=tomogram_size,
            output_centered=output_centered,
        )
        if "rlnImageName" in df.columns:
            relion_df["rlnImageName"] = df["rlnImageName"].astype(str).values
        if "tag" in df.columns:
            relion_df["rlnTomoParticleId"] = df["tag"].astype(int).values
        return relion_df

    # STAR source -> preserve RELION coordinates/micrograph fields,
    # then overwrite pose/origin with aligned results.
    out = df.copy()
    if all(c in out.columns for c in ["tdrot", "tilt", "narot"]):
        ang_zxz = np.column_stack([out["tdrot"], out["tilt"], out["narot"]])
        ang_zyz = convert_euler(ang_zxz, src_convention="dynamo", dst_convention="relion", degrees=True)
        ang_zyz = np.atleast_2d(ang_zyz)
        out["rlnAngleRot"] = ang_zyz[:, 0]
        out["rlnAngleTilt"] = ang_zyz[:, 1]
        out["rlnAnglePsi"] = ang_zyz[:, 2]
    if all(c in out.columns for c in ["dx", "dy", "dz"]):
        out["rlnOriginXAngst"] = -out["dx"].astype(float) * float(pixel_size)
        out["rlnOriginYAngst"] = -out["dy"].astype(float) * float(pixel_size)
        out["rlnOriginZAngst"] = -out["dz"].astype(float) * float(pixel_size)
    if "tag" in out.columns and "rlnTomoParticleId" not in out.columns:
        out["rlnTomoParticleId"] = out["tag"].astype(int)

    preferred_cols = [
        "rlnImageName",
        "rlnMicrographName",
        "rlnTomoName",
        "rlnTomoParticleId",
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
    existing = [c for c in preferred_cols if c in out.columns]
    if existing:
        return out[existing].copy()
    return out.copy()


def _row_to_tbl_vector(row: dict, tomo_name_to_id: dict) -> np.ndarray:
    """Convert one refined row dict to a 35-col Dynamo tbl vector."""
    v = np.zeros(35, dtype=float)
    tag = float(row.get("tag", 0) or 0)
    v[0] = tag if tag > 0 else 1.0
    v[1] = float(row.get("aligned", 1) or 1)
    v[2] = float(row.get("averaged", 1) or 1)
    v[3] = float(row.get("dx", 0.0) or 0.0)
    v[4] = float(row.get("dy", 0.0) or 0.0)
    v[5] = float(row.get("dz", 0.0) or 0.0)
    v[6] = float(row.get("tdrot", 0.0) or 0.0)
    v[7] = float(row.get("tilt", 0.0) or 0.0)
    v[8] = float(row.get("narot", 0.0) or 0.0)
    v[9] = float(row.get("cc", 0.0) or 0.0)
    v[10] = float(row.get("cc2", row.get("cc", 0.0)) or 0.0)
    micro = row.get("rlnMicrographName")
    if micro is not None:
        micro = str(micro)
        if micro not in tomo_name_to_id:
            tomo_name_to_id[micro] = float(len(tomo_name_to_id) + 1)
        v[19] = tomo_name_to_id[micro]
    else:
        v[19] = float(row.get("tomo", 1) or 1)
    v[23] = float(row.get("x", 0.0) or 0.0)
    v[24] = float(row.get("y", 0.0) or 0.0)
    v[25] = float(row.get("z", 0.0) or 0.0)
    v[33] = float(row.get("ref", 1) or 1)
    if "sref" in row:
        v[34] = float(row.get("sref", 0) or 0)
    return v


def _format_tbl_row(v: np.ndarray) -> str:
    """Format one 35-col Dynamo row as text line."""
    int_cols_0based = {0, 1, 2, 12, 19, 20, 21, 22, 30, 31, 33, 34}
    parts = [
        str(int(round(v[j]))) if j in int_cols_0based else format(float(v[j]), ".6g")
        for j in range(len(v))
    ]
    return " ".join(parts) + "\n"


def _write_refined_dynamo_tbl(df: pd.DataFrame, output_tbl: Path) -> None:
    """
    Write refined alignment results as Dynamo-style tbl while preserving
    refined shifts/eulers/cc/ref values.
    """
    n = len(df)
    T = np.zeros((n, 35), dtype=float)
    if n == 0:
        output_tbl.parent.mkdir(parents=True, exist_ok=True)
        output_tbl.write_text("", encoding="utf-8")
        return

    # Stable tomo id mapping when micrograph names are available.
    tomo_ids = np.ones(n, dtype=float)
    if "rlnMicrographName" in df.columns:
        names = df["rlnMicrographName"].fillna("1").astype(str).tolist()
        name_to_id = {}
        next_id = 1
        mapped = []
        for name in names:
            if name not in name_to_id:
                name_to_id[name] = next_id
                next_id += 1
            mapped.append(name_to_id[name])
        tomo_ids = np.asarray(mapped, dtype=float)
    elif "tomo" in df.columns:
        tomo_ids = pd.to_numeric(df["tomo"], errors="coerce").fillna(1).to_numpy(dtype=float)

    def _num_series(key: str, default):
        if key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce")
        else:
            s = pd.Series([default] * n, index=df.index, dtype=float)
        return s.fillna(default).to_numpy(dtype=float)

    T[:, 0] = _num_series("tag", 0.0)
    if np.all(T[:, 0] == 0):
        T[:, 0] = np.arange(1, n + 1, dtype=float)
    T[:, 1] = _num_series("aligned", 1.0)
    T[:, 2] = _num_series("averaged", 1.0)
    T[:, 3] = _num_series("dx", 0.0)
    T[:, 4] = _num_series("dy", 0.0)
    T[:, 5] = _num_series("dz", 0.0)
    T[:, 6] = _num_series("tdrot", 0.0)
    T[:, 7] = _num_series("tilt", 0.0)
    T[:, 8] = _num_series("narot", 0.0)
    T[:, 9] = _num_series("cc", 0.0)
    T[:, 10] = _num_series("cc2", 0.0 if "cc" not in df.columns else np.nan)
    if "cc2" not in df.columns and "cc" in df.columns:
        T[:, 10] = T[:, 9]
    T[:, 19] = tomo_ids
    T[:, 23] = _num_series("x", 0.0)
    T[:, 24] = _num_series("y", 0.0)
    T[:, 25] = _num_series("z", 0.0)
    T[:, 33] = _num_series("ref", 1.0)
    if "sref" in df.columns:
        T[:, 34] = _num_series("sref", 0.0)

    int_cols_0based = {0, 1, 2, 12, 19, 20, 21, 22, 30, 31, 33, 34}
    output_tbl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tbl, "w", encoding="utf-8") as fh:
        for row in T:
            parts = [
                str(int(round(row[j]))) if j in int_cols_0based else format(float(row[j]), ".6g")
                for j in range(len(row))
            ]
            fh.write(" ".join(parts) + "\n")


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


def _err(msg: str, args, code: int = 1, config=None, config_path=None):
    write_error(msg, args=args, config=config, config_path=config_path)
    if getattr(args, "json_errors", False):
        import json
        print(json.dumps({"error": msg, "code": code}), file=sys.stderr)
    else:
        print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)

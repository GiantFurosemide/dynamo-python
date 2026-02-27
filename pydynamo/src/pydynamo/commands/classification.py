"""pydynamo classification — MRA: multireference alignment and classification."""
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import mrcfile
import numpy as np
import starfile
import yaml

from ..core.align import align_one_particle
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
    resolve_path,
    write_error,
)

logger = logging.getLogger(__name__)

# Process-local cache for classification CPU workers (ref_vols, align_mask, wedge_mask)
_classification_worker_cache = {}


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


def _classification_cpu_worker(payload: dict):
    """
    Run one classification alignment task in a worker process (CPU path).
    payload: config_path, config, ref_paths, pidx, p_path, full_path, seed_row, swap, ref_to_align.
    Returns (best_ref, best_row, tr). On load failure or shape mismatch returns (0, None, None).
    tr is the transformed volume for accumulation, or None.
    """
    global _classification_worker_cache
    from ..runtime import load_realspace_mask, resolve_path

    config_path = payload["config_path"]
    config = payload["config"]
    ref_paths = payload["ref_paths"]
    pidx = payload["pidx"]
    p_path = payload["p_path"]
    full_path = payload["full_path"]
    seed_row = payload["seed_row"]
    swap = payload["swap"]
    ref_to_align = payload.get("ref_to_align")

    cache_key = (config_path, tuple(ref_paths))
    if cache_key not in _classification_worker_cache:
        ref_vols = []
        for rp in ref_paths:
            with mrcfile.open(rp, mode="r", permissive=True) as mrc:
                ref_vols.append(mrc.data.copy().astype(np.float32))
        align_mask = load_realspace_mask(
            config.get("nmask"), config_path=config_path, expected_shape=ref_vols[0].shape
        )
        wedge_mask = None
        if config.get("apply_wedge_scoring", False):
            wedge_mask = get_wedge_mask(
                ref_vols[0].shape,
                ftype=int(config.get("wedge_ftype", 1)),
                ymintilt=float(config.get("wedge_ymin", -48)),
                ymaxtilt=float(config.get("wedge_ymax", 48)),
                xmintilt=float(config.get("wedge_xmin", -60)),
                xmaxtilt=float(config.get("wedge_xmax", 60)),
            ).astype(np.float32)
        _classification_worker_cache[cache_key] = (ref_vols, align_mask, wedge_mask)
    ref_vols, align_mask, wedge_mask = _classification_worker_cache[cache_key]

    try:
        with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
            part = np.asarray(mrc.data, dtype=np.float32)
            if not part.flags.writeable:
                part = part.copy()
    except Exception:
        return 0, None, None
    if part.shape != ref_vols[0].shape:
        return 0, None, None

    nref = len(ref_vols)
    refs_to_align = range(nref) if swap else [max(0, min(nref - 1, ref_to_align or 0))]
    best_cc = -2
    best_ref = 0
    best_row = None

    wedge_ftype = int(config.get("wedge_ftype", 1))
    seed = (
        float(seed_row.get("tdrot", 0.0)),
        float(seed_row.get("tilt", 0.0)),
        float(seed_row.get("narot", 0.0)),
    )
    fsampling = None
    if seed_row:
        fsampling = {
            "ftype": seed_row.get("ftype", wedge_ftype),
            "ymintilt": seed_row.get("ymintilt", float(config.get("wedge_ymin", -48))),
            "ymaxtilt": seed_row.get("ymaxtilt", float(config.get("wedge_ymax", 48))),
            "xmintilt": seed_row.get("xmintilt", float(config.get("wedge_xmin", -60))),
            "xmaxtilt": seed_row.get("xmaxtilt", float(config.get("wedge_xmax", 60))),
            "fs1": seed_row.get("fs1", np.nan),
            "fs2": seed_row.get("fs2", np.nan),
        }

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
    multigrid_levels = int(config.get("multigrid_levels", 1))
    wedge_apply_to = str(config.get("wedge_apply_to", "auto"))
    fsampling_mode = str(config.get("fsampling_mode", "none"))
    subpixel_method = str(config.get("subpixel_method", "auto"))

    for r in refs_to_align:
        tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
            part,
            ref_vols[r],
            mask=align_mask,
            cone_step=cone_step,
            tdrot_step=tdrot_step,
            tdrot_range=tdrot_range,
            cone_range=cone_range,
            inplane_step=inplane_step,
            inplane_range=inplane_range,
            shift_search=shift_search,
            lowpass_angstrom=lowpass,
            pixel_size=pixel_size,
            multigrid_levels=multigrid_levels,
            shift_mode=shift_mode,
            subpixel=subpixel,
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
            device="cpu",
            device_id=None,
        )
        row = dict(seed_row)
        row.update({
            "tag": pidx + 1,
            "tdrot": tdrot,
            "tilt": tilt,
            "narot": narot,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "cc": cc,
            "cc2": cc,
            "ref": r + 1,
            "aligned": 1,
            "averaged": 1,
            "x": row.get("x", 0),
            "y": row.get("y", 0),
            "z": row.get("z", 0),
            "rlnImageName": str(full_path),
            "particle_idx": pidx,
        })
        if cc > best_cc:
            best_cc = cc
            best_ref = r
            best_row = row
    if best_row is None:
        return 0, None, None
    tr = apply_inverse_transform(
        part,
        float(best_row["tdrot"]),
        float(best_row["tilt"]),
        float(best_row["narot"]),
        float(best_row["dx"]),
        float(best_row["dy"]),
        float(best_row["dz"]),
    )
    if align_mask is not None:
        tr = tr * align_mask
    return best_ref, best_row, tr


def run(config_path: str, rest: list, args) -> int:
    """Run classification (MRA) command. Returns exit code."""
    config = _load_config(config_path, args)
    configure_logging(args, config, __name__, config_path=config_path)
    log_command_inputs(logger, "classification", config=config, config_path=config_path, args=args, rest=rest)

    particles = config.get("particles")
    subtomograms = config.get("subtomograms")
    references = config.get("references")
    tables = config.get("tables")
    output_dir = Path(config.get("output_dir", "mra_output"))
    output_average = config.get("output_average")
    output_average_dir = config.get("output_average_dir")
    average_symmetry = str(config.get("average_symmetry", "c1"))
    max_iterations = int(config.get("max_iterations", 5))
    swap = config.get("swap", True)
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
    multigrid_levels = int(config.get("multigrid_levels", 1))
    device = str(config.get("device", "auto"))
    device_id = config.get("device_id")
    gpu_ids = config.get("gpu_ids")
    mask_path = config.get("nmask")
    progress_log_every = max(1, int(config.get("progress_log_every", 10)))
    num_workers = resolve_cpu_workers(config.get("num_workers"), default=1)
    resume = bool(config.get("resume", False))
    resume_from_iteration = config.get("resume_from_iteration")
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

    if not all([particles, subtomograms, references, output_dir]):
        _err("Missing required: particles, subtomograms, references, output_dir", args, config=config, config_path=config_path)
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
    try:
        align_mask = load_realspace_mask(mask_path, config_path=config_path, expected_shape=ref_vols[0].shape)
    except Exception as e:
        _err(f"Failed to load nmask: {e}", args, config=config, config_path=config_path)
        return 1
    if align_mask is not None:
        frac = float(np.mean(align_mask))
        logger.info("Classification mask coverage: %.4f", frac)
        if frac < mask_consistency_min_fraction:
            logger.warning(
                "Classification mask coverage %.4f < threshold %.4f; pose/average consistency may degrade",
                frac,
                mask_consistency_min_fraction,
            )
    wedge_mask = None
    if apply_wedge_scoring:
        wedge_mask = get_wedge_mask(
            ref_vols[0].shape,
            ftype=wedge_ftype,
            ymintilt=wedge_ymin,
            ymaxtilt=wedge_ymax,
            xmintilt=wedge_xmin,
            xmaxtilt=wedge_xmax,
        ).astype(np.float32)
        logger.info(
            "Classification wedge-aware scoring enabled: ftype=%d y=[%.0f,%.0f] x=[%.0f,%.0f]",
            wedge_ftype,
            wedge_ymin,
            wedge_ymax,
            wedge_xmin,
            wedge_xmax,
        )

    # Initial per-ref tables (swap: each particle in all refs)
    if tables and isinstance(tables, list) and len(tables) == nref:
        ref_tables = [read_dynamo_tbl(t) for t in tables]
    else:
        ref_tables = [tbl_df.copy() for _ in range(nref)]

    resolved_device, gpu_ids = _resolve_execution_devices(device, device_id, gpu_ids)
    start_iteration = _resolve_start_iteration(
        output_dir=output_dir,
        max_iterations=max_iterations,
        resume=resume,
        resume_from_iteration=resume_from_iteration,
    )
    if start_iteration > 0:
        _load_reference_state_from_iteration(output_dir, start_iteration, ref_vols)
        logger.info(
            "Classification resume enabled: continuing from iteration %d/%d",
            start_iteration + 1,
            max_iterations,
        )

    def _align_particle_task(pidx, p_path):
        full_path = _resolve_particle_path(p_path, base_dir, subtomograms)
        try:
            with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                part = np.asarray(mrc.data, dtype=np.float32)
                if not part.flags.writeable:
                    part = part.copy()
        except Exception:
            return None, None, None
        if part.shape != ref_vols[0].shape:
            return None, None, None

        best_cc = -2
        best_ref = 0
        best_row = None
        seed = (0.0, 0.0, 0.0)
        fsampling = None
        if pidx < len(tbl_df):
            row0 = tbl_df.iloc[pidx]
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
        refs_to_align = range(nref) if swap else [max(0, min(nref - 1, int(ref_tables[0].iloc[pidx].get("ref", 1)) - 1))]
        local_device_id = None
        if resolved_device == "cuda" and gpu_ids:
            local_device_id = gpu_ids[pidx % len(gpu_ids)]

        for r in refs_to_align:
            tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
                part, ref_vols[r],
                mask=align_mask,
                cone_step=cone_step, tdrot_step=tdrot_step, tdrot_range=tdrot_range, cone_range=cone_range,
                inplane_step=inplane_step, inplane_range=inplane_range,
                shift_search=shift_search,
                lowpass_angstrom=lowpass, pixel_size=pixel_size,
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
        if best_row is None:
            return 0, None, None
        tr = apply_inverse_transform(
            part,
            float(best_row["tdrot"]),
            float(best_row["tilt"]),
            float(best_row["narot"]),
            float(best_row["dx"]),
            float(best_row["dy"]),
            float(best_row["dz"]),
        )
        if align_mask is not None:
            tr = tr * align_mask
        return best_ref, best_row, tr

    for ite in range(start_iteration, max_iterations):
        logger.info("MRA iteration %d/%d", ite + 1, max_iterations)
        ite_dir = output_dir / f"ite_{ite+1:03d}"
        ite_dir.mkdir(parents=True, exist_ok=True)
        row_paths = [ite_dir / f"rows_ref_{r+1:03d}.jsonl" for r in range(nref)]
        row_fhs = [open(p, "w", encoding="utf-8") for p in row_paths]
        per_ref_counts = [0 for _ in range(nref)]
        per_ref_acc = [np.zeros((sidelength, sidelength, sidelength), dtype=np.float64) for _ in range(nref)]
        per_ref_used = [0 for _ in range(nref)]
        ite_total = len(paths)
        ite_processed = 0
        ite_success = 0
        ite_failed = 0
        ite_start = time.time()
        ref_paths_resolved = [resolve_path(r, config_path) for r in refs]
        if resolved_device == "cuda" and len(gpu_ids) > 1 and len(paths) > 1:
            logger.info("Classification multi-GPU scheduling on devices: %s", gpu_ids)
            with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
                futures = [ex.submit(_align_particle_task, pidx, p_path) for pidx, p_path in enumerate(paths)]
                for f in progress_iter(as_completed(futures), total=len(futures), desc=f"classify ite{ite+1}"):
                    ite_processed += 1
                    try:
                        best_ref, best_row, tr = f.result()
                        if best_row is not None:
                            best_row["ref"] = best_ref + 1
                            best_row["grep_average"] = 1
                            row_fhs[best_ref].write(json.dumps(best_row, ensure_ascii=True) + "\n")
                            per_ref_counts[best_ref] += 1
                            if tr is not None:
                                per_ref_acc[best_ref] += tr
                                per_ref_used[best_ref] += 1
                            ite_success += 1
                        else:
                            ite_failed += 1
                    except Exception as e:
                        logger.warning("Classification task failed: %s", e)
                        ite_failed += 1
                    if ite_processed % progress_log_every == 0 or ite_processed == ite_total:
                        logger.info(
                            "Classification iteration %d progress %d/%d (success=%d failed=%d, %s)",
                            ite + 1,
                            ite_processed,
                            ite_total,
                            ite_success,
                            ite_failed,
                            progress_timing_text(ite_start, ite_processed, ite_total),
                        )
        elif resolved_device == "cpu" and num_workers > 1 and len(paths) > 1:
            logger.info("Classification CPU multi-process with num_workers=%d", num_workers)
            payloads = []
            for pidx, p_path in enumerate(paths):
                full_path = _resolve_particle_path(p_path, base_dir, subtomograms)
                seed_row = tbl_df.iloc[pidx].to_dict() if pidx < len(tbl_df) else {}
                ref_to_align = None
                if not swap and ref_tables and len(ref_tables) > 0 and pidx < len(ref_tables[0]):
                    ref_to_align = max(0, min(nref - 1, int(ref_tables[0].iloc[pidx].get("ref", 1)) - 1))
                payloads.append({
                    "config_path": config_path,
                    "config": config,
                    "ref_paths": ref_paths_resolved,
                    "pidx": pidx,
                    "p_path": p_path,
                    "full_path": str(full_path),
                    "seed_row": seed_row,
                    "swap": swap,
                    "ref_to_align": ref_to_align,
                })
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(_classification_cpu_worker, p) for p in payloads]
                for f in progress_iter(as_completed(futures), total=len(futures), desc=f"classify ite{ite+1}"):
                    ite_processed += 1
                    try:
                        best_ref, best_row, tr = f.result()
                        if best_row is not None:
                            best_row["ref"] = best_ref + 1
                            best_row["grep_average"] = 1
                            row_fhs[best_ref].write(json.dumps(best_row, ensure_ascii=True) + "\n")
                            per_ref_counts[best_ref] += 1
                            if tr is not None:
                                per_ref_acc[best_ref] += tr
                                per_ref_used[best_ref] += 1
                            ite_success += 1
                        else:
                            ite_failed += 1
                    except Exception as e:
                        logger.warning("Classification task failed: %s", e)
                        ite_failed += 1
                    if ite_processed % progress_log_every == 0 or ite_processed == ite_total:
                        logger.info(
                            "Classification iteration %d progress %d/%d (success=%d failed=%d, %s)",
                            ite + 1,
                            ite_processed,
                            ite_total,
                            ite_success,
                            ite_failed,
                            progress_timing_text(ite_start, ite_processed, ite_total),
                        )
        else:
            for pidx, p_path in progress_iter(list(enumerate(paths)), total=len(paths), desc=f"classify ite{ite+1}"):
                ite_processed += 1
                try:
                    best_ref, best_row, tr = _align_particle_task(pidx, p_path)
                    if best_row is not None:
                        best_row["ref"] = best_ref + 1
                        best_row["grep_average"] = 1
                        row_fhs[best_ref].write(json.dumps(best_row, ensure_ascii=True) + "\n")
                        per_ref_counts[best_ref] += 1
                        if tr is not None:
                            per_ref_acc[best_ref] += tr
                            per_ref_used[best_ref] += 1
                        ite_success += 1
                    else:
                        ite_failed += 1
                except Exception as e:
                    logger.warning("Classification task failed: %s", e)
                    ite_failed += 1
                if ite_processed % progress_log_every == 0 or ite_processed == ite_total:
                    logger.info(
                        "Classification iteration %d progress %d/%d (success=%d failed=%d, %s)",
                        ite + 1,
                        ite_processed,
                        ite_total,
                        ite_success,
                        ite_failed,
                        progress_timing_text(ite_start, ite_processed, ite_total),
                    )

        for fh in row_fhs:
            fh.close()

        # Assemble + MRA: use per-ref accumulators from alignment (no re-read of particles).
        for r in range(nref):
            tbl_path = ite_dir / f"refined_table_ref_{r+1:03d}.tbl"
            avg_path = ite_dir / f"average_ref_{r+1:03d}.mrc"
            tomo_name_to_id = {}
            with open(tbl_path, "w", encoding="utf-8") as tbl_fh:
                with open(row_paths[r], "r", encoding="utf-8") as row_fh:
                    for line in row_fh:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        tbl_fh.write(_format_tbl_row(_row_to_tbl_vector(row, tomo_name_to_id)))
            used = per_ref_used[r]
            if used > 0:
                avg = (per_ref_acc[r] / used).astype(np.float32)
                with mrcfile.new(str(avg_path), overwrite=True) as mrc:
                    mrc.set_data(avg)
                ref_vols[r] = avg
            logger.info(
                "Classification ite %d ref %d: rows=%d used_for_avg=%d",
                ite + 1,
                r + 1,
                per_ref_counts[r],
                used,
            )

        _write_iteration_checkpoint(
            output_dir=output_dir,
            iteration=ite + 1,
            max_iterations=max_iterations,
            nref=nref,
            per_ref_counts=per_ref_counts,
            success=ite_success,
            failed=ite_failed,
        )

    # Save final averages for direct downstream usage.
    final_avg_dir = Path(output_average_dir) if output_average_dir else output_dir
    final_avg_dir.mkdir(parents=True, exist_ok=True)
    if nref == 1:
        final_avg = apply_symmetry(ref_vols[0], average_symmetry)
        final_avg_path = Path(output_average) if output_average else (final_avg_dir / "average.mrc")
        final_avg_path.parent.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(str(final_avg_path), overwrite=True) as mrc:
            mrc.set_data(final_avg.astype(np.float32))
            try:
                mrc.voxel_size = float(pixel_size)
            except Exception:
                logger.warning("Failed to set final average voxel_size from pixel_size=%s", pixel_size)
        logger.info("Saved final classification average: %s", final_avg_path)
    else:
        if output_average:
            logger.warning("output_average is ignored when references>1; writing per-reference averages")
        for r in range(nref):
            final_avg = apply_symmetry(ref_vols[r], average_symmetry)
            final_avg_path = final_avg_dir / f"average_ref_{r+1:03d}.mrc"
            final_avg_path.parent.mkdir(parents=True, exist_ok=True)
            with mrcfile.new(str(final_avg_path), overwrite=True) as mrc:
                mrc.set_data(final_avg.astype(np.float32))
                try:
                    mrc.voxel_size = float(pixel_size)
                except Exception:
                    logger.warning("Failed to set final average voxel_size from pixel_size=%s", pixel_size)
        logger.info("Saved final classification averages to: %s", final_avg_dir)

    logger.info("MRA complete: %d iterations, output in %s", max_iterations, output_dir)
    return 0


def _resolve_start_iteration(
    output_dir: Path,
    max_iterations: int,
    resume: bool,
    resume_from_iteration,
) -> int:
    """Return zero-based iteration index to start from."""
    if resume_from_iteration is not None:
        try:
            completed = int(resume_from_iteration)
        except Exception:
            completed = 0
        completed = max(0, min(completed, max_iterations))
        return completed
    if not resume:
        return 0
    completed = 0
    for ite in range(1, max_iterations + 1):
        ckpt_path = output_dir / f"ite_{ite:03d}" / "checkpoint.yaml"
        if not ckpt_path.exists():
            break
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                ckpt = yaml.safe_load(f) or {}
        except Exception:
            break
        if bool(ckpt.get("completed", False)):
            completed = ite
        else:
            break
    return completed


def _load_reference_state_from_iteration(output_dir: Path, iteration_completed: int, ref_vols: list) -> None:
    """Load per-reference averages from completed iteration if available."""
    if iteration_completed <= 0:
        return
    ite_dir = output_dir / f"ite_{iteration_completed:03d}"
    for r in range(len(ref_vols)):
        avg_path = ite_dir / f"average_ref_{r+1:03d}.mrc"
        if not avg_path.exists():
            continue
        try:
            with mrcfile.open(str(avg_path), mode="r", permissive=True) as mrc:
                ref_vols[r] = mrc.data.copy().astype(np.float32)
        except Exception as e:
            logger.warning("Failed loading resume average %s: %s", avg_path, e)


def _write_iteration_checkpoint(
    output_dir: Path,
    iteration: int,
    max_iterations: int,
    nref: int,
    per_ref_counts: list[int],
    success: int,
    failed: int,
) -> None:
    """Persist one iteration checkpoint for restart/resume."""
    ite_dir = output_dir / f"ite_{iteration:03d}"
    ite_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "iteration": int(iteration),
        "max_iterations": int(max_iterations),
        "nref": int(nref),
        "per_ref_counts": [int(v) for v in per_ref_counts],
        "success": int(success),
        "failed": int(failed),
        "completed": True,
    }
    with open(ite_dir / "checkpoint.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(ckpt, f, sort_keys=False)


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


def _load_config(path: str, args=None) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        if args:
            _err(f"Config file not found: {path}", args, config_path=path)
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


def _err(msg: str, args, code: int = 1, config=None, config_path=None):
    write_error(msg, args=args, config=config, config_path=config_path)
    if getattr(args, "json_errors", False):
        import json
        print(json.dumps({"error": msg, "code": code}), file=sys.stderr)
    else:
        print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)

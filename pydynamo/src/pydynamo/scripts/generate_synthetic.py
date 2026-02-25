#!/usr/bin/env python3
"""
Generate synthetic tomogram data for testing.
Ref: requirements_refined_004.md, requirements_refined_002.md §2 (TomoPANDA-pick)

- Internal: RELION star format (rlnCenteredCoordinate Å, rlnAngleRot/Tilt/Psi ZYZ). I/O converts to tbl.
- Sample orientations and coordinates FIRST, then generate tomogram
- Tomogram background: Gaussian noise. Missing wedge: RETAIN ±48° frequencies
- Output: tbl, vll, star; crop subtomos; classification set (real + noise)
"""
import logging
import os
from pathlib import Path
from typing import List, Tuple

import mrcfile
import numpy as np
import pandas as pd
import starfile
import yaml

from ..runtime import configure_logging, log_command_inputs, progress_iter, write_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_template(path: str) -> Tuple[np.ndarray, float]:
    """Load template MRC, return (data, pixel_size)."""
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data = mrc.data.copy()
        logger.info(
            "DEBUG template values: min=%.6g max=%.6g mean=%.6g std=%.6g",
            float(np.min(data)), float(np.max(data)), float(np.mean(data)), float(np.std(data)),
        )
        voxel = mrc.voxel_size
        if voxel is None or (hasattr(voxel, "x") and voxel.x is None):
            pixel_size = float(getattr(mrc, "voxel_size", 4.284) or 4.284)
            if hasattr(pixel_size, "__iter__") and not isinstance(pixel_size, str):
                pixel_size = float(pixel_size[0])
        else:
            pixel_size = float(voxel.x) if hasattr(voxel, "x") else float(voxel)
    return data, pixel_size


def euler_zyz_to_rotation_matrix(rot: float, tilt: float, psi: float, degrees: bool = True) -> np.ndarray:
    """RELION ZYZ Euler to 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_euler("ZYZ", [rot, tilt, psi], degrees=degrees)
    return r.as_matrix()


def rotate_volume(vol: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply rotation matrix R to volume."""
    from scipy.ndimage import map_coordinates

    center = np.array(vol.shape, dtype=float) / 2.0 - 0.5
    coords = np.mgrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]].astype(float)
    coords = coords - center.reshape(3, 1, 1, 1)
    flat = coords.reshape(3, -1)
    rotated = R @ flat
    rotated += center.reshape(3, 1)
    coords_new = rotated.reshape(3, vol.shape[0], vol.shape[1], vol.shape[2])
    return map_coordinates(vol, coords_new, order=1, mode="constant", cval=0)


def sample_particles_star(
    n_particles: int,
    tomogram_size: Tuple[int, int, int],
    sidelength: int,
    pixel_size: float,
    seed: int = 42,
) -> List[dict]:
    """
    Sample orientations and coordinates BEFORE generating tomogram.
    Internal: RELION star format — rlnCenteredCoordinate (Å), rlnAngleRot/Tilt/Psi (ZYZ).
    Returns list of dicts. tomogram_size = (nx, ny, nz); MRC shape = (nz, ny, nx).
    """
    np.random.seed(seed)
    nx, ny, nz = tomogram_size
    half = sidelength // 2
    center = np.array([nx, ny, nz], dtype=float) / 2.0
    x_min, x_max = half, nx - half - 1
    y_min, y_max = half, ny - half - 1
    z_min, z_max = half, nz - half - 1
    if x_min > x_max or y_min > y_max or z_min > z_max:
        raise ValueError(
            f"Tomogram {tomogram_size} too small for sidelength {sidelength}. "
            f"Need each dim >= {sidelength}"
        )
    rows = []
    for i in range(n_particles):
        rot = float(np.random.uniform(0, 360))
        tilt = float(np.random.uniform(0, 180))
        psi = float(np.random.uniform(0, 360))
        x_abs = float(np.random.uniform(x_min, x_max))
        y_abs = float(np.random.uniform(y_min, y_max))
        z_abs = float(np.random.uniform(z_min, z_max))
        centered_px = np.array([x_abs, y_abs, z_abs]) - center
        centered_angstrom = centered_px * pixel_size
        rows.append({
            "tag": i + 1,
            "rlnCenteredCoordinateXAngst": centered_angstrom[0],
            "rlnCenteredCoordinateYAngst": centered_angstrom[1],
            "rlnCenteredCoordinateZAngst": centered_angstrom[2],
            "rlnAngleRot": rot,
            "rlnAngleTilt": tilt,
            "rlnAnglePsi": psi,
            "rlnOriginXAngst": 0.0,
            "rlnOriginYAngst": 0.0,
            "rlnOriginZAngst": 0.0,
            "rlnMicrographName": "tomo1",
            "ref": 1,
        })
    return rows


def star_to_absolute_pixels(
    row: dict,
    pixel_size: float,
    tomogram_size: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Convert RELION star row to absolute (x, y, z) pixels for embed/crop."""
    center = np.array(tomogram_size, dtype=float) / 2.0
    centered_ang = np.array([
        row["rlnCenteredCoordinateXAngst"],
        row["rlnCenteredCoordinateYAngst"],
        row["rlnCenteredCoordinateZAngst"],
    ])
    centered_px = centered_ang / pixel_size
    absolute = centered_px + center
    return float(absolute[0]), float(absolute[1]), float(absolute[2])


def _embed_with_debug(tomogram, particle, cx, cy, cz, tomogram_size, idx, log_first_n=3):
    """Wrapper that logs first N embeds for debugging."""
    slice_sum = float(np.sum(particle)) if idx < log_first_n else 0.0
    embed_particle(tomogram, particle, cx, cy, cz, tomogram_size)
    if idx < log_first_n:
        logger.info("DEBUG embed #%d: particle_sum=%.4g at (%.1f,%.1f,%.1f)", idx, slice_sum, cx, cy, cz)


def embed_particle(
    tomogram: np.ndarray,
    particle: np.ndarray,
    cx: float,
    cy: float,
    cz: float,
    tomogram_size: Tuple[int, int, int],
) -> None:
    """
    Embed particle at center (cx, cy, cz). Physical (x,y,z) -> volume[z,y,x].
    tomogram shape: (nz, ny, nx), tomogram_size: (nx, ny, nz). Ref: requirements_refined_004 §3.7.
    """
    nx, ny, nz = tomogram_size
    s0, s1, s2 = particle.shape
    h0, h1, h2 = s0 // 2, s1 // 2, s2 // 2
    cz0, cy0, cx0 = int(np.round(cz)), int(np.round(cy)), int(np.round(cx))
    r0 = cz0 - h0
    r1 = cy0 - h1
    r2 = cx0 - h2
    i0 = max(0, -r0)
    i1 = min(s0, nz - r0)
    j0 = max(0, -r1)
    j1 = min(s1, ny - r1)
    k0 = max(0, -r2)
    k1 = min(s2, nx - r2)
    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        logger.warning("DEBUG embed SKIP: particle out of bounds at (%.1f,%.1f,%.1f)", cx, cy, cz)
        return
    slice_sum = float(np.sum(particle[i0:i1, j0:j1, k0:k1]))
    ti0, ti1 = r0 + i0, r0 + i1
    tj0, tj1 = r1 + j0, r1 + j1
    tk0, tk1 = r2 + k0, r2 + k1
    tomogram[ti0:ti1, tj0:tj1, tk0:tk1] += particle[i0:i1, j0:j1, k0:k1]


def run(config_path=None, cli_args=None):
    """Generate synthetic data. Called from CLI or __main__."""
    cfg = {}
    path = config_path or "config/synthetic_defaults.yaml"
    if path and os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    configure_logging(cli_args, cfg, __name__, config_path=config_path)
    log_command_inputs(logger, "gen_synthetic", config=cfg, config_path=config_path, args=cli_args, rest=[])

    template_path = str(cfg.get("template", "synthetic_data/emd_32820_bin4.mrc"))
    out_root = Path(cfg.get("output_root", "synthetic_data"))
    n_particles = int(cfg.get("n_particles", 1000))
    n_noise = int(cfg.get("n_noise", 1000))
    tomogram_size = tuple(cfg.get("tomogram_size", [2000, 2000, 800]))
    apply_missing_wedge = cfg.get("apply_missing_wedge", True)
    wedge_ftype = int(cfg.get("wedge_ftype", 1))
    wedge_ymin = float(cfg.get("wedge_ymin", -48))
    wedge_ymax = float(cfg.get("wedge_ymax", 48))
    wedge_xmin = float(cfg.get("wedge_xmin", -60))
    wedge_xmax = float(cfg.get("wedge_xmax", 60))
    wedge_params = (wedge_ftype, wedge_ymin, wedge_ymax, wedge_xmin, wedge_xmax) if apply_missing_wedge else None
    noise_sigma = float(cfg.get("noise_sigma", 1.0))
    particle_scale_ratio = float(cfg.get("particle_scale_ratio", 3.0))
    seed = int(cfg.get("seed", 42))

    template, pixel_size = load_template(template_path)
    L = template.shape[0]
    logger.info("Template %s: shape=%s, pixel_size=%s", template_path, template.shape, pixel_size)

    # ---------- 1. SAMPLE FIRST (internal: RELION star format) ----------
    sample_rows = sample_particles_star(n_particles, tomogram_size, L, pixel_size, seed=seed)
    logger.info("Sampled %d particles (internal: RELION star format)", n_particles)

    # ---------- 2. TOMOGRAM: Gaussian background + embed particles ----------
    nx, ny, nz = tomogram_size
    tomogram = np.random.randn(nz, ny, nx).astype(np.float32) * noise_sigma
    embed_count = 0
    try:
        for row in progress_iter(sample_rows, total=len(sample_rows), desc="gen embed"):
            x_abs, y_abs, z_abs = star_to_absolute_pixels(row, pixel_size, tomogram_size)
            R = euler_zyz_to_rotation_matrix(
                row["rlnAngleRot"], row["rlnAngleTilt"], row["rlnAnglePsi"]
            )
            part_rot = rotate_volume(template, R)
            if wedge_params:
                from ..core.wedge import apply_wedge
                part_rot = apply_wedge(part_rot, ftype=wedge_params[0], ymintilt=wedge_params[1],
                                      ymaxtilt=wedge_params[2], xmintilt=wedge_params[3], xmaxtilt=wedge_params[4])
            # scale particle std to (particle_scale_ratio * noise_sigma)
            pstd = float(np.std(part_rot)) + 1e-12
            target_std = particle_scale_ratio * noise_sigma
            part_rot = part_rot * (target_std / pstd)
            # embed uses 0-based; star_to_absolute_pixels gives 0-based
            _embed_with_debug(tomogram, part_rot, x_abs, y_abs, z_abs, tomogram_size, embed_count)
            embed_count += 1
    except Exception as e:
        write_error(str(e), args=cli_args, config=cfg, config_path=config_path)
        raise
    logger.info("Built tomogram with Gaussian background (sigma=%.2f)", noise_sigma)

    # ---------- 3. Save tomogram, tbl, vll, star ----------
    out_tomos = out_root / "out_tomograms"
    out_tomos.mkdir(parents=True, exist_ok=True)
    tomo_path = out_tomos / "tomo1.mrc"
    with mrcfile.new(str(tomo_path), overwrite=True) as mrc:
        mrc.set_data(tomogram)
        mrc.voxel_size = pixel_size
    logger.info("Saved tomogram %s", tomo_path)

    vll_path = out_tomos / "tomograms.vll"
    abs_tomo = tomo_path.resolve()
    with open(vll_path, "w") as f:
        f.write(str(abs_tomo) + "\n")
    logger.info("Saved vll %s", vll_path)

    star_df = _star_rows_to_dataframe(sample_rows)
    star_path = out_tomos / "particles.star"
    starfile.write(star_df, str(star_path), overwrite=True)
    logger.info("Saved star %s", star_path)

    tbl_path = out_tomos / "particles.tbl"
    _write_tbl_from_star(star_df, str(tbl_path), pixel_size, tomogram_size)
    logger.info("Saved tbl %s", tbl_path)

    # ---------- 4. CROP subtomograms (using star internal) ----------
    out_sub = out_root / "out_subtomograms"
    out_sub.mkdir(parents=True, exist_ok=True)
    try:
        from ..core.crop import crop_volume, save_subtomo
    except ImportError:
        from pydynamo.core.crop import crop_volume, save_subtomo

    for i, row in progress_iter(list(enumerate(sample_rows)), total=len(sample_rows), desc="gen crop"):
        x_abs, y_abs, z_abs = star_to_absolute_pixels(row, pixel_size, tomogram_size)
        # crop_volume uses 1-based position; our star gives 0-based -> add 1
        pos = (z_abs + 1, y_abs + 1, x_abs + 1)
        sub, report = crop_volume(tomogram, L, pos, fill=0)
        if sub is not None:
            save_subtomo(sub, str(out_sub / f"particle_{i+1:012d}.mrc"))
    sub_star_df = star_df.copy()
    sub_star_df["rlnImageName"] = [f"particle_{i+1:012d}.mrc" for i in range(len(star_df))]
    starfile.write(sub_star_df, str(out_sub / "particles.star"), overwrite=True)
    _write_tbl_from_star(sub_star_df, str(out_sub / "particles.tbl"), pixel_size, tomogram_size)
    logger.info("Cropped %d subtomograms to %s", n_particles, out_sub)

    # ---------- 5. Classification set: real + noise ----------
    out_4cl = out_root / "out_tomograms4classification"
    out_4cl.mkdir(parents=True, exist_ok=True)
    noise_scale = float(np.std(tomogram)) * 0.5
    if noise_scale <= 0:
        noise_scale = 1.0
    all_star_rows = []
    for i, row in progress_iter(list(enumerate(sample_rows)), total=len(sample_rows), desc="gen class-real"):
        src = out_sub / f"particle_{i+1:012d}.mrc"
        dst = out_4cl / f"particle_{i+1:012d}.mrc"
        if src.exists():
            import shutil
            shutil.copy(str(src), str(dst))
        r = {**row, "rlnImageName": f"particle_{i+1:012d}.mrc", "ref": 1}
        all_star_rows.append(r)
    try:
        from ..core.crop import save_subtomo
    except ImportError:
        from pydynamo.core.crop import save_subtomo
    for i in progress_iter(range(n_noise), total=n_noise, desc="gen class-noise"):
        noise = np.random.randn(L, L, L).astype(np.float32) * noise_scale
        save_subtomo(noise, str(out_4cl / f"noise_{i+1:012d}.mrc"))
        r = sample_rows[i % n_particles].copy()
        r["tag"] = n_particles + i + 1
        r["rlnImageName"] = f"noise_{i+1:012d}.mrc"
        r["ref"] = 2
        r["rlnCenteredCoordinateXAngst"] = r["rlnCenteredCoordinateYAngst"] = r["rlnCenteredCoordinateZAngst"] = 0.0
        r["rlnAngleRot"] = r["rlnAngleTilt"] = r["rlnAnglePsi"] = 0.0
        all_star_rows.append(r)

    cl_star_df = _star_rows_to_dataframe(all_star_rows)
    cl_star_df["rlnImageName"] = [r["rlnImageName"] for r in all_star_rows]
    starfile.write(cl_star_df, str(out_4cl / "particles.star"), overwrite=True)
    _write_tbl_from_star(cl_star_df, str(out_4cl / "particles.tbl"), pixel_size, tomogram_size)
    with open(out_4cl / "tomograms.vll", "w") as f:
        f.write(str(abs_tomo) + "\n")
    logger.info("Saved classification set to %s (%d real + %d noise)", out_4cl, n_particles, n_noise)


def _star_rows_to_dataframe(rows: List[dict]) -> pd.DataFrame:
    """Build DataFrame from internal RELION star rows."""
    return pd.DataFrame(rows)


def _write_tbl_from_star(
    star_df: pd.DataFrame,
    path: str,
    pixel_size: float,
    tomogram_size: Tuple[int, int, int],
) -> None:
    """Convert internal star DataFrame to Dynamo tbl (TomoPANDA-pick relion_star_to_dynamo_tbl)."""
    try:
        from ..io import relion_star_to_dynamo_tbl
    except ImportError:
        from pydynamo.io import relion_star_to_dynamo_tbl
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as tf:
        starfile.write(star_df, tf.name, overwrite=True)
        try:
            relion_star_to_dynamo_tbl(
                star_path=tf.name,
                pixel_size=pixel_size,
                tomogram_size=tomogram_size,
                output_file=path,
            )
        finally:
            os.unlink(tf.name)


def main():
    run()


if __name__ == "__main__":
    main()

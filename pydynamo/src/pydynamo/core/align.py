"""
Simplified alignment: FFT-based cross-correlation with angular/shift search.
Matches Dynamo concept: search orientations and shifts, output best CC.
Supports multigrid (coarse-to-fine) and optional PyTorch GPU.
"""
import warnings

import numpy as np
from scipy.ndimage import map_coordinates, shift, uniform_filter, zoom
from scipy.fft import fftn, ifftn, fftfreq
from scipy.spatial.transform import Rotation

from .wedge import get_wedge_mask

_ROSEMAN_APPROX_WARNED = False


def _angle_list(lo: float, hi: float, step: float, inclusive_upper: bool = False) -> np.ndarray:
    """Build sampled angle list with light validation."""
    step = float(step)
    if step <= 0:
        raise ValueError("angle step must be > 0")
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    if inclusive_upper:
        return np.arange(lo, hi + step * 0.5, step, dtype=np.float32)
    return np.arange(lo, hi, step, dtype=np.float32)


def _iter_integer_shifts(shift_search: int, shift_mode: str, shift_center: tuple = (0, 0, 0)):
    """Yield integer shift offsets for configured search geometry."""
    r = int(max(0, shift_search))
    mode = str(shift_mode or "cube").lower()
    cx, cy, cz = int(shift_center[0]), int(shift_center[1]), int(shift_center[2])
    if mode == "center_only":
        yield 0, 0, 0
        return
    for sx in range(-r, r + 1):
        for sy in range(-r, r + 1):
            for sz in range(-r, r + 1):
                if mode in ("ellipsoid_center",):
                    if r > 0:
                        rr = ((cx + sx) * (cx + sx) + (cy + sy) * (cy + sy) + (cz + sz) * (cz + sz)) / float(r * r)
                        if rr > 1.0:
                            continue
                elif mode in ("ellipsoid", "ellipsoid_follow"):
                    if r > 0:
                        rr = (sx * sx + sy * sy + sz * sz) / float(r * r)
                        if rr > 1.0:
                            continue
                elif mode == "cylinder_z_center":
                    if r > 0 and (((cx + sx) * (cx + sx) + (cy + sy) * (cy + sy)) / float(r * r) > 1.0):
                        continue
                elif mode == "cylinder_z_follow":
                    if r > 0 and ((sx * sx + sy * sy) / float(r * r) > 1.0):
                        continue
                yield sx, sy, sz


def _parabolic_subpixel_offset(v_minus: float, v0: float, v_plus: float) -> float:
    """Estimate subpixel peak offset in [-1,1] by parabola fit."""
    denom = (v_minus - 2.0 * v0 + v_plus)
    if abs(denom) < 1e-8:
        return 0.0
    delta = 0.5 * (v_minus - v_plus) / denom
    return float(np.clip(delta, -1.0, 1.0))


def _subpixel_offset_3d_quadratic(cc_at) -> tuple[float, float, float] | None:
    """
    Estimate local subpixel peak offset using 3D quadratic fit on 3x3x3 neighborhood.
    Returns (ox, oy, oz) in [-1, 1] or None when fit is unstable.
    """
    rows = []
    vals = []
    for ox in (-1.0, 0.0, 1.0):
        for oy in (-1.0, 0.0, 1.0):
            for oz in (-1.0, 0.0, 1.0):
                v = float(cc_at(ox, oy, oz))
                rows.append([ox * ox, oy * oy, oz * oz, ox * oy, ox * oz, oy * oz, ox, oy, oz, 1.0])
                vals.append(v)
    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(vals, dtype=np.float64)
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    a, b2, c, d, e, f, g, h, i, _ = coef
    H = np.asarray(
        [
            [2.0 * a, d, e],
            [d, 2.0 * b2, f],
            [e, f, 2.0 * c],
        ],
        dtype=np.float64,
    )
    grad = np.asarray([g, h, i], dtype=np.float64)
    try:
        eigvals = np.linalg.eigvalsh(H)
        if not np.all(eigvals < -1e-8):
            return None
        delta = -np.linalg.solve(H, grad)
    except Exception:
        return None
    if not np.all(np.isfinite(delta)):
        return None
    delta = np.clip(delta, -1.0, 1.0)
    return float(delta[0]), float(delta[1]), float(delta[2])


def _apply_fourier_support_np(vol: np.ndarray, wedge_mask: np.ndarray) -> np.ndarray:
    """Apply per-particle Fourier support mask for wedge-aware scoring."""
    f = np.fft.fftshift(fftn(vol))
    f *= wedge_mask
    return np.real(ifftn(np.fft.ifftshift(f))).astype(np.float32)


def _resolve_wedge_apply_to(mode: str, fsampling: dict | None) -> str:
    """Resolve wedge application side: both|particle|template."""
    m = str(mode or "both").lower()
    if m in ("both", "particle", "template"):
        return m
    if m != "auto":
        return "both"
    if not fsampling:
        return "both"
    fs1 = fsampling.get("fs1", np.nan)
    fs2 = fsampling.get("fs2", np.nan)
    try:
        fs1 = float(fs1)
        fs2 = float(fs2)
    except Exception:
        return "both"
    if fs1 > 0 and fs2 <= 0:
        return "particle"
    if fs2 > 0 and fs1 <= 0:
        return "template"
    return "both"


def _normalize_aperture(value, max_deg: float = 360.0) -> float:
    """Normalize aperture input from scalar or [lo, hi]."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lo, hi = float(value[0]), float(value[1])
        if lo == 0:
            aperture = hi
        else:
            aperture = abs(hi - lo)
    else:
        aperture = float(value)
    return float(np.clip(aperture, 0.0, max_deg))


def _compose_zxz(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """Compose ZXZ rotations: R = R(a) * R(b), return ZXZ triplet."""
    ra = Rotation.from_euler("ZXZ", [a[0], a[1], a[2]], degrees=True)
    rb = Rotation.from_euler("ZXZ", [b[0], b[1], b[2]], degrees=True)
    rc = ra * rb
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Gimbal lock detected")
        tdrot, tilt, narot = rc.as_euler("ZXZ", degrees=True)
    return float(tdrot), float(tilt), float(narot)


def _dynamo_angleincrement2list(
    cone_range: float,
    cone_sampling: float,
    inplane_range: float,
    inplane_sampling: float,
    old_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Approximate Dynamo angleincrement2list:
    - cone_range is aperture (0..360)
    - cone_sampling controls spherical spacing of axis orientations
    - inplane_range is symmetric around seed narot
    """
    cone_sampling = max(float(cone_sampling), 1e-6)
    inplane_sampling = max(float(inplane_sampling), 1e-6)
    aperture_half = _normalize_aperture(cone_range, 360.0) / 2.0
    inplane_half = _normalize_aperture(inplane_range, 360.0) / 2.0
    old_tdrot, old_tilt, old_narot = float(old_angles[0]), float(old_angles[1]), float(old_angles[2])

    list_tilt_north = np.arange(0.0, aperture_half + cone_sampling * 0.5, cone_sampling, dtype=np.float32)
    if list_tilt_north.size == 0:
        list_tilt_north = np.asarray([0.0], dtype=np.float32)

    angles: list[tuple[float, float, float]] = []
    last_mat = None
    for tilt_north in list_tilt_north:
        if float(tilt_north) in (0.0, 180.0):
            list_tdrots_north = [0.0]
        else:
            c = float(np.cos(np.deg2rad(float(tilt_north))))
            s = float(np.sin(np.deg2rad(float(tilt_north))))
            num = np.cos(np.deg2rad(cone_sampling)) - c * c
            den = max(s * s, 1e-8)
            ratio = np.clip(num / den, -1.0, 1.0)
            interval_tdrot = float(np.rad2deg(np.arccos(ratio)))
            if not np.isfinite(interval_tdrot) or interval_tdrot <= 0:
                list_tdrots_north = [0.0]
            else:
                n_sectors = int(np.floor(360.0 / interval_tdrot)) + 1
                list_tdrots_north = np.linspace(0.0, 360.0, n_sectors + 1, endpoint=False, dtype=np.float32).tolist()

        for tdrot_north in list_tdrots_north:
            tdrot, tilt, narot_temp = _compose_zxz(
                (old_tdrot, old_tilt, 0.0),
                (float(tdrot_north), float(tilt_north), 0.0),
            )
            narot_seed = old_narot + narot_temp

            pos = np.arange(narot_seed, narot_seed + inplane_half + inplane_sampling * 0.5, inplane_sampling, dtype=np.float32)
            neg = np.arange(narot_seed - inplane_sampling, narot_seed - inplane_half - inplane_sampling * 0.5, -inplane_sampling, dtype=np.float32)
            list_narot = np.concatenate([pos, neg]) if pos.size or neg.size else np.asarray([narot_seed], dtype=np.float32)
            if list_narot.size == 0:
                list_narot = np.asarray([narot_seed], dtype=np.float32)

            for narot in list_narot:
                cand = (float(tdrot), float(tilt), float(narot))
                mat = Rotation.from_euler("ZXZ", cand, degrees=True).as_matrix()
                if last_mat is not None and np.max(np.abs(last_mat - mat)) <= 1e-8:
                    continue
                angles.append(cand)
                last_mat = mat

    if not angles:
        return np.asarray([[old_tdrot, old_tilt, old_narot]], dtype=np.float32)
    return np.asarray(angles, dtype=np.float32)


def _downsample(vol: np.ndarray, factor: int) -> np.ndarray:
    """Downsample volume by integer factor."""
    if factor <= 1:
        return vol
    new_shape = tuple(max(1, s // factor) for s in vol.shape)
    zoom_f = tuple(n / o for n, o in zip(new_shape, vol.shape))
    return zoom(vol, zoom_f, order=1).astype(np.float32)


def _get_device(config_device: str) -> str:
    """Resolve device: cpu, cuda, or auto."""
    if config_device in ("cpu", "cuda"):
        return config_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute normalized CC between two same-sized volumes on shared support."""
    if mask is not None:
        valid = mask.astype(bool)
        if not np.any(valid):
            return 0.0
        a = a.astype(np.float64)[valid]
        b = b.astype(np.float64)[valid]
    else:
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


def _local_normalized_cross_correlation(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray = None,
    win: int = 5,
    eps: float = 1e-8,
) -> float:
    """
    Local normalized CC (Roseman-like approximation) on full volume.
    Uses local mean/std within cubic window then averages normalized product.
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    if win <= 1:
        win = 1
    if win % 2 == 0:
        win = win + 1

    if mask is None:
        mask_f = np.ones_like(a, dtype=np.float32)
    else:
        mask_f = mask.astype(np.float32, copy=False)

    # Mask-aware local statistics approximate Dynamo localnc behavior better
    # than unweighted window normalization on the full volume.
    local_w = uniform_filter(mask_f, size=win, mode="constant")
    local_w = np.maximum(local_w, eps)
    mean_a = uniform_filter(a * mask_f, size=win, mode="constant") / local_w
    mean_b = uniform_filter(b * mask_f, size=win, mode="constant") / local_w
    var_a = uniform_filter((a * a) * mask_f, size=win, mode="constant") / local_w - mean_a * mean_a
    var_b = uniform_filter((b * b) * mask_f, size=win, mode="constant") / local_w - mean_b * mean_b
    var_a = np.maximum(var_a, eps)
    var_b = np.maximum(var_b, eps)
    z_a = (a - mean_a) / np.sqrt(var_a)
    z_b = (b - mean_b) / np.sqrt(var_b)
    prod = z_a * z_b
    valid = mask_f > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(prod[valid]))


def _compute_cc_np(
    particle_eval: np.ndarray,
    ref_eval: np.ndarray,
    cc_mode: str,
    mask: np.ndarray = None,
    cc_local_window: int = 5,
    cc_local_eps: float = 1e-8,
) -> float:
    """Dispatch CPU correlation backend."""
    mode = str(cc_mode or "ncc").lower()
    if mode == "ncc":
        return normalized_cross_correlation(particle_eval, ref_eval, mask=mask)
    if mode == "roseman_local":
        global _ROSEMAN_APPROX_WARNED
        if not _ROSEMAN_APPROX_WARNED:
            warnings.warn(
                "cc_mode=roseman_local is an approximate local CC backend, not strict Dynamo parity",
                RuntimeWarning,
            )
            _ROSEMAN_APPROX_WARNED = True
        return _local_normalized_cross_correlation(
            particle_eval,
            ref_eval,
            mask=mask,
            win=cc_local_window,
            eps=cc_local_eps,
        )
    raise ValueError(f"Unsupported cc_mode: {cc_mode}")


def rotate_volume(vol: np.ndarray, tdrot: float, tilt: float, narot: float) -> np.ndarray:
    """Apply ZXZ Euler rotation (degrees) to volume."""
    r = Rotation.from_euler("ZXZ", [tdrot, tilt, narot], degrees=True)
    mat = r.as_matrix()
    center = np.array(vol.shape) / 2.0 - 0.5
    coords = np.mgrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]].astype(float)
    coords = coords - center.reshape(3, 1, 1, 1)
    coords_flat = coords.reshape(3, -1)
    rotated = mat @ coords_flat
    rotated += center.reshape(3, 1)
    coords_new = rotated.reshape(3, vol.shape[0], vol.shape[1], vol.shape[2])
    return map_coordinates(vol, coords_new, order=1, mode="constant", cval=0).astype(np.float32)


def _align_single_scale(
    particle: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    tdrot_step: float,
    cone_step: float,
    inplane_step: float,
    shift_search: int,
    tdrot_lo: float = 0.0,
    tdrot_hi: float = 360.0,
    tilt_lo: float = 0.0,
    tilt_hi: float = 180.0,
    inplane_lo: float = 0.0,
    inplane_hi: float = 360.0,
    shift_center: tuple = (0, 0, 0),
    shift_mode: str = "cube",
    subpixel: bool = True,
    cc_mode: str = "ncc",
    cc_local_window: int = 5,
    cc_local_eps: float = 1e-8,
    angle_sampling_mode: str = "legacy",
    old_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wedge_mask: np.ndarray = None,
    wedge_apply_to: str = "both",
    subpixel_method: str = "auto",
) -> tuple:
    """Single-scale alignment search. Returns (tdrot, tilt, narot, dx, dy, dz, cc)."""
    ref_m = reference * mask
    ref_m = ref_m - np.mean(ref_m[mask])
    ref_m[~mask] = 0
    particle_m = particle * mask
    cx, cy, cz = shift_center

    best_cc = -2.0
    best_params = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    best_ref_rot = None

    shifts = list(_iter_integer_shifts(shift_search, shift_mode, shift_center=shift_center))
    if not shifts:
        shifts = [(0, 0, 0)]
    apply_to = str(wedge_apply_to or "both").lower()
    use_particle_wedge = wedge_mask is not None and apply_to in ("both", "particle")
    use_template_wedge = wedge_mask is not None and apply_to in ("both", "template")
    particle_eval = _apply_fourier_support_np(particle_m, wedge_mask) if use_particle_wedge else particle_m
    mode = str(angle_sampling_mode or "legacy").lower()
    if mode == "dynamo":
        cone_aperture = _normalize_aperture([tilt_lo, tilt_hi], 360.0)
        inplane_aperture = _normalize_aperture([inplane_lo, inplane_hi], 360.0)
        triplets = _dynamo_angleincrement2list(
            cone_aperture,
            cone_step,
            inplane_aperture,
            inplane_step,
            old_angles=old_angles,
        )
    else:
        tdrots = _angle_list(tdrot_lo, tdrot_hi, tdrot_step, inclusive_upper=False)
        tilts = _angle_list(tilt_lo, min(tilt_hi, 180.0), cone_step, inclusive_upper=True)
        t = []
        for tilt in tilts:
            inplanes = np.arange(inplane_lo, inplane_hi, inplane_step) if tilt not in (0, 180) else [0]
            for narot in inplanes:
                for tdrot in tdrots:
                    t.append((float(tdrot), float(tilt), float(narot)))
        triplets = np.asarray(t, dtype=np.float32) if t else np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    for tdrot, tilt, narot in triplets:
        ref_rot = rotate_volume(ref_m, float(tdrot), float(tilt), float(narot))
        if use_template_wedge:
            ref_rot = _apply_fourier_support_np(ref_rot, wedge_mask)
        for sx, sy, sz in shifts:
            dx, dy, dz = cx + sx, cy + sy, cz + sz
            ref_shifted = shift(ref_rot, (dx, dy, dz), order=1, mode="constant", cval=0)
            cc = _compute_cc_np(
                particle_eval,
                ref_shifted,
                cc_mode=cc_mode,
                mask=mask,
                cc_local_window=cc_local_window,
                cc_local_eps=cc_local_eps,
            )
            if cc > best_cc:
                best_cc = cc
                best_params = (float(tdrot), float(tilt), float(narot), float(dx), float(dy), float(dz))
                best_ref_rot = ref_rot

    if subpixel and best_ref_rot is not None:
        tdrot, tilt, narot, dx, dy, dz = best_params

        def _cc_at(sx: float, sy: float, sz: float) -> float:
            ref_shifted = shift(best_ref_rot, (sx, sy, sz), order=1, mode="constant", cval=0)
            return _compute_cc_np(
                particle_eval,
                ref_shifted,
                cc_mode=cc_mode,
                mask=mask,
                cc_local_window=cc_local_window,
                cc_local_eps=cc_local_eps,
            )

        # Prefer guarded 3D quadratic local fit; fallback to axis-wise 1D parabola.
        d3 = None
        if str(subpixel_method or "auto").lower() in ("auto", "quadratic3d", "3d"):
            d3 = _subpixel_offset_3d_quadratic(lambda ox, oy, oz: _cc_at(dx + ox, dy + oy, dz + oz))
        if d3 is not None:
            dx = dx + d3[0]
            dy = dy + d3[1]
            dz = dz + d3[2]
        else:
            c0 = _cc_at(dx, dy, dz)
            c_xm, c_xp = _cc_at(dx - 1.0, dy, dz), _cc_at(dx + 1.0, dy, dz)
            dx = dx + _parabolic_subpixel_offset(c_xm, c0, c_xp)
            c0 = _cc_at(dx, dy, dz)
            c_ym, c_yp = _cc_at(dx, dy - 1.0, dz), _cc_at(dx, dy + 1.0, dz)
            dy = dy + _parabolic_subpixel_offset(c_ym, c0, c_yp)
            c0 = _cc_at(dx, dy, dz)
            c_zm, c_zp = _cc_at(dx, dy, dz - 1.0), _cc_at(dx, dy, dz + 1.0)
            dz = dz + _parabolic_subpixel_offset(c_zm, c0, c_zp)
        best_cc = _cc_at(dx, dy, dz)
        best_params = (tdrot, tilt, narot, float(dx), float(dy), float(dz))

    return best_params + (best_cc,)


def _align_one_particle_torch_gpu(
    particle: np.ndarray,
    reference: np.ndarray,
    mask,
    tdrot_step: float,
    cone_step: float,
    cone_range: tuple,
    tdrot_range: tuple,
    inplane_step: float,
    inplane_range: tuple,
    shift_search: int,
    lowpass_angstrom: float,
    pixel_size: float,
    multigrid_levels: int,
    shift_mode: str = "cube",
    subpixel: bool = True,
    cc_mode: str = "ncc",
    cc_local_window: int = 5,
    cc_local_eps: float = 1e-8,
    angle_sampling_mode: str = "legacy",
    old_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wedge_mask: np.ndarray = None,
    wedge_apply_to: str = "both",
    fsampling: dict | None = None,
    fsampling_mode: str = "none",
    subpixel_method: str = "auto",
    device_id: int = None,
):
    """PyTorch GPU path. Uses GPU for shift + NCC search."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    # Same logic as CPU path until batched GPU rotation+CC implemented
    if mask is None:
        mask = np.ones_like(particle, dtype=bool)
    tdrot_lo, tdrot_hi = tdrot_range[0], tdrot_range[1]
    tilt_lo, tilt_hi = cone_range[0], cone_range[1]
    inplane_lo, inplane_hi = inplane_range[0], inplane_range[1]
    p, r = particle, reference
    if lowpass_angstrom and pixel_size > 0:
        p = _lowpass_filter(particle, lowpass_angstrom, pixel_size)
        r = _lowpass_filter(reference, lowpass_angstrom, pixel_size)
    device = torch.device(f"cuda:{device_id}") if device_id is not None else torch.device("cuda")

    if multigrid_levels <= 1:
        return _align_single_scale_torch_gpu(
            p,
            r,
            mask,
            tdrot_step=tdrot_step,
            cone_step=cone_step,
            inplane_step=inplane_step,
            shift_search=shift_search,
            tdrot_lo=tdrot_lo, tdrot_hi=tdrot_hi,
            tilt_lo=tilt_lo, tilt_hi=tilt_hi, inplane_lo=inplane_lo, inplane_hi=inplane_hi,
            shift_mode=shift_mode,
            subpixel=subpixel,
            cc_mode=cc_mode,
            cc_local_window=cc_local_window,
            cc_local_eps=cc_local_eps,
            angle_sampling_mode=angle_sampling_mode,
            old_angles=old_angles,
            wedge_mask=wedge_mask,
            wedge_apply_to=wedge_apply_to,
            subpixel_method=subpixel_method,
            device=device,
        )
    factor = 2
    p_coarse = _downsample(p, factor)
    ref_coarse = _downsample(r, factor)
    mask_coarse = _downsample(mask.astype(np.float32), factor) > 0.5
    coarse_step_c = max(cone_step * 2, 30.0)
    coarse_step_i = max(inplane_step * 2, 30.0)
    tdrot, tilt, narot, dx_c, dy_c, dz_c, _ = _align_single_scale_torch_gpu(
        p_coarse, ref_coarse, mask_coarse,
        tdrot_step=max(tdrot_step * 2.0, 30.0),
        cone_step=coarse_step_c,
        inplane_step=coarse_step_i,
        shift_search=max(1, shift_search // 2),
        tdrot_lo=tdrot_lo, tdrot_hi=tdrot_hi,
        shift_mode=shift_mode,
        subpixel=False,
        cc_mode=cc_mode,
        cc_local_window=cc_local_window,
        cc_local_eps=cc_local_eps,
        angle_sampling_mode=angle_sampling_mode,
        old_angles=old_angles,
        wedge_mask=wedge_mask,
        wedge_apply_to=wedge_apply_to,
        subpixel_method=subpixel_method,
        device=device,
    )
    shift_center = (int(dx_c * factor), int(dy_c * factor), int(dz_c * factor))
    margin = max(cone_step, inplane_step)
    tilt_lo = max(0, tilt - margin)
    tilt_hi = min(180, tilt + margin)
    inplane_lo = (narot - margin) % 360
    inplane_hi = (narot + margin + 1) % 360
    if inplane_lo > inplane_hi:
        inplane_lo, inplane_hi = 0, 360
    tdrot_lo = (tdrot - margin) % 360
    tdrot_hi = (tdrot + margin + 1) % 360
    if tdrot_lo > tdrot_hi:
        tdrot_lo, tdrot_hi = 0, 360
    return _align_single_scale_torch_gpu(
        p, r, mask,
        tdrot_step=tdrot_step,
        cone_step=cone_step,
        inplane_step=inplane_step,
        shift_search=shift_search,
        tdrot_lo=tdrot_lo, tdrot_hi=tdrot_hi if tdrot_hi > tdrot_lo else 360,
        tilt_lo=tilt_lo, tilt_hi=tilt_hi,
        inplane_lo=inplane_lo, inplane_hi=inplane_hi if inplane_hi > inplane_lo else 360,
        shift_center=shift_center,
        shift_mode=shift_mode,
        subpixel=subpixel,
        cc_mode=cc_mode,
        cc_local_window=cc_local_window,
        cc_local_eps=cc_local_eps,
        angle_sampling_mode=angle_sampling_mode,
        old_angles=(tdrot, tilt, narot),
        wedge_mask=wedge_mask,
        wedge_apply_to=wedge_apply_to,
        subpixel_method=subpixel_method,
        device=device,
    )


def _shift_tensor_zero(vol_t, dx: int, dy: int, dz: int):
    """Shift 3D tensor with zero padding (not circular roll)."""
    import torch

    out = torch.zeros_like(vol_t)
    sz0, sz1, sz2 = vol_t.shape

    if dx >= 0:
        src0_s, src0_e = 0, sz0 - dx
        dst0_s, dst0_e = dx, sz0
    else:
        src0_s, src0_e = -dx, sz0
        dst0_s, dst0_e = 0, sz0 + dx

    if dy >= 0:
        src1_s, src1_e = 0, sz1 - dy
        dst1_s, dst1_e = dy, sz1
    else:
        src1_s, src1_e = -dy, sz1
        dst1_s, dst1_e = 0, sz1 + dy

    if dz >= 0:
        src2_s, src2_e = 0, sz2 - dz
        dst2_s, dst2_e = dz, sz2
    else:
        src2_s, src2_e = -dz, sz2
        dst2_s, dst2_e = 0, sz2 + dz

    if src0_s >= src0_e or src1_s >= src1_e or src2_s >= src2_e:
        return out
    out[dst0_s:dst0_e, dst1_s:dst1_e, dst2_s:dst2_e] = vol_t[src0_s:src0_e, src1_s:src1_e, src2_s:src2_e]
    return out


def _ncc_torch(a_t, b_t, mask_t):
    """Normalized cross correlation on masked voxels (torch tensors)."""
    import torch

    ma = a_t[mask_t]
    mb = b_t[mask_t]
    if ma.numel() == 0:
        return 0.0
    ma = ma - torch.mean(ma)
    mb = mb - torch.mean(mb)
    denom = torch.sqrt(torch.sum(ma * ma) * torch.sum(mb * mb))
    if float(denom) < 1e-12:
        return 0.0
    return float(torch.sum(ma * mb) / denom)


def _local_normalized_cross_correlation_torch(
    a_t,
    b_t,
    mask_t,
    win: int = 5,
    eps: float = 1e-8,
):
    """Mask-aware local normalized CC using torch pooling."""
    import torch
    import torch.nn.functional as F

    if win <= 1:
        win = 1
    if win % 2 == 0:
        win = win + 1
    pad = win // 2

    a = a_t.float().unsqueeze(0).unsqueeze(0)
    b = b_t.float().unsqueeze(0).unsqueeze(0)
    w = mask_t.float().unsqueeze(0).unsqueeze(0)

    local_w = F.avg_pool3d(w, kernel_size=win, stride=1, padding=pad)
    local_w = torch.clamp(local_w, min=eps)

    mean_a = F.avg_pool3d(a * w, kernel_size=win, stride=1, padding=pad) / local_w
    mean_b = F.avg_pool3d(b * w, kernel_size=win, stride=1, padding=pad) / local_w
    var_a = F.avg_pool3d((a * a) * w, kernel_size=win, stride=1, padding=pad) / local_w - mean_a * mean_a
    var_b = F.avg_pool3d((b * b) * w, kernel_size=win, stride=1, padding=pad) / local_w - mean_b * mean_b
    var_a = torch.clamp(var_a, min=eps)
    var_b = torch.clamp(var_b, min=eps)

    z_a = (a - mean_a) / torch.sqrt(var_a)
    z_b = (b - mean_b) / torch.sqrt(var_b)
    prod = z_a * z_b

    denom = torch.sum(w)
    if float(denom) <= 0:
        return 0.0
    return float(torch.sum(prod * w) / denom)


def _align_single_scale_torch_gpu(
    particle: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    tdrot_step: float,
    cone_step: float,
    inplane_step: float,
    shift_search: int,
    tdrot_lo: float = 0.0,
    tdrot_hi: float = 360.0,
    tilt_lo: float = 0.0,
    tilt_hi: float = 180.0,
    inplane_lo: float = 0.0,
    inplane_hi: float = 360.0,
    shift_center: tuple = (0, 0, 0),
    shift_mode: str = "cube",
    subpixel: bool = True,
    cc_mode: str = "ncc",
    cc_local_window: int = 5,
    cc_local_eps: float = 1e-8,
    angle_sampling_mode: str = "legacy",
    old_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wedge_mask: np.ndarray = None,
    wedge_apply_to: str = "both",
    subpixel_method: str = "auto",
    device=None,
) -> tuple:
    """Single-scale alignment where shift + NCC evaluation runs on GPU."""
    import torch
    import torch.nn.functional as F

    if device is None:
        device = torch.device("cuda")

    if mask is None:
        mask = np.ones_like(particle, dtype=bool)
    mask_t = torch.as_tensor(mask.astype(bool), device=device)
    part_t = torch.as_tensor((particle * mask).astype(np.float32), device=device)
    ref_src_t = torch.as_tensor(reference.astype(np.float32), device=device)
    apply_to = str(wedge_apply_to or "both").lower()
    use_particle_wedge = wedge_mask is not None and apply_to in ("both", "particle")
    use_template_wedge = wedge_mask is not None and apply_to in ("both", "template")
    wm_t = torch.as_tensor(wedge_mask.astype(np.float32), device=device) if wedge_mask is not None else None
    if use_particle_wedge:
        part_eval_t = torch.real(
            torch.fft.ifftn(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fftn(part_t)) * wm_t))
        )
    else:
        part_eval_t = part_t
    cx, cy, cz = shift_center

    def _rotate_volume_torch_gpu(vol_t, tdrot: float, tilt: float, narot: float):
        """Rotate volume on GPU using trilinear sampling (ZXZ, degrees)."""
        mat_np = Rotation.from_euler("ZXZ", [tdrot, tilt, narot], degrees=True).as_matrix().astype(np.float32)
        mat_t = torch.as_tensor(mat_np, device=device)
        d, h, w = vol_t.shape
        center = torch.tensor([(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0], device=device, dtype=torch.float32)

        zz, yy, xx = torch.meshgrid(
            torch.arange(d, device=device, dtype=torch.float32),
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([zz, yy, xx], dim=0).reshape(3, -1)
        coords_centered = coords - center[:, None]
        rot = mat_t @ coords_centered
        rot = rot + center[:, None]
        z = rot[0].reshape(d, h, w)
        y = rot[1].reshape(d, h, w)
        x = rot[2].reshape(d, h, w)
        grid = torch.stack(
            [
                (2.0 * x / max(1.0, float(w - 1))) - 1.0,
                (2.0 * y / max(1.0, float(h - 1))) - 1.0,
                (2.0 * z / max(1.0, float(d - 1))) - 1.0,
            ],
            dim=-1,
        ).unsqueeze(0)
        out = F.grid_sample(
            vol_t.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return out[0, 0]

    best_cc = -2.0
    best_params = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    best_ref_t = None
    shifts = list(_iter_integer_shifts(shift_search, shift_mode, shift_center=shift_center))
    if not shifts:
        shifts = [(0, 0, 0)]
    mode = str(angle_sampling_mode or "legacy").lower()
    if mode == "dynamo":
        cone_aperture = _normalize_aperture([tilt_lo, tilt_hi], 360.0)
        inplane_aperture = _normalize_aperture([inplane_lo, inplane_hi], 360.0)
        triplets = _dynamo_angleincrement2list(
            cone_aperture,
            cone_step,
            inplane_aperture,
            inplane_step,
            old_angles=old_angles,
        )
    else:
        tdrots = _angle_list(tdrot_lo, tdrot_hi, tdrot_step, inclusive_upper=False)
        tilts = _angle_list(tilt_lo, min(tilt_hi, 180.0), cone_step, inclusive_upper=True)
        t = []
        for tilt in tilts:
            inplanes = np.arange(inplane_lo, inplane_hi, inplane_step) if tilt not in (0, 180) else [0]
            for narot in inplanes:
                for tdrot in tdrots:
                    t.append((float(tdrot), float(tilt), float(narot)))
        triplets = np.asarray(t, dtype=np.float32) if t else np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    for tdrot, tilt, narot in triplets:
        ref_t = _rotate_volume_torch_gpu(ref_src_t, float(tdrot), float(tilt), float(narot))
        if use_template_wedge:
            ref_t = torch.real(
                torch.fft.ifftn(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fftn(ref_t)) * wm_t))
            )
        for sx, sy, sz in shifts:
            dx, dy, dz = cx + sx, cy + sy, cz + sz
            ref_shifted_t = _shift_tensor_zero(ref_t, int(dx), int(dy), int(dz))
            cc_backend = str(cc_mode or "ncc").lower()
            if cc_backend == "ncc":
                cc = _ncc_torch(part_eval_t, ref_shifted_t, mask_t)
            elif cc_backend == "roseman_local":
                global _ROSEMAN_APPROX_WARNED
                if not _ROSEMAN_APPROX_WARNED:
                    warnings.warn(
                        "cc_mode=roseman_local is an approximate local CC backend, not strict Dynamo parity",
                        RuntimeWarning,
                    )
                    _ROSEMAN_APPROX_WARNED = True
                cc = _local_normalized_cross_correlation_torch(
                    part_eval_t,
                    ref_shifted_t,
                    mask_t,
                    win=cc_local_window,
                    eps=cc_local_eps,
                )
            else:
                raise ValueError(f"Unsupported cc_mode: {cc_mode}")
            if cc > best_cc:
                best_cc = cc
                best_params = (float(tdrot), float(tilt), float(narot), float(dx), float(dy), float(dz))
                best_ref_t = ref_t.detach().clone()

    if subpixel and best_ref_t is not None:
        tdrot, tilt, narot, dx, dy, dz = best_params
        # Keep GPU subpixel objective consistent with the main search objective:
        # evaluate subpixel shifts on GPU with the same cc backend and preprocessed particle.
        def _shift_tensor_interp(vol_t, sx: float, sy: float, sz: float):
            d, h, w = vol_t.shape
            zz, yy, xx = torch.meshgrid(
                torch.arange(d, device=device, dtype=torch.float32),
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            z_src = zz - float(sx)
            y_src = yy - float(sy)
            x_src = xx - float(sz)
            grid = torch.stack(
                [
                    (2.0 * x_src / max(1.0, float(w - 1))) - 1.0,
                    (2.0 * y_src / max(1.0, float(h - 1))) - 1.0,
                    (2.0 * z_src / max(1.0, float(d - 1))) - 1.0,
                ],
                dim=-1,
            ).unsqueeze(0)
            out = F.grid_sample(
                vol_t.unsqueeze(0).unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            return out[0, 0]

        def _cc_at(sx: float, sy: float, sz: float) -> float:
            ref_shifted_t = _shift_tensor_interp(best_ref_t, sx, sy, sz)
            cc_backend = str(cc_mode or "ncc").lower()
            if cc_backend == "ncc":
                return _ncc_torch(part_eval_t, ref_shifted_t, mask_t)
            if cc_backend == "roseman_local":
                return _local_normalized_cross_correlation_torch(
                    part_eval_t,
                    ref_shifted_t,
                    mask_t,
                    win=cc_local_window,
                    eps=cc_local_eps,
                )
            raise ValueError(f"Unsupported cc_mode: {cc_mode}")

        d3 = None
        if str(subpixel_method or "auto").lower() in ("auto", "quadratic3d", "3d"):
            d3 = _subpixel_offset_3d_quadratic(lambda ox, oy, oz: _cc_at(dx + ox, dy + oy, dz + oz))
        if d3 is not None:
            dx = dx + d3[0]
            dy = dy + d3[1]
            dz = dz + d3[2]
        else:
            c0 = _cc_at(dx, dy, dz)
            c_xm, c_xp = _cc_at(dx - 1.0, dy, dz), _cc_at(dx + 1.0, dy, dz)
            dx = dx + _parabolic_subpixel_offset(c_xm, c0, c_xp)
            c0 = _cc_at(dx, dy, dz)
            c_ym, c_yp = _cc_at(dx, dy - 1.0, dz), _cc_at(dx, dy + 1.0, dz)
            dy = dy + _parabolic_subpixel_offset(c_ym, c0, c_yp)
            c0 = _cc_at(dx, dy, dz)
            c_zm, c_zp = _cc_at(dx, dy, dz - 1.0), _cc_at(dx, dy, dz + 1.0)
            dz = dz + _parabolic_subpixel_offset(c_zm, c0, c_zp)
        best_cc = _cc_at(dx, dy, dz)
        best_params = (tdrot, tilt, narot, float(dx), float(dy), float(dz))

    return best_params + (best_cc,)


def _lowpass_filter(vol: np.ndarray, lowpass_angstrom: float, pixel_size: float) -> np.ndarray:
    """Apply lowpass filter in Fourier space. lowpass_angstrom: cut-off in Angstrom."""
    if lowpass_angstrom is None or lowpass_angstrom <= 0:
        return vol
    from scipy.fft import fftn, ifftn, fftfreq
    freq_cut = 1.0 / (lowpass_angstrom / pixel_size)  # in 1/pixel
    f = np.fft.fftshift(fftn(vol))
    nz, ny, nx = vol.shape
    kz = np.fft.fftshift(fftfreq(nz))
    ky = np.fft.fftshift(fftfreq(ny))
    kx = np.fft.fftshift(fftfreq(nx))
    kz3, ky3, kx3 = np.meshgrid(kz, ky, kx, indexing="ij")
    k = np.sqrt(kz3**2 + ky3**2 + kx3**2)
    sigma = freq_cut / 3
    mask = np.exp(-(k**2) / (2 * sigma**2)).astype(np.float32)
    f *= mask
    return np.real(ifftn(np.fft.ifftshift(f))).astype(np.float32)


def align_one_particle(
    particle: np.ndarray,
    reference: np.ndarray,
    mask=None,
    cone_step: float = 15.0,
    tdrot_step: float = 15.0,
    tdrot_range: tuple = (0.0, 360.0),
    cone_range: tuple = (0.0, 180.0),
    inplane_step: float = 15.0,
    inplane_range: tuple = (0.0, 360.0),
    shift_search: int = 3,
    lowpass_angstrom: float = None,
    pixel_size: float = 1.0,
    multigrid_levels: int = 1,
    shift_mode: str = "cube",
    subpixel: bool = True,
    cc_mode: str = "ncc",
    cc_local_window: int = 5,
    cc_local_eps: float = 1e-8,
    angle_sampling_mode: str = "legacy",
    old_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wedge_mask: np.ndarray = None,
    wedge_apply_to: str = "both",
    fsampling: dict | None = None,
    fsampling_mode: str = "none",
    subpixel_method: str = "auto",
    device: str = "cpu",
    device_id: int = None,
):
    """
    Search for best alignment of particle to reference.
    cone_range, inplane_range: (lo, hi) in degrees (Dynamo dcp).
    lowpass_angstrom: bandpass before alignment; None=no filter.
    device: cpu|cuda|auto — cuda uses PyTorch GPU when available.
    Returns (tdrot, tilt, narot, dx, dy, dz, cc).
    """
    use_gpu = (
        device == "cuda"
        or (device == "auto" and _get_device("auto") == "cuda")
    )
    resolved_wedge_mask = wedge_mask
    resolved_apply_to = _resolve_wedge_apply_to(wedge_apply_to, fsampling)
    if str(fsampling_mode or "none").lower() == "table" and fsampling is not None:
        try:
            ftype = int(fsampling.get("ftype", 1))
            ymintilt = float(fsampling.get("ymintilt", -48))
            ymaxtilt = float(fsampling.get("ymaxtilt", 48))
            xmintilt = float(fsampling.get("xmintilt", -60))
            xmaxtilt = float(fsampling.get("xmaxtilt", 60))
            resolved_wedge_mask = get_wedge_mask(
                reference.shape,
                ftype=ftype,
                ymintilt=ymintilt,
                ymaxtilt=ymaxtilt,
                xmintilt=xmintilt,
                xmaxtilt=xmaxtilt,
            ).astype(np.float32)
            resolved_apply_to = _resolve_wedge_apply_to("auto" if wedge_apply_to == "auto" else wedge_apply_to, fsampling)
        except Exception:
            resolved_wedge_mask = wedge_mask
            resolved_apply_to = _resolve_wedge_apply_to(wedge_apply_to, fsampling)

    if use_gpu:
        try:
            return _align_one_particle_torch_gpu(
                particle, reference, mask,
                tdrot_step, cone_step, cone_range, tdrot_range, inplane_step, inplane_range,
                shift_search, lowpass_angstrom, pixel_size, multigrid_levels,
                shift_mode=shift_mode, subpixel=subpixel, cc_mode=cc_mode,
                cc_local_window=cc_local_window, cc_local_eps=cc_local_eps,
                angle_sampling_mode=angle_sampling_mode, old_angles=old_angles,
                wedge_mask=resolved_wedge_mask, wedge_apply_to=resolved_apply_to,
                fsampling=fsampling, fsampling_mode=fsampling_mode,
                subpixel_method=subpixel_method,
                device_id=device_id,
            )
        except RuntimeError:
            pass  # fallback to CPU when CUDA unavailable
    if mask is None:
        mask = np.ones_like(particle, dtype=bool)
    tdrot_lo, tdrot_hi = tdrot_range[0], tdrot_range[1]
    tilt_lo, tilt_hi = cone_range[0], cone_range[1]
    inplane_lo, inplane_hi = inplane_range[0], inplane_range[1]
    p, r = particle, reference
    if lowpass_angstrom and pixel_size > 0:
        p = _lowpass_filter(particle, lowpass_angstrom, pixel_size)
        r = _lowpass_filter(reference, lowpass_angstrom, pixel_size)

    if multigrid_levels <= 1:
        return _align_single_scale(
            p, r, mask, tdrot_step, cone_step, inplane_step, shift_search,
            tdrot_lo=tdrot_lo, tdrot_hi=tdrot_hi,
            tilt_lo=tilt_lo, tilt_hi=tilt_hi, inplane_lo=inplane_lo, inplane_hi=inplane_hi,
            shift_mode=shift_mode,
            subpixel=subpixel,
            cc_mode=cc_mode,
            cc_local_window=cc_local_window,
            cc_local_eps=cc_local_eps,
            angle_sampling_mode=angle_sampling_mode,
            old_angles=old_angles,
            wedge_mask=resolved_wedge_mask,
            wedge_apply_to=resolved_apply_to,
            subpixel_method=subpixel_method,
        )

    # Multigrid: coarse then fine
    factor = 2
    p_coarse = _downsample(p, factor)
    ref_coarse = _downsample(r, factor)
    # downscale mask by taking every factor-th voxel
    mask_coarse = _downsample(mask.astype(np.float32), factor) > 0.5

    coarse_step_c = max(cone_step * 2, 30.0)
    coarse_step_i = max(inplane_step * 2, 30.0)
    tdrot, tilt, narot, dx_c, dy_c, dz_c, _ = _align_single_scale(
        p_coarse, ref_coarse, mask_coarse,
        max(tdrot_step * 2.0, 30.0), coarse_step_c, coarse_step_i, max(1, shift_search // 2),
        tdrot_lo=tdrot_lo, tdrot_hi=tdrot_hi,
        shift_mode=shift_mode,
        subpixel=False,
        cc_mode=cc_mode,
        cc_local_window=cc_local_window,
        cc_local_eps=cc_local_eps,
        angle_sampling_mode=angle_sampling_mode,
        old_angles=old_angles,
        wedge_mask=resolved_wedge_mask,
        wedge_apply_to=resolved_apply_to,
        subpixel_method=subpixel_method,
    )
    # Scale shift to full-res voxels
    shift_center = (int(dx_c * factor), int(dy_c * factor), int(dz_c * factor))

    # Fine level: narrow angular search around coarse best, shift search around coarse shift
    margin = max(cone_step, inplane_step)
    tdrot_lo = (tdrot - margin) % 360
    tdrot_hi = (tdrot + margin + 1) % 360
    if tdrot_lo > tdrot_hi:
        tdrot_lo, tdrot_hi = 0, 360
    tilt_lo = max(0, tilt - margin)
    tilt_hi = min(180, tilt + margin)
    inplane_lo = (narot - margin) % 360
    inplane_hi = (narot + margin + 1) % 360
    if inplane_lo > inplane_hi:
        inplane_lo, inplane_hi = 0, 360
    tdrot, tilt, narot, dx, dy, dz, cc = _align_single_scale(
        p, r, mask,
        tdrot_step, cone_step, inplane_step, shift_search,
        tdrot_lo=tdrot_lo, tdrot_hi=tdrot_hi if tdrot_hi > tdrot_lo else 360,
        tilt_lo=tilt_lo, tilt_hi=tilt_hi,
        inplane_lo=inplane_lo, inplane_hi=inplane_hi if inplane_hi > inplane_lo else 360,
        shift_center=shift_center,
        shift_mode=shift_mode,
        subpixel=subpixel,
        cc_mode=cc_mode,
        cc_local_window=cc_local_window,
        cc_local_eps=cc_local_eps,
        angle_sampling_mode=angle_sampling_mode,
        old_angles=(tdrot, tilt, narot),
        wedge_mask=resolved_wedge_mask,
        wedge_apply_to=resolved_apply_to,
        subpixel_method=subpixel_method,
    )
    # Refine shift at full res (already in full-res voxels from coarse * factor)
    # _align_single_scale returns integer shifts; we keep them
    return (tdrot, tilt, narot, dx, dy, dz, cc)

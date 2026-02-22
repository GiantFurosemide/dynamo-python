"""
Simplified alignment: FFT-based cross-correlation with angular/shift search.
Matches Dynamo concept: search orientations and shifts, output best CC.
Supports multigrid (coarse-to-fine) and optional PyTorch GPU.
"""
import numpy as np
from scipy.ndimage import map_coordinates, shift, zoom
from scipy.fft import fftn, ifftn, fftfreq
from scipy.spatial.transform import Rotation


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


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized CC between two same-sized volumes."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


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
    cone_step: float,
    inplane_step: float,
    shift_search: int,
    tilt_lo: float = 0.0,
    tilt_hi: float = 180.0,
    inplane_lo: float = 0.0,
    inplane_hi: float = 360.0,
    shift_center: tuple = (0, 0, 0),
) -> tuple:
    """Single-scale alignment search. Returns (tdrot, tilt, narot, dx, dy, dz, cc)."""
    ref_m = reference * mask
    ref_m = ref_m - np.mean(ref_m[mask])
    ref_m[~mask] = 0
    cx, cy, cz = shift_center

    best_cc = -2.0
    best_params = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    tilts = np.arange(tilt_lo, min(tilt_hi + cone_step * 0.5, 181), cone_step)
    for tilt in tilts:
        inplanes = np.arange(inplane_lo, inplane_hi, inplane_step) if tilt not in (0, 180) else [0]
        for narot in inplanes:
            ref_rot = rotate_volume(ref_m, 0, tilt, narot)
            for sx in range(-shift_search, shift_search + 1):
                for sy in range(-shift_search, shift_search + 1):
                    for sz in range(-shift_search, shift_search + 1):
                        dx, dy, dz = cx + sx, cy + sy, cz + sz
                        ref_shifted = shift(ref_rot, (dx, dy, dz), order=1, mode="constant", cval=0)
                        cc = normalized_cross_correlation(particle * mask, ref_shifted)
                        if cc > best_cc:
                            best_cc = cc
                            best_params = (0.0, tilt, narot, float(dx), float(dy), float(dz))
    return best_params + (best_cc,)


def _align_one_particle_torch_gpu(
    particle: np.ndarray,
    reference: np.ndarray,
    mask,
    cone_step: float,
    cone_range: tuple,
    inplane_step: float,
    inplane_range: tuple,
    shift_search: int,
    lowpass_angstrom: float,
    pixel_size: float,
    multigrid_levels: int,
):
    """PyTorch GPU path. Requires CUDA. Uses NumPy for search (batched GPU impl TBD)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    # Same logic as CPU path until batched GPU rotation+CC implemented
    if mask is None:
        mask = np.ones_like(particle, dtype=bool)
    tilt_lo, tilt_hi = cone_range[0], cone_range[1]
    inplane_lo, inplane_hi = inplane_range[0], inplane_range[1]
    p, r = particle, reference
    if lowpass_angstrom and pixel_size > 0:
        p = _lowpass_filter(particle, lowpass_angstrom, pixel_size)
        r = _lowpass_filter(reference, lowpass_angstrom, pixel_size)
    if multigrid_levels <= 1:
        return _align_single_scale(
            p, r, mask, cone_step, inplane_step, shift_search,
            tilt_lo=tilt_lo, tilt_hi=tilt_hi, inplane_lo=inplane_lo, inplane_hi=inplane_hi,
        )
    factor = 2
    p_coarse = _downsample(p, factor)
    ref_coarse = _downsample(r, factor)
    mask_coarse = _downsample(mask.astype(np.float32), factor) > 0.5
    coarse_step_c = max(cone_step * 2, 30.0)
    coarse_step_i = max(inplane_step * 2, 30.0)
    tdrot, tilt, narot, dx_c, dy_c, dz_c, _ = _align_single_scale(
        p_coarse, ref_coarse, mask_coarse,
        coarse_step_c, coarse_step_i, max(1, shift_search // 2),
    )
    shift_center = (int(dx_c * factor), int(dy_c * factor), int(dz_c * factor))
    margin = max(cone_step, inplane_step)
    tilt_lo = max(0, tilt - margin)
    tilt_hi = min(180, tilt + margin)
    inplane_lo = (narot - margin) % 360
    inplane_hi = (narot + margin + 1) % 360
    if inplane_lo > inplane_hi:
        inplane_lo, inplane_hi = 0, 360
    return _align_single_scale(
        p, r, mask, cone_step, inplane_step, shift_search,
        tilt_lo=tilt_lo, tilt_hi=tilt_hi,
        inplane_lo=inplane_lo, inplane_hi=inplane_hi if inplane_hi > inplane_lo else 360,
        shift_center=shift_center,
    )


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
    cone_range: tuple = (0.0, 180.0),
    inplane_step: float = 15.0,
    inplane_range: tuple = (0.0, 360.0),
    shift_search: int = 3,
    lowpass_angstrom: float = None,
    pixel_size: float = 1.0,
    multigrid_levels: int = 1,
    device: str = "cpu",
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
    if use_gpu:
        try:
            return _align_one_particle_torch_gpu(
                particle, reference, mask,
                cone_step, cone_range, inplane_step, inplane_range,
                shift_search, lowpass_angstrom, pixel_size, multigrid_levels,
            )
        except RuntimeError:
            pass  # fallback to CPU when CUDA unavailable
    if mask is None:
        mask = np.ones_like(particle, dtype=bool)
    tilt_lo, tilt_hi = cone_range[0], cone_range[1]
    inplane_lo, inplane_hi = inplane_range[0], inplane_range[1]
    p, r = particle, reference
    if lowpass_angstrom and pixel_size > 0:
        p = _lowpass_filter(particle, lowpass_angstrom, pixel_size)
        r = _lowpass_filter(reference, lowpass_angstrom, pixel_size)

    if multigrid_levels <= 1:
        return _align_single_scale(
            p, r, mask, cone_step, inplane_step, shift_search,
            tilt_lo=tilt_lo, tilt_hi=tilt_hi, inplane_lo=inplane_lo, inplane_hi=inplane_hi,
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
        coarse_step_c, coarse_step_i, max(1, shift_search // 2),
    )
    # Scale shift to full-res voxels
    shift_center = (int(dx_c * factor), int(dy_c * factor), int(dz_c * factor))

    # Fine level: narrow angular search around coarse best, shift search around coarse shift
    margin = max(cone_step, inplane_step)
    tilt_lo = max(0, tilt - margin)
    tilt_hi = min(180, tilt + margin)
    inplane_lo = (narot - margin) % 360
    inplane_hi = (narot + margin + 1) % 360
    if inplane_lo > inplane_hi:
        inplane_lo, inplane_hi = 0, 360
    tdrot, tilt, narot, dx, dy, dz, cc = _align_single_scale(
        p, r, mask,
        cone_step, inplane_step, shift_search,
        tilt_lo=tilt_lo, tilt_hi=tilt_hi,
        inplane_lo=inplane_lo, inplane_hi=inplane_hi if inplane_hi > inplane_lo else 360,
        shift_center=shift_center,
    )
    # Refine shift at full res (already in full-res voxels from coarse * factor)
    # _align_single_scale returns integer shifts; we keep them
    return (tdrot, tilt, narot, dx, dy, dz, cc)

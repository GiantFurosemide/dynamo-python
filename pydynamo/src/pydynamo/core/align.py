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
    device_id: int = None,
):
    """PyTorch GPU path. Uses GPU for shift + NCC search."""
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
    device = torch.device(f"cuda:{device_id}") if device_id is not None else torch.device("cuda")

    if multigrid_levels <= 1:
        return _align_single_scale_torch_gpu(
            p, r, mask, cone_step, inplane_step, shift_search,
            tilt_lo=tilt_lo, tilt_hi=tilt_hi, inplane_lo=inplane_lo, inplane_hi=inplane_hi,
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
        coarse_step_c, coarse_step_i, max(1, shift_search // 2),
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
    return _align_single_scale_torch_gpu(
        p, r, mask, cone_step, inplane_step, shift_search,
        tilt_lo=tilt_lo, tilt_hi=tilt_hi,
        inplane_lo=inplane_lo, inplane_hi=inplane_hi if inplane_hi > inplane_lo else 360,
        shift_center=shift_center,
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


def _align_single_scale_torch_gpu(
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
    tilts = np.arange(tilt_lo, min(tilt_hi + cone_step * 0.5, 181), cone_step)
    for tilt in tilts:
        inplanes = np.arange(inplane_lo, inplane_hi, inplane_step) if tilt not in (0, 180) else [0]
        for narot in inplanes:
            ref_t = _rotate_volume_torch_gpu(ref_src_t, 0.0, float(tilt), float(narot))
            for sx in range(-shift_search, shift_search + 1):
                for sy in range(-shift_search, shift_search + 1):
                    for sz in range(-shift_search, shift_search + 1):
                        dx, dy, dz = cx + sx, cy + sy, cz + sz
                        ref_shifted_t = _shift_tensor_zero(ref_t, int(dx), int(dy), int(dz))
                        cc = _ncc_torch(part_t, ref_shifted_t, mask_t)
                        if cc > best_cc:
                            best_cc = cc
                            best_params = (0.0, tilt, narot, float(dx), float(dy), float(dz))

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
    cone_range: tuple = (0.0, 180.0),
    inplane_step: float = 15.0,
    inplane_range: tuple = (0.0, 360.0),
    shift_search: int = 3,
    lowpass_angstrom: float = None,
    pixel_size: float = 1.0,
    multigrid_levels: int = 1,
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
    if use_gpu:
        try:
            return _align_one_particle_torch_gpu(
                particle, reference, mask,
                cone_step, cone_range, inplane_step, inplane_range,
                shift_search, lowpass_angstrom, pixel_size, multigrid_levels, device_id=device_id,
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

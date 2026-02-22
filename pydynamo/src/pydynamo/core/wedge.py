"""
Missing wedge utilities for subtomogram averaging.
Convention follows Dynamo table columns 13-17 (ftype, ymintilt, ymaxtilt, xmintilt, xmaxtilt).
Default: ftype=1 (single tilt around Y), Z=beam, Y=tilt axis → wedge in kZ-kY plane.
https://www.dynamo-em.org/w/index.php?title=Table
"""
import numpy as np
from scipy.fft import fftfreq, fftn, ifftn


def get_wedge_mask(
    shape: tuple,
    ftype: int = 1,
    ymintilt: float = -48,
    ymaxtilt: float = 48,
    xmintilt: float = -60,
    xmaxtilt: float = 60,
) -> np.ndarray:
    """
    Create wedge mask in Fourier space per Dynamo tbl cols 13-17.
    ftype: 0=full, 1=single(tilt about Y), 2=singlex(tilt about X), 3=cone, 4=double.
    Default ftype=1: Z=beam, Y=tilt → angle in kZ-kY plane = arctan2(|ky|, |kz|).
    RETAIN frequencies where angle in [ymintilt, ymaxtilt]; rest 0.
    shape: (nz, ny, nx) as MRC volume; dim0=z, dim1=y, dim2=x.
    """
    nz, ny, nx = shape
    kz = np.fft.fftshift(fftfreq(nz))
    ky = np.fft.fftshift(fftfreq(ny))
    kx = np.fft.fftshift(fftfreq(nx))
    kz3, ky3, kx3 = np.meshgrid(kz, ky, kx, indexing="ij")
    eps = 1e-12

    if ftype == 0:
        return np.ones(shape, dtype=np.float32)
    if ftype == 1:
        # tilt about Y: wedge in kZ-kY plane; angle = arctan2(|ky|, |kz|)
        angle_rad = np.arctan2(np.abs(ky3) + eps, np.abs(kz3) + eps)
        angle_deg = np.degrees(angle_rad)
        mask = ((angle_deg >= ymintilt) & (angle_deg <= ymaxtilt)).astype(np.float32)
        return mask
    if ftype == 2:
        # tilt about X: wedge in kZ-kX plane
        angle_rad = np.arctan2(np.abs(kx3) + eps, np.abs(kz3) + eps)
        angle_deg = np.degrees(angle_rad)
        mask = ((angle_deg >= xmintilt) & (angle_deg <= xmaxtilt)).astype(np.float32)
        return mask
    if ftype == 3:
        # cone: angle from kz in any direction (legacy)
        tilt_rad = np.arctan2(np.sqrt(ky3**2 + kx3**2 + eps), np.abs(kz3) + eps)
        tilt_deg = np.degrees(tilt_rad)
        mask = ((tilt_deg >= ymintilt) & (tilt_deg <= ymaxtilt)).astype(np.float32)
        return mask
    if ftype == 4:
        # double: intersect Y-wedge and X-wedge
        ang_y = np.degrees(np.arctan2(np.abs(ky3) + eps, np.abs(kz3) + eps))
        ang_x = np.degrees(np.arctan2(np.abs(kx3) + eps, np.abs(kz3) + eps))
        mask = (
            ((ang_y >= ymintilt) & (ang_y <= ymaxtilt))
            & ((ang_x >= xmintilt) & (ang_x <= xmaxtilt))
        ).astype(np.float32)
        return mask
    # default ftype 1
    angle_rad = np.arctan2(np.abs(ky3) + eps, np.abs(kz3) + eps)
    angle_deg = np.degrees(angle_rad)
    return ((angle_deg >= ymintilt) & (angle_deg <= ymaxtilt)).astype(np.float32)


def apply_wedge(
    vol: np.ndarray,
    ftype: int = 1,
    ymintilt: float = -48,
    ymaxtilt: float = 48,
    xmintilt: float = -60,
    xmaxtilt: float = 60,
) -> np.ndarray:
    """
    Apply missing wedge mask in Fourier space.
    RETAIN frequencies per Dynamo tbl 13-17; zero out the rest.
    Returns real-space volume.
    """
    mask = get_wedge_mask(vol.shape, ftype, ymintilt, ymaxtilt, xmintilt, xmaxtilt)
    f = np.fft.fftshift(fftn(vol))
    f *= mask
    return np.real(ifftn(np.fft.ifftshift(f))).astype(np.float32)

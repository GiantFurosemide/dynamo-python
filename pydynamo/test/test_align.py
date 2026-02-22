"""Tests for pydynamo alignment."""
import numpy as np
import pytest

from pydynamo.core.align import align_one_particle, _get_device


def test_align_one_particle_cpu():
    """Basic alignment on CPU."""
    p = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    r = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
        p, r, cone_step=45, inplane_step=45, shift_search=1, device="cpu"
    )
    assert -2 <= cc <= 2
    assert 0 <= tilt <= 180
    assert 0 <= narot <= 360


def test_align_with_ranges():
    """Alignment with cone_range and inplane_range."""
    p = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    r = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
        p, r,
        cone_step=45, cone_range=(0, 180),
        inplane_step=45, inplane_range=(0, 360),
        device="cpu",
    )
    assert 0 <= tilt <= 180


@pytest.mark.skip(reason="GPU test implemented; skip by default (run manually when GPU available)")
def test_align_one_particle_gpu():
    """Alignment on GPU when CUDA available. Skip when no GPU."""
    p = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    r = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
        p, r, cone_step=45, inplane_step=45, shift_search=1, device="cuda"
    )
    assert -2 <= cc <= 2

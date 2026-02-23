"""Tests for pydynamo alignment."""
import numpy as np
import pytest
import torch

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_align_one_particle_gpu():
    """Alignment on GPU when CUDA available."""
    p = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    r = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
        p, r, cone_step=45, inplane_step=45, shift_search=1, device="cuda", device_id=0
    )
    assert -2 <= cc <= 2


def test_get_device_auto():
    """auto device resolves to cpu/cuda."""
    d = _get_device("auto")
    assert d in ("cpu", "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_path_does_not_use_cpu_rotate(monkeypatch):
    """GPU path should not call CPU rotate_volume implementation."""
    import pydynamo.core.align as align_mod

    def _raise_cpu_rotate(*_args, **_kwargs):
        raise AssertionError("CPU rotate_volume should not be used on GPU path")

    monkeypatch.setattr(align_mod, "rotate_volume", _raise_cpu_rotate)
    p = np.random.randn(16, 16, 16).astype(np.float32)
    r = np.random.randn(16, 16, 16).astype(np.float32)
    _ = align_mod.align_one_particle(
        p,
        r,
        cone_step=180,
        inplane_step=360,
        shift_search=0,
        multigrid_levels=1,
        device="cuda",
        device_id=0,
    )

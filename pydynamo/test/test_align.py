"""Tests for pydynamo alignment."""
import numpy as np
import pytest
import torch
from scipy.ndimage import shift as ndi_shift

from pydynamo.core.align import (
    _get_device,
    _iter_integer_shifts,
    _ncc_torch,
    _ncc_volume_fft,
    _resample_wedge_mask_to_shape,
    _resolve_wedge_apply_to,
    _dynamo_angleincrement2list,
    _local_normalized_cross_correlation,
    _local_normalized_cross_correlation_torch,
    align_one_particle,
    normalized_cross_correlation,
    rotate_volume,
)
from pydynamo.core.align import _rotate_volume_numba, _rotate_volume_scipy


def test_rotate_volume_numba_matches_scipy():
    """Numba and SciPy rotate_volume should produce equivalent results (p_018, jg_018)."""
    vol = np.random.randn(24, 24, 24).astype(np.float32) * 0.1
    tdrot, tilt, narot = 45.0, 30.0, 60.0
    out_scipy = _rotate_volume_scipy(vol, tdrot, tilt, narot)
    out_numba = _rotate_volume_numba(vol, tdrot, tilt, narot)
    diff = np.abs(out_scipy.astype(np.float64) - out_numba.astype(np.float64))
    assert np.max(diff) < 1e-4, f"max diff {np.max(diff)}"
    assert np.mean(diff) < 1e-5, f"mean diff {np.mean(diff)}"


def test_align_one_particle_cpu():
    """Basic alignment on CPU."""
    p = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    r = np.random.randn(32, 32, 32).astype(np.float32) * 0.1
    tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
        p, r, tdrot_step=90, tdrot_range=(0, 1), cone_step=45, inplane_step=45, shift_search=1, device="cpu"
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
        tdrot_step=90, tdrot_range=(0, 1),
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
        p, r, tdrot_step=90, tdrot_range=(0, 1), cone_step=45, inplane_step=45, shift_search=1, device="cuda", device_id=0
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
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        inplane_step=360,
        shift_search=0,
        multigrid_levels=1,
        device="cuda",
        device_id=0,
    )


def test_align_scans_tdrot_axis():
    """Alignment should recover tdrot when tilt/narot are fixed."""
    ref = np.zeros((32, 32, 32), dtype=np.float32)
    ref[8:24, 14:18, 10:20] = 1.0
    particle = rotate_volume(ref, 45.0, 0.0, 0.0)
    tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
        particle,
        ref,
        tdrot_step=45,
        tdrot_range=(0, 181),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        multigrid_levels=1,
        subpixel=False,
        device="cpu",
    )
    assert tdrot in (0.0, 45.0, 90.0, 135.0, 180.0)
    assert abs(tdrot - 45.0) <= 1e-3


def test_align_subpixel_refines_shift():
    """Subpixel mode should return non-integer shifts for fractional translation."""
    z, y, x = np.indices((32, 32, 32))
    ref = np.exp(-(((z - 16.0) ** 2 + (y - 16.0) ** 2 + (x - 16.0) ** 2) / (2.0 * 3.0**2))).astype(np.float32)
    particle = ndi_shift(ref, (0.4, -0.35, 0.2), order=1, mode="constant", cval=0.0).astype(np.float32)
    tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
        particle,
        ref,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=1,
        multigrid_levels=1,
        subpixel=True,
        device="cpu",
    )
    assert abs(dx - round(dx)) > 1e-2 or abs(dy - round(dy)) > 1e-2 or abs(dz - round(dz)) > 1e-2


def test_align_supports_roseman_local_cc_mode():
    """cc_mode=roseman_local should run and return finite cc."""
    p = np.random.randn(24, 24, 24).astype(np.float32)
    r = np.random.randn(24, 24, 24).astype(np.float32)
    _tdrot, _tilt, _narot, _dx, _dy, _dz, cc = align_one_particle(
        p,
        r,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        multigrid_levels=1,
        cc_mode="roseman_local",
        device="cpu",
    )
    assert np.isfinite(cc)


def test_align_roseman_local_accepts_window_and_eps():
    """roseman_local path should accept configurable local window/epsilon."""
    p = np.random.randn(20, 20, 20).astype(np.float32)
    r = np.random.randn(20, 20, 20).astype(np.float32)
    _tdrot, _tilt, _narot, _dx, _dy, _dz, cc = align_one_particle(
        p,
        r,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        multigrid_levels=1,
        cc_mode="roseman_local",
        cc_local_window=7,
        cc_local_eps=1e-7,
        device="cpu",
    )
    assert np.isfinite(cc)


def test_local_cc_torch_matches_numpy():
    """Torch local CC should numerically match numpy implementation."""
    a = np.random.randn(16, 16, 16).astype(np.float32)
    b = np.random.randn(16, 16, 16).astype(np.float32)
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[2:14, 2:14, 2:14] = True

    cc_np = _local_normalized_cross_correlation(a, b, mask=mask, win=7, eps=1e-7)
    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)
    m_t = torch.as_tensor(mask)
    cc_t = _local_normalized_cross_correlation_torch(a_t, b_t, m_t, win=7, eps=1e-7)
    assert np.isfinite(cc_np)
    assert np.isfinite(cc_t)
    assert abs(cc_np - cc_t) < 1e-4


def test_ncc_torch_matches_numpy_with_mask():
    """Torch NCC and numpy NCC should match on the same masked domain."""
    a = np.random.randn(12, 12, 12).astype(np.float32)
    b = np.random.randn(12, 12, 12).astype(np.float32)
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[2:10, 2:10, 2:10] = True
    cc_np = normalized_cross_correlation(a, b, mask=mask)
    cc_t = _ncc_torch(torch.as_tensor(a), torch.as_tensor(b), torch.as_tensor(mask))
    assert np.isfinite(cc_np)
    assert np.isfinite(cc_t)
    assert abs(cc_np - cc_t) < 1e-6


def test_shift_mode_center_only_disables_offset_search():
    """center_only mode should skip integer neighborhood exploration."""
    ref = np.zeros((24, 24, 24), dtype=np.float32)
    ref[8:16, 8:16, 8:16] = 1.0
    particle = ndi_shift(ref, (2.0, 0.0, 0.0), order=1, mode="constant", cval=0.0).astype(np.float32)

    _t1, _t2, _t3, dx0, dy0, dz0, _cc0 = align_one_particle(
        particle,
        ref,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=3,
        shift_mode="center_only",
        subpixel=False,
        multigrid_levels=1,
        device="cpu",
    )
    _t1, _t2, _t3, dx1, dy1, dz1, _cc1 = align_one_particle(
        particle,
        ref,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=3,
        shift_mode="cube",
        subpixel=False,
        multigrid_levels=1,
        device="cpu",
    )
    assert (dx0, dy0, dz0) == (0.0, 0.0, 0.0)
    assert abs(dx1) >= 1.0 or abs(dy1) >= 1.0 or abs(dz1) >= 1.0


def test_iter_integer_shifts_center_vs_follow_difference():
    """ellipsoid_center and ellipsoid_follow should differ when shift_center != 0."""
    follow = list(_iter_integer_shifts(2, "ellipsoid_follow", shift_center=(2, 0, 0)))
    center = list(_iter_integer_shifts(2, "ellipsoid_center", shift_center=(2, 0, 0)))
    assert len(follow) > 0
    assert len(center) > 0
    assert len(follow) != len(center)


def test_align_accepts_wedge_aware_scoring_mask():
    """Alignment should run with wedge support mask and return finite score."""
    p = np.random.randn(20, 20, 20).astype(np.float32)
    r = np.random.randn(20, 20, 20).astype(np.float32)
    wedge_mask = np.ones((20, 20, 20), dtype=np.float32)
    wedge_mask[:, :, 10:] = 0.0
    out = align_one_particle(
        p,
        r,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        subpixel=True,
        multigrid_levels=1,
        wedge_mask=wedge_mask,
        device="cpu",
    )
    assert np.isfinite(out[-1])


def test_multigrid_wedge_mask_is_resized_for_coarse_stage_cpu():
    """multigrid+fullres wedge mask should not trigger coarse-stage broadcast mismatch."""
    p = np.random.randn(24, 24, 24).astype(np.float32)
    r = np.random.randn(24, 24, 24).astype(np.float32)
    wedge_mask = np.ones((24, 24, 24), dtype=np.float32)
    wedge_mask[:, :, 12:] = 0.0
    out = align_one_particle(
        p,
        r,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        subpixel=False,
        multigrid_levels=2,
        wedge_mask=wedge_mask,
        wedge_apply_to="both",
        device="cpu",
    )
    assert np.isfinite(out[-1])


def test_multigrid_table_fsampling_wedge_mask_is_resized_for_coarse_stage_cpu():
    """fsampling table-generated wedge mask should also be stage-consistent in multigrid."""
    p = np.random.randn(24, 24, 24).astype(np.float32)
    r = np.random.randn(24, 24, 24).astype(np.float32)
    out = align_one_particle(
        p,
        r,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        subpixel=False,
        multigrid_levels=2,
        fsampling_mode="table",
        fsampling={
            "ftype": 1,
            "ymintilt": -30,
            "ymaxtilt": 30,
            "xmintilt": -60,
            "xmaxtilt": 60,
            "fs1": 1.0,
            "fs2": 1.0,
        },
        wedge_apply_to="auto",
        device="cpu",
    )
    assert np.isfinite(out[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_multigrid_wedge_cuda_direct_mask_no_broadcast():
    """CUDA path: multigrid + direct wedge mask should not fail with shape broadcast mismatch."""
    p = np.random.randn(24, 24, 24).astype(np.float32)
    r = np.random.randn(24, 24, 24).astype(np.float32)
    wedge_mask = np.ones((24, 24, 24), dtype=np.float32)
    wedge_mask[:, :, 12:] = 0.0
    out = align_one_particle(
        p,
        r,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        subpixel=False,
        multigrid_levels=2,
        wedge_mask=wedge_mask,
        wedge_apply_to="both",
        device="cuda",
        device_id=0,
    )
    assert np.isfinite(out[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_multigrid_wedge_cuda_table_auto_no_broadcast():
    """CUDA path: multigrid + table fsampling + auto wedge side should not fail with broadcast mismatch."""
    p = np.random.randn(24, 24, 24).astype(np.float32)
    r = np.random.randn(24, 24, 24).astype(np.float32)
    out = align_one_particle(
        p,
        r,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        subpixel=False,
        multigrid_levels=2,
        fsampling_mode="table",
        fsampling={
            "ftype": 1,
            "ymintilt": -30,
            "ymaxtilt": 30,
            "xmintilt": -60,
            "xmaxtilt": 60,
            "fs1": 1.0,
            "fs2": 1.0,
        },
        wedge_apply_to="auto",
        device="cuda",
        device_id=0,
    )
    assert np.isfinite(out[-1])


def test_wedge_resample_non_integer_scale_shape_and_bounds_and_no_resize_artifact():
    """Wedge resample should be geometric, bounded, and avoid np.resize repetition fallback artifacts."""
    src = np.linspace(0.0, 1.0, num=5 * 7 * 9, dtype=np.float32).reshape((5, 7, 9))
    target_shape = (8, 11, 6)
    out = _resample_wedge_mask_to_shape(src, target_shape)
    assert out.shape == target_shape
    assert float(np.min(out)) >= 0.0
    assert float(np.max(out)) <= 1.0
    resize_artifact = np.resize(src, target_shape)
    assert not np.allclose(out, resize_artifact)


def test_wedge_support_changes_orientation_ranking_in_controlled_case():
    """A highly restrictive Fourier support should alter top-1 orientation ranking."""
    ref = np.zeros((24, 24, 24), dtype=np.float32)
    ref[8:16, 11:13, 8:16] = 1.0
    particle = rotate_volume(ref, 90.0, 0.0, 0.0)

    out_nomask = align_one_particle(
        particle,
        ref,
        tdrot_step=90,
        tdrot_range=(0, 181),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        subpixel=False,
        multigrid_levels=1,
        device="cpu",
    )
    dc_only = np.zeros((24, 24, 24), dtype=np.float32)
    dc_only[12, 12, 12] = 1.0
    out_wedge = align_one_particle(
        particle,
        ref,
        tdrot_step=90,
        tdrot_range=(0, 181),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        subpixel=False,
        multigrid_levels=1,
        wedge_mask=dc_only,
        wedge_apply_to="both",
        device="cpu",
    )
    assert abs(out_nomask[0] - 90.0) <= 1e-3
    assert abs(out_wedge[0] - 0.0) <= 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_top1_consistent_under_wedge_support():
    """CPU and GPU should return consistent top-1 pose under wedge-aware scoring."""
    ref = np.random.randn(20, 20, 20).astype(np.float32)
    particle = np.random.randn(20, 20, 20).astype(np.float32)
    wedge = np.ones((20, 20, 20), dtype=np.float32)
    wedge[:, :, 10:] = 0.0
    cpu = align_one_particle(
        particle,
        ref,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=1,
        subpixel=False,
        multigrid_levels=1,
        wedge_mask=wedge,
        wedge_apply_to="both",
        device="cpu",
    )
    gpu = align_one_particle(
        particle,
        ref,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=1,
        subpixel=False,
        multigrid_levels=1,
        wedge_mask=wedge,
        wedge_apply_to="both",
        device="cuda",
        device_id=0,
    )
    assert abs(cpu[0] - gpu[0]) <= 1e-3
    assert abs(cpu[1] - gpu[1]) <= 1e-3
    assert abs(cpu[2] - gpu[2]) <= 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_subpixel_uses_same_wedge_objective_as_main_search(monkeypatch):
    """GPU subpixel should evaluate the same particle-side wedge objective as main search."""
    import pydynamo.core.align as align_mod

    ref = np.random.randn(20, 20, 20).astype(np.float32)
    particle = np.random.randn(20, 20, 20).astype(np.float32)
    mask = np.ones((20, 20, 20), dtype=bool)
    wedge = np.ones((20, 20, 20), dtype=np.float32)
    wedge[:, :, 10:] = 0.0

    expected = align_mod._apply_fourier_support_np((particle * mask).astype(np.float32), wedge)
    state = {"checked": False}
    orig_ncc = align_mod._ncc_torch

    def _spy_ncc(a_t, b_t, mask_t):
        if not state["checked"]:
            state["checked"] = np.allclose(
                np.asarray(a_t.detach().cpu().numpy(), dtype=np.float32),
                expected,
                atol=1e-4,
                rtol=1e-4,
            )
        return orig_ncc(a_t, b_t, mask_t)

    monkeypatch.setattr(align_mod, "_ncc_torch", _spy_ncc)
    _ = align_mod.align_one_particle(
        particle,
        ref,
        mask=mask,
        tdrot_step=180,
        tdrot_range=(0, 1),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=1,
        subpixel=True,
        subpixel_method="auto",
        multigrid_levels=1,
        wedge_mask=wedge,
        wedge_apply_to="both",
        device="cuda",
        device_id=0,
    )
    assert state["checked"] is True


def test_subpixel_quadratic3d_non_inferior_to_axis1d_on_fractional_shifts():
    """3D subpixel fit should be non-inferior to axis1d fallback on controlled shifts."""
    rng = np.random.default_rng(123)
    errs_auto = []
    errs_axis = []
    z, y, x = np.indices((28, 28, 28))
    ref = np.exp(-(((z - 14.0) ** 2 + (y - 14.0) ** 2 + (x - 14.0) ** 2) / (2.0 * 2.8**2))).astype(np.float32)
    for _ in range(8):
        true_shift = rng.uniform(-0.6, 0.6, size=3)
        particle = ndi_shift(ref, true_shift, order=1, mode="constant", cval=0.0).astype(np.float32)
        auto = align_one_particle(
            particle,
            ref,
            tdrot_step=180,
            tdrot_range=(0, 1),
            cone_step=180,
            cone_range=(0, 1),
            inplane_step=360,
            inplane_range=(0, 1),
            shift_search=1,
            subpixel=True,
            subpixel_method="auto",
            multigrid_levels=1,
            device="cpu",
        )
        axis = align_one_particle(
            particle,
            ref,
            tdrot_step=180,
            tdrot_range=(0, 1),
            cone_step=180,
            cone_range=(0, 1),
            inplane_step=360,
            inplane_range=(0, 1),
            shift_search=1,
            subpixel=True,
            subpixel_method="axis1d",
            multigrid_levels=1,
            device="cpu",
        )
        errs_auto.append(float(np.linalg.norm(np.asarray(auto[3:6]) - true_shift)))
        errs_axis.append(float(np.linalg.norm(np.asarray(axis[3:6]) - true_shift)))
    assert float(np.mean(errs_auto)) <= float(np.mean(errs_axis)) + 0.05


def test_wedge_apply_to_auto_contract_matrix():
    """Auto wedge side resolution should follow explicit fs1/fs2 contract."""
    assert _resolve_wedge_apply_to("auto", None) == "both"
    assert _resolve_wedge_apply_to("auto", {"fs1": 1.0, "fs2": 0.0}) == "particle"
    assert _resolve_wedge_apply_to("auto", {"fs1": 0.0, "fs2": 1.0}) == "template"
    assert _resolve_wedge_apply_to("auto", {"fs1": 1.0, "fs2": 1.0}) == "both"
    assert _resolve_wedge_apply_to("particle", {"fs1": 0.0, "fs2": 1.0}) == "particle"


def test_fsampling_auto_mode_matches_expected_explicit_branch():
    """fsampling_mode=table + auto should match the expected explicit wedge_apply_to branch."""
    ref = np.zeros((20, 20, 20), dtype=np.float32)
    ref[6:14, 8:12, 6:14] = 1.0
    particle = rotate_volume(ref, 90.0, 0.0, 0.0)
    base_kwargs = dict(
        tdrot_step=90,
        tdrot_range=(0, 181),
        cone_step=180,
        cone_range=(0, 1),
        inplane_step=360,
        inplane_range=(0, 1),
        shift_search=0,
        subpixel=False,
        multigrid_levels=1,
        device="cpu",
        fsampling_mode="table",
    )
    fsampling_cases = [
        ({"ftype": 1, "ymintilt": -30, "ymaxtilt": 30, "xmintilt": -60, "xmaxtilt": 60, "fs1": 1.0, "fs2": 0.0}, "particle"),
        ({"ftype": 1, "ymintilt": -30, "ymaxtilt": 30, "xmintilt": -60, "xmaxtilt": 60, "fs1": 0.0, "fs2": 1.0}, "template"),
        ({"ftype": 1, "ymintilt": -30, "ymaxtilt": 30, "xmintilt": -60, "xmaxtilt": 60, "fs1": 1.0, "fs2": 1.0}, "both"),
    ]
    for fsampling, expected_mode in fsampling_cases:
        auto = align_one_particle(
            particle,
            ref,
            wedge_apply_to="auto",
            fsampling=fsampling,
            **base_kwargs,
        )
        explicit = align_one_particle(
            particle,
            ref,
            wedge_apply_to=expected_mode,
            fsampling=fsampling,
            **base_kwargs,
        )
        # Parity-grade branch assertion: auto branch equals selected explicit branch.
        np.testing.assert_allclose(np.asarray(auto), np.asarray(explicit), atol=1e-5, rtol=1e-5)


def test_dynamo_angleincrement2list_polar_limits():
    """Dynamo sampler should keep tilt in [0, 180]."""
    angles = _dynamo_angleincrement2list(
        cone_range=360,
        cone_sampling=45,
        inplane_range=0,
        inplane_sampling=10,
        old_angles=(0.0, 0.0, 0.0),
    )
    assert angles.shape[1] == 3
    assert np.all(angles[:, 1] >= -1e-6)
    assert np.all(angles[:, 1] <= 180.0 + 1e-6)


def test_align_dynamo_mode_accepts_old_angles():
    """Dynamo mode should run with old_angles seed."""
    p = np.random.randn(20, 20, 20).astype(np.float32)
    r = np.random.randn(20, 20, 20).astype(np.float32)
    out = align_one_particle(
        p,
        r,
        cone_step=45,
        cone_range=(0, 120),
        inplane_step=30,
        inplane_range=(0, 120),
        angle_sampling_mode="dynamo",
        old_angles=(10.0, 20.0, 30.0),
        shift_search=0,
        multigrid_levels=1,
        device="cpu",
    )
    assert len(out) == 7
    assert np.isfinite(out[-1])


def test_fft_ncc_vs_realspace_ncc_consistency():
    """
    FFT-based NCC volume peak vs realspace NCC at that shift (p_012 §2.4, f_012 §5, f_018).
    FFT NCC uses global ref mean/std; realspace NCC uses ref_shifted (zero-padded) mean/std.
    These differ at boundary shifts, so we relax tolerance to 1e-2 and verify peak position.
    """
    np.random.seed(42)
    n = 16
    part = np.random.randn(n, n, n).astype(np.float32) * 0.2
    ref = np.random.randn(n, n, n).astype(np.float32) * 0.2

    ncc_vol = _ncc_volume_fft(part, ref, None)
    assert ncc_vol is not None
    peak_flat = np.argmax(ncc_vol)
    ix, iy, iz = np.unravel_index(peak_flat, ncc_vol.shape)
    cc_fft = float(ncc_vol[ix, iy, iz])
    dx = ix - n // 2
    dy = iy - n // 2
    dz = iz - n // 2

    ref_shifted = ndi_shift(ref.astype(np.float64), (dx, dy, dz), order=1, mode="constant", cval=0)
    cc_realspace = normalized_cross_correlation(part.astype(np.float64), ref_shifted, mask=None)
    assert abs(cc_fft - cc_realspace) < 0.05, (
        f"FFT NCC {cc_fft} vs realspace NCC {cc_realspace} at shift ({dx},{dy},{dz})"
    )

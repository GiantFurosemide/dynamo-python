# Align Investigation Report (Scoring + Implementation)
# align 纠察报告（打分问题 + 实现问题）

## 1) Problem Statement / 问题定义

- **Observed symptom / 现象**: after `alignment`, reconstructed volume from refined particles is abnormally poor.
- **Focus / 范围**: `align` path only, but include downstream transform usage that directly affects reconstruction quality.
- **Method / 方法**: source-level review of Dynamo vs pydynamo behavior, with emphasis on scoring correctness and transform correctness.

---

## 2) Executive Summary / 结论摘要

Most likely causes are **not only scoring quality**, but also at least one **high-impact implementation bug** in transform inversion used for reconstruction:

1. **Critical implementation risk**: inverse transform composition/order in `core/average.py` is likely incorrect for alignment semantics.
2. **High algorithm gap**: pydynamo align scoring does not replicate Dynamo missing-wedge/fsampling handling.
3. **High algorithm/implementation gap**: CPU/GPU NCC definition differs under mask handling.
4. **Medium gap**: local correlation path is Roseman-like approximation, not full Dynamo-equivalent implementation.
5. **Medium gap**: subpixel strategy is axis-wise 1D parabola (not Dynamo-style peak extraction semantics).

These together can produce seemingly “good CC” but poor reconstruction consistency.

---

## 3) Findings with Code Locations / 发现与代码定位

## F1 (Critical) Inverse transform order is likely wrong

- **Location / 位置**: `pydynamo/src/pydynamo/core/average.py`, `apply_inverse_transform()`.
- **Current behavior / 现实现象**:
  - inverse rotation first,
  - then inverse shift.
- **Risk / 风险**:
  - If alignment model is `particle ≈ Shift(dx) * Rotate(R) * reference`, inverse should apply in reverse operator order.
  - Current order can produce systematic misregistration and severely blur average/reconstruction.
- **Why severe / 严重性说明**:
  - This is a direct geometry consistency issue; even perfect scoring would still reconstruct poorly if inverse mapping is wrong.

## F2 (Critical/High) Inverse Euler handling may be mathematically unsafe

- **Location**: `pydynamo/src/pydynamo/core/average.py`, `apply_inverse_transform()`.
- **Current behavior**:
  - inverse matrix built via `euler_zxz_to_rotation_matrix(-tdrot, -tilt, -narot)`.
- **Risk**:
  - For Euler composition, simple element-wise negation is not generally equivalent to exact inverse in same parameterization ordering.
  - Safe way is to invert rotation object/matrix directly.
- **Impact**:
  - Rotational mismatch can accumulate and strongly degrade final map.

## F3 (High) Missing-wedge / fsampling scoring not replicated in align

- **Dynamo reference**:
  - `dynamo__align_motor.m`: explicit fsampling-type logic + filtermask composition + `use_CC` branches.
- **pydynamo location**:
  - `pydynamo/src/pydynamo/core/align.py` scoring paths do not include equivalent per-particle wedge/fsampling constraints.
- **Risk**:
  - In tomographic data with anisotropic missing information, orientation ranking becomes biased.
- **Impact**:
  - High probability of “reasonable CC but wrong pose” under real data.

## F4 (High) CPU vs GPU NCC semantics are inconsistent

- **Locations**:
  - CPU NCC: `normalized_cross_correlation()` in `core/align.py` (operates on flattened full array after masking-by-zero).
  - GPU NCC: `_ncc_torch()` in `core/align.py` (operates only on mask voxels).
- **Risk**:
  - Same config may produce different scores/choices across CPU and GPU.
- **Impact**:
  - Reproducibility drift; hard-to-debug quality inconsistency.

## F5 (Medium) Local CC is approximation, not full Dynamo-equivalent

- **Locations**:
  - CPU: `_local_normalized_cross_correlation()`
  - GPU: `_local_normalized_cross_correlation_torch()`
- **Risk**:
  - “Roseman-like” local normalization may differ from Dynamo’s exact numeric procedure and peak behavior.
- **Impact**:
  - Ranking differences in difficult low-SNR or masked-edge regions.

## F6 (Medium) Subpixel refinement strategy differs from Dynamo peak extraction style

- **Location**: `_align_single_scale()` in `core/align.py`.
- **Current behavior**:
  - axis-wise 1D parabolic updates for x/y/z.
- **Risk**:
  - Bias under anisotropic or coupled curvature around peak.
- **Impact**:
  - Smaller than F1/F3 but can still hurt final reconstruction sharpness.

## F7 (Medium) No explicit parity check between align-score mask and reconstruction mask semantics

- **Locations**:
  - align command uses `nmask` for scoring,
  - reconstruction applies `nmask` during averaging.
- **Risk**:
  - If scoring/transform and averaging masking are not mathematically aligned, selected poses can underperform in final average.

---

## 4) Why this can look like “scoring issue” / 为什么会表现成“打分问题”

- If inverse transform is wrong (F1/F2), even correct best-pose scoring will reconstruct badly.
- If wedge scoring is absent (F3), score landscape itself is biased.
- If CPU/GPU scoring differs (F4), quality appears unstable across runs/devices.

So the symptom can be a mixed failure of:
- score definition,
- transform inversion,
- and geometry consistency.

---

## 5) Recommended Debug/Validation Plan / 建议验证计划

## T1 Geometry consistency test (must do first)

- Synthetic reference -> apply known `(R, shift)` to generate particle.
- Run alignment, then reconstruct from the recovered pose.
- Verify:
  - recovered pose error,
  - reconstructed map correlation to original reference.
- Purpose:
  - quickly isolate F1/F2 transform issues.

## T2 Scoring parity test with/without wedge

- Same synthetic set with imposed anisotropic Fourier support.
- Compare score ranking stability for candidate orientations.
- Purpose:
  - quantify F3 impact.

## T3 CPU/GPU score consistency probe

- Fixed particle/reference/mask and candidate set.
- Compare top-K candidates and score values CPU vs GPU.
- Purpose:
  - detect F4 reproducibility drift.

## T4 Subpixel sensitivity test

- Controlled fractional shifts (e.g., 0.2/0.4/0.6 voxel).
- Compare recovered shifts and reconstruction quality.
- Purpose:
  - evaluate F6 contribution.

---

## 6) Improvement Recommendations / 改进建议

Priority order for practical recovery:

1. **P0** Fix inverse transform correctness (F1/F2).
2. **P0** Unify CPU/GPU NCC mask-domain definition (F4).
3. **P1** Add wedge/fsampling-aware scoring mode aligned with Dynamo semantics (F3).
4. **P1** Improve local CC contract and document numeric differences (F5).
5. **P2** Upgrade subpixel from sequential 1D to robust 3D local fit + fallback (F6).

---

## 7) Immediate suspect list for your current bad reconstruction / 针对你当前异常的首要怀疑点

Most likely direct contributors:

- **#1** `apply_inverse_transform()` operator order and inverse rotation handling.
- **#2** Missing-wedge scoring mismatch in align on real tomographic data.
- **#3** CPU/GPU scoring inconsistency if mixed execution or environment variation exists.

These should be triaged before tuning search ranges or hyperparameters.

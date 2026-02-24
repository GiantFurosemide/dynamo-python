# Worker-001 Fix Progress & Result (Round A + Round B)

## Progress log

1. Read review artifacts:
   - `align-error-list.md`
   - `align-investigation-report.md`
2. Confirmed high-priority implementation targets:
   - ALN-001, ALN-002, ALN-004
3. Landed code changes in:
   - `pydynamo/src/pydynamo/core/average.py`
   - `pydynamo/src/pydynamo/core/align.py`
   - `pydynamo/test/test_average.py`
   - `pydynamo/test/test_align.py`
4. Executed focused regression:
   - command:
     - `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_average.py pydynamo/test/test_align.py pydynamo/test/test_alignment_command.py pydynamo/test/test_reconstruction_command.py`
   - result:
     - `31 passed`

## Implemented fixes

### ALN-001 (S0): inverse transform composition order

- **Status:** DONE
- **Change:** in `apply_inverse_transform`, switched from:
  - inverse rotation -> inverse shift
  to:
  - inverse shift -> inverse rotation
- **Rationale:** this matches reverse composition of align forward model.

### ALN-002 (S0/S1): unsafe inverse Euler handling

- **Status:** DONE
- **Change:** replaced Euler sign-negation inverse with exact inverse from rotation object:
  - `Rotation.from_euler(...).inv().as_matrix()`
- **Rationale:** avoids non-equivalence risks from simple angle negation in composed Euler chains.

### ALN-004 (S1): CPU/GPU NCC mask-domain mismatch

- **Status:** DONE
- **Change:**
  - CPU `normalized_cross_correlation` now supports `mask` and computes on masked support.
  - `_compute_cc_np(..., cc_mode="ncc")` now uses the same masked domain.
- **Rationale:** align CPU semantics with existing GPU `_ncc_torch` masked behavior.

## New/updated tests

1. `pydynamo/test/test_average.py`
   - `test_apply_inverse_transform_recovers_align_forward_model`
2. `pydynamo/test/test_align.py`
   - `test_ncc_torch_matches_numpy_with_mask`

## Remaining issues from error list

- ALN-003: completed in Round B
- ALN-005: completed in Round B
- ALN-006: completed in Round B
- ALN-007: completed in Round B
- ALN-008: completed in Round B

## Round conclusion

This round addresses the highest-risk implementation correctness issues in transform inversion and cross-device NCC semantics, with regression tests added to prevent recurrence.

---

## Round B progress log

1. Implemented remaining pending items in align/recon/classification command paths.
2. Added config surface for wedge-aware scoring and mask-consistency diagnostics.
3. Added regression tests for:
   - wedge-aware scoring argument propagation,
   - expanded shift-mode semantics,
   - masked-NCC and subpixel behavior safety.
4. Executed full-suite regression:
   - command:
     - `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`
   - result:
     - `59 passed`

## Round B implemented fixes

### ALN-003 (S1): missing-wedge / fsampling scoring parity gap

- **Status:** DONE
- **Changes:**
  - Added wedge-aware scoring support in `core/align.py` via optional `wedge_mask`.
  - Added command-level wedge configuration and mask construction in:
    - `commands/alignment.py`
    - `commands/classification.py`
  - Added tests:
    - `test_align_accepts_wedge_aware_scoring_mask`
    - `test_alignment_command_builds_wedge_mask_for_scoring`
    - `test_classification_builds_wedge_mask_for_scoring`

### ALN-005 (S1/S2): roseman_local approximation ambiguity

- **Status:** DONE
- **Changes:**
  - Added explicit runtime warning on first `roseman_local` use:
    - documented as approximate backend, not strict Dynamo parity.
  - Warning is emitted once to avoid log flooding.

### ALN-006 (S2): subpixel sequential 1D bias risk

- **Status:** DONE
- **Changes:**
  - Added guarded 3D quadratic local fit (`_subpixel_offset_3d_quadratic`).
  - Kept previous 1D parabola as robust fallback when 3D fit is unstable.

### ALN-007 (S2): mask-domain consistency diagnostics missing

- **Status:** DONE
- **Changes:**
  - Added mask coverage diagnostics and threshold warning in:
    - `commands/alignment.py`
    - `commands/reconstruction.py`
    - `commands/classification.py`
  - Added config knob:
    - `mask_consistency_min_fraction` (default `0.01`)

### ALN-008 (S2): shift-mode semantics too narrow

- **Status:** DONE
- **Changes:**
  - Expanded shift search semantics in `_iter_integer_shifts` with:
    - `center_only`
    - `cylinder_z_center`
    - `cylinder_z_follow`
  - Corrected center-vs-follow geometry behavior by making
    `ellipsoid_center` and `ellipsoid_follow` semantically different when `shift_center != 0`.
  - Added tests:
    - `test_shift_mode_center_only_disables_offset_search`
    - `test_iter_integer_shifts_center_vs_follow_difference`

## Updated config surface

- `pydynamo/config/alignment_defaults.yaml`
- `pydynamo/config/classification_defaults.yaml`
- `muwang_test/alignment/alignment_defaults.yaml`
- `muwang_test/classification/classification_defaults.yaml`

Added/updated options:
- `apply_wedge_scoring`, `wedge_ftype`, `wedge_ymin`, `wedge_ymax`, `wedge_xmin`, `wedge_xmax`
- `mask_consistency_min_fraction`
- extended `shift_mode` comments for new modes

## Final status

All items in `align-error-list.md` are now implemented and regression-tested in code:

- ALN-001 ✅
- ALN-002 ✅
- ALN-003 ✅
- ALN-004 ✅
- ALN-005 ✅
- ALN-006 ✅
- ALN-007 ✅
- ALN-008 ✅

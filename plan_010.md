# Plan 010 — Alignment multigrid/wedge shape mismatch full fix

## Background

Observed runtime warnings in alignment:

- `operands could not be broadcast together with shapes (48,48,48) (96,96,96) (48,48,48)`

Root cause is a stage-shape mismatch: multigrid coarse volumes use downsampled shape while wedge mask remains full resolution.

## Scope

1. Fix stage-shape consistency for wedge masks in alignment core.
2. Cover both CPU and GPU multigrid paths.
3. Cover both direct wedge scoring and fsampling table-generated wedge masks.
4. Add command-level diagnostics to improve failure localization.
5. Add regression tests preventing this issue from reappearing.

## Implementation Plan

### 1) Core alignment stage-aware wedge handling

File: `pydynamo/src/pydynamo/core/align.py`

- Add helper to resample wedge masks to stage shape:
  - `_resample_wedge_mask_to_shape(...)`
  - `_get_stage_wedge_mask(...)`
- Add explicit shape guard in `_apply_fourier_support_np(...)`.
- Ensure `_align_single_scale(...)` and `_align_single_scale_torch_gpu(...)` auto-align wedge mask shape with current stage/reference shape.

### 2) Multigrid coarse/fine wiring

File: `pydynamo/src/pydynamo/core/align.py`

- In CPU multigrid path (`align_one_particle(...)`), create and use `wedge_mask_coarse` for coarse stage.
- In GPU multigrid path (`_align_one_particle_torch_gpu(...)`), create and use `wedge_mask_coarse` for coarse stage.
- Keep fine stage on full-resolution wedge mask.

### 3) Command diagnostics

File: `pydynamo/src/pydynamo/commands/alignment.py`

- Log shape context once per run:
  - `ref_shape`, `multigrid_levels`, wedge enabled flag, wedge shape.
- Enrich per-task failure logs with particle path and shape/config context.

### 4) Regression tests

Files:

- `pydynamo/test/test_align.py`
  - Add CPU regression test: multigrid + fullres wedge mask runs without broadcast mismatch.
  - Add CPU regression test: multigrid + `fsampling_mode=table` wedge generation also runs.
- `pydynamo/test/test_alignment_command.py`
  - Add command-level smoke test for `multigrid_levels=2` + `apply_wedge_scoring=true`.

## Validation

1. Focused tests:
   - `pydynamo/test/test_align.py`
   - `pydynamo/test/test_alignment_command.py`
2. Confirm no linter errors in touched files.
3. Record outcomes in `r_010.md`.

## Expected Outcome

- No broadcast mismatch when `multigrid_levels > 1` and wedge scoring is enabled.
- Stable behavior for both wedge config paths (`apply_wedge_scoring` and `fsampling_mode=table`).
- Clearer logs for future triage.

# Worker-001 Fix Progress & Result — Round 2

## 1) Execution summary

Round 2 was executed from audit directives and completed without scope cut:

- D-1 parity completion
- D-2 parity-oriented tests
- D-3 subpixel quality evidence
- D-4 wedge path optimization
- D-5 regression locks

## 2) Status discipline (DONE / PARTIAL / TODO)

| Item | Previous audit status | Round-2 status | Notes |
|---|---|---|---|
| ALN-001 | Confirmed | **DONE** | unchanged, tests green |
| ALN-002 | Confirmed | **DONE** | unchanged, tests green |
| ALN-003 | PARTIAL | **DONE** | table-driven fsampling + side-branch behavior + parity tests |
| ALN-004 | Confirmed | **DONE** | unchanged, tests green |
| ALN-005 | Confirmed (scope-limited) | **DONE** | approximation warning retained and explicit |
| ALN-006 | Implemented but under-evidenced | **DONE** | quantitative non-inferiority evidence added |
| ALN-007 | Confirmed | **DONE** | diagnostics retained |
| ALN-008 | Confirmed implementation | **DONE** | semantics retained + tests kept |

Current TODO: **none** for worker-001 directive set.

## 3) Implemented modifications (code-level)

## D-1 / ALN-003 parity completion

### Core

- File: `pydynamo/src/pydynamo/core/align.py`
- Added/updated:
  - table-driven fsampling support parameters in `align_one_particle(...)`:
    - `fsampling`, `fsampling_mode`
  - explicit side behavior gates:
    - `wedge_apply_to = auto|both|particle|template`
  - helper:
    - `_resolve_wedge_apply_to(...)` for fsampling-informed side branch selection
  - `align_one_particle(...)` now builds per-particle wedge support from fsampling table fields when `fsampling_mode=table`.

### Command integration

- File: `pydynamo/src/pydynamo/commands/alignment.py`
  - Added config wiring:
    - `fsampling_mode`, `wedge_apply_to`, `subpixel_method`
  - Per-row fsampling metadata is extracted from table columns (`ftype`, `ymintilt`, `ymaxtilt`, `xmintilt`, `xmaxtilt`, `fs1`, `fs2`) and passed to align core.

- File: `pydynamo/src/pydynamo/commands/classification.py`
  - Same config and per-row fsampling wiring as alignment command.

## D-3 / ALN-006 quality evidence path

- File: `pydynamo/src/pydynamo/core/align.py`
  - Added `subpixel_method` (`auto|quadratic3d|axis1d`).
  - Keeps guarded 3D quadratic fit first in `auto`, with robust axis1d fallback.

## D-4 wedge scoring optimization

- File: `pydynamo/src/pydynamo/core/align.py`
  - Removed repeated Fourier support filtering in innermost shift loops.
  - Optimization structure:
    - particle-side support precompute once per particle/stage;
    - template-side support precompute once per orientation;
    - shift loop only performs shift + scoring.

## Config surface updates

- `pydynamo/config/alignment_defaults.yaml`
- `pydynamo/config/classification_defaults.yaml`
- `muwang_test/alignment/alignment_defaults.yaml`
- `muwang_test/classification/classification_defaults.yaml`

Added knobs:

- `fsampling_mode: none|table`
- `wedge_apply_to: auto|both|particle|template`
- `subpixel_method: auto|quadratic3d|axis1d`

## 4) Test evidence

## Focused suite

Command:

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_align.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py pydynamo/test/test_average.py pydynamo/test/test_reconstruction_command.py`

Result:

- `47 passed`

## Full suite

Command:

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`

Result:

- `64 passed`

## New/expanded tests (round 2)

- `pydynamo/test/test_align.py`
  - `test_wedge_support_changes_orientation_ranking_in_controlled_case`
  - `test_cpu_gpu_top1_consistent_under_wedge_support` (CUDA gated)
  - `test_subpixel_quadratic3d_non_inferior_to_axis1d_on_fractional_shifts`
- `pydynamo/test/test_alignment_command.py`
  - `test_alignment_tbl_fsampling_mode_passes_table_fsampling`
- `pydynamo/test/test_classification.py`
  - `test_classification_tbl_fsampling_mode_passes_table_fsampling`

## 5) Metric evidence

## ALN-006 quality metric (subpixel)

Command output:

- `subpixel_auto_mean_err 0.014885187006436769`
- `subpixel_axis1d_mean_err 0.01945487111769469`

Interpretation:

- `auto(3D+fallback)` improves mean shift error over pure `axis1d` on controlled fractional-shift cases.

## D-4 runtime sanity metric (wedge optimization)

Command output:

- `time_no_wedge_s 3.811`
- `time_wedge_s 3.9283`
- `ratio_wedge_over_no_wedge 1.0308`

Interpretation:

- wedge-aware path overhead stayed near ~3.1% in this controlled run, consistent with optimization target (no unacceptable blow-up).

## 6) Notes

- `roseman_local` warning remains intentional and explicit:
  - it is approximation-oriented by design, not strict Dynamo numeric parity.

# Worker-001 Fix Progress & Result — Round 5

## 1) Objective

Address audit round-4 open blocker:

- `F-R2-2` (GPU subpixel/wedge objective mismatch)

## 2) Core implementation fix

- File: `pydynamo/src/pydynamo/core/align.py`
- Function: `_align_single_scale_torch_gpu(...)`

### What changed

1. Replaced GPU subpixel objective evaluation path:
   - **Before:** subpixel `_cc_at` used CPU `_compute_cc_np` on numpy-shifted reference.
   - **Now:** subpixel `_cc_at` evaluates entirely on GPU with same backend as main search:
     - interpolation shift via GPU `grid_sample`,
     - score via `_ncc_torch` or `_local_normalized_cross_correlation_torch`.
2. Subpixel objective now uses the same particle-side objective tensor (`part_eval_t`) as main search.

### Why this closes F-R2-2

- Main-search and subpixel stages now optimize the same objective contract under wedge modes on GPU.
- No CPU-side objective mismatch remains in GPU subpixel path.

## 3) Test updates

- File: `pydynamo/test/test_align.py`
- Updated test:
  - `test_gpu_subpixel_uses_same_wedge_objective_as_main_search`
  - switched spy hook from `_compute_cc_np` to `_ncc_torch` to match new GPU-only subpixel scoring path.

## 4) Validation evidence

### Focused

Command:

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_align.py pydynamo/test/test_average.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py pydynamo/test/test_reconstruction_command.py`

Result:

- `50 passed`

### Full suite

Command:

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`

Result:

- `67 passed`

## 5) Updated audit status snapshot

| Item | Round-5 status |
|---|---|
| F-R2-2 (GPU subpixel vs wedge objective consistency) | **DONE** |
| F-R2-3 (test strength) | DONE |
| F-R2-4 (auto contract tests) | DONE |

## 6) Note

- `roseman_local` one-time warning remains intentional design behavior.

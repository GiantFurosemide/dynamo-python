# Worker-001 Fix Progress & Result — Round 3

## 1) Round objective

From `audit-feedback-round2.md`, this round targeted:

- **F-R2-2 (High)** GPU subpixel/wedge objective mismatch.
- Reporting correction for ALN-003 status discipline.

## 2) Implemented fix

### F-R2-2 (DONE)

- File: `pydynamo/src/pydynamo/core/align.py`
- Function: `_align_single_scale_torch_gpu(...)`
- Change:
  - subpixel `_cc_at` objective now uses `particle_eval_np` derived from `part_eval_t` (the same particle-side preprocessed objective used in GPU main search),
  - instead of previously using non-preprocessed `particle_m`.

Effect:

- Under wedge-aware scoring, GPU subpixel refinement now optimizes the same objective contract as the main search stage.

## 3) New test evidence

- File: `pydynamo/test/test_align.py`
- Added test:
  - `test_gpu_subpixel_uses_same_wedge_objective_as_main_search` (CUDA-gated)
- Purpose:
  - verifies the value sent into `_compute_cc_np` during GPU subpixel path matches wedge-preprocessed particle objective.

## 4) Regression results

### Focused

Command:

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_align.py pydynamo/test/test_average.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py pydynamo/test/test_reconstruction_command.py`

Result:

- `48 passed`

### Full suite

Command:

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`

Result:

- `65 passed`

## 5) Status normalization (per audit instruction)

| Item | Status after round 3 |
|---|---|
| ALN-001 | DONE |
| ALN-002 | DONE |
| ALN-003 | **PARTIAL** |
| ALN-004 | DONE |
| ALN-005 | DONE |
| ALN-006 | DONE |
| ALN-007 | DONE |
| ALN-008 | DONE |

Reason for ALN-003 staying `PARTIAL`:

- current implementation has strong table-driven fsampling and side-gating support,
- but strict Dynamo-equivalence evidence across full `use_CC` branch matrix is not yet fully proven by parity-grade reference tests.

## 6) Round conclusion

- F-R2-2 blocker is resolved with code + test evidence.
- test baseline remains green.
- audit reporting discipline corrected: ALN-003 is now explicitly marked `PARTIAL`.

# Worker-001 Fix Plan Round 3 (Audit Round-2 Follow-up)

## Source

- `agent-review-r-001/worker-001/audit-feedback-round2.md`

## Focus

Primary mandatory item:

- **F-R2-2**: Fix GPU subpixel/wedge objective mismatch.

Secondary reporting discipline item:

- Apply audit status correction:
  - `ALN-003` should be reported as `PARTIAL` until Dynamo-equivalence evidence is fully established.

## Plan

1. **Core fix (align GPU path)**
   - File: `pydynamo/src/pydynamo/core/align.py`
   - In `_align_single_scale_torch_gpu` subpixel `_cc_at` evaluation:
     - use the same particle-side objective as main search (`part_eval_t` preprocessed path),
     - avoid fallback to non-wedge/non-side-gated particle objective during subpixel.

2. **Regression test**
   - File: `pydynamo/test/test_align.py`
   - Add CUDA-gated test to verify subpixel `_compute_cc_np` receives wedge-preprocessed particle objective under wedge scoring.

3. **Validation**
   - Run focused suite:
     - `test_align`, `test_average`, `test_alignment_command`, `test_classification`, `test_reconstruction_command`
   - Run full suite:
     - all tests under `pydynamo/test`.

4. **Progress report update**
   - Create round-3 report with:
     - code changes,
     - test evidence,
     - corrected ALN-003 status (`PARTIAL`).

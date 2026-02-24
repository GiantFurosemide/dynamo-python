# Worker-001 Fix Plan Round 2 (Audit-Driven)

## Inputs

- `audit-feedback-summary.md`
- `modification-directives-and-criteria.md`

## Goal

Close all residual directives D-1..D-5 with measurable test/metric evidence and update status reporting discipline (`DONE/PARTIAL/TODO`).

## Execution plan

1. **D-1 / ALN-003 parity completion**
   - Introduce table-driven fsampling path in align core.
   - Add behavior gates for data/template side support usage (`wedge_apply_to`).
   - Keep compatibility with existing global wedge mode.

2. **D-2 parity-oriented tests**
   - Add wedge-constrained ranking-change test.
   - Add CPU/GPU consistency test under wedge-aware scoring.
   - Add command tests proving table fsampling metadata is forwarded.

3. **D-3 subpixel quality proof**
   - Keep 3D quadratic as preferred path, axis1d fallback retained.
   - Add quantitative non-inferiority test.
   - Add explicit metric run for report.

4. **D-4 wedge scoring optimization**
   - Precompute particle-side Fourier support once per particle/stage.
   - Precompute template-side support once per orientation (outside shift inner loop).
   - Add runtime sanity metric.

5. **D-5 regression lock**
   - Ensure ALN-001/002/004 related tests stay green.
   - Run focused + full suite.

## Planned deliverables

- Code updates in:
  - `pydynamo/src/pydynamo/core/align.py`
  - `pydynamo/src/pydynamo/commands/alignment.py`
  - `pydynamo/src/pydynamo/commands/classification.py`
  - config defaults in `pydynamo/config/` and `muwang_test/`
- Tests:
  - `pydynamo/test/test_align.py`
  - `pydynamo/test/test_alignment_command.py`
  - `pydynamo/test/test_classification.py`
- Progress report:
  - `agent-review-r-001/worker-001/fix-progress-result-round2.md`

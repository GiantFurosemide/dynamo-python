# Worker-001 Fix Plan Round 4 (Audit R2 F-R2-3/F-R2-4)

## Source

- `agent-review-r-001/worker-001/audit-feedback-round2.md`

## Round target

Address residual audit requests:

1. **F-R2-3**: strengthen ALN-003 tests beyond argument wiring.
2. **F-R2-4**: add explicit contract tests for `wedge_apply_to=auto`.

## Plan

1. Add contract-matrix test for `_resolve_wedge_apply_to(...)`:
   - verify `auto` behavior under `fs1/fs2` combinations.
2. Add parity-branch test for fsampling path:
   - with `fsampling_mode=table`, assert `wedge_apply_to=auto` yields numerically equivalent results to expected explicit branch (`particle/template/both`).
3. Run:
   - focused: `pydynamo/test/test_align.py`
   - full suite: `pytest -q`
4. Update progress report with:
   - implemented tests,
   - run evidence,
   - remaining status clarity.

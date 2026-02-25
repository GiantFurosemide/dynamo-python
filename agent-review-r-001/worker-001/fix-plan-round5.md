# Worker-001 Fix Plan Round 5 (Audit Round-4 Blocker)

## Trigger

- `agent-review-r-001/worker-001/audit-feedback-round4.md`

## Target

Close the remaining open blocker:

- **F-R2-2**: GPU subpixel vs wedge objective consistency.

## Plan

1. Implement core fix in `pydynamo/src/pydynamo/core/align.py`:
   - remove GPU subpixel CPU-side objective fallback,
   - evaluate GPU subpixel with the same GPU objective/backend used in main search.
2. Adapt/extend regression test in `pydynamo/test/test_align.py`:
   - assert GPU subpixel consumes wedge-preprocessed objective path.
3. Run focused + full suite and write round-5 progress report.

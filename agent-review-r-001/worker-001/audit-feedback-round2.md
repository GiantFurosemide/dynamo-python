# Audit Feedback — Round 2
# 第二轮审计反馈

## Audit verdict / 总体结论

- **Overall:** good progress, major directives are mostly implemented.
- **Status recommendation:** `PARTIAL PASS` (not full PASS yet).

Reason: while tests pass (`47 focused`, `64 full`), there are still parity and implementation consistency gaps that block full sign-off.

---

## Verified as correct / 已确认通过

1. **ALN-001 / ALN-002 core fixes are retained**
   - `apply_inverse_transform` now uses inverse shift then exact inverse rotation object.
   - Regression tests still pass.

2. **ALN-004 mask-domain consistency improved**
   - CPU NCC now supports masked-domain evaluation.
   - CPU/GPU focused tests pass.

3. **Round-2 test additions exist and run**
   - New tests for wedge-related behavior, fsampling argument forwarding, and subpixel method selection are present.
   - Reported test counts are reproducible:
     - focused: `47 passed`
     - full: `64 passed`

4. **D-4 optimization direction is implemented**
   - Particle-side wedge support precomputed outside inner shift loop.
   - Template-side wedge support applied once per orientation before shift loop.

---

## Findings (must address before full sign-off) / 阻塞问题

## F-R2-1 (High): ALN-003 parity completion is still overstated

- Current round-2 report marks ALN-003 as `DONE`.
- Audit judgment: still **PARTIAL** for Dynamo parity.
- Why:
  - implementation introduces fsampling-driven wedge mask + side gating, which is good;
  - but it still does **not** fully replicate Dynamo `use_CC` branch richness and data/template recomputation semantics in `dynamo__align_motor.m`.
- Required action:
  - downgrade status to `PARTIAL` unless equivalence tests against Dynamo behavior are added and pass.

## F-R2-2 (High): GPU subpixel evaluation is inconsistent with wedge scoring path

- Location:
  - `pydynamo/src/pydynamo/core/align.py`, `_align_single_scale_torch_gpu`, subpixel `_cc_at`.
- Problem:
  - main GPU search uses wedge-aware preprocessed tensors (`part_eval_t`, optionally wedge-filtered `ref_t`);
  - but subpixel refinement calls CPU `_compute_cc_np(...)` with `particle_m` and no wedge/application-side context.
- Impact:
  - best shift from subpixel can drift from the objective used in main search when wedge scoring is enabled.
- Required action:
  - make subpixel objective consistent with main scoring contract under wedge modes (both CPU and GPU paths).

## F-R2-3 (Medium): ALN-003 tests mostly prove wiring/relative behavior, not Dynamo-equivalence

- Existing tests verify:
  - wedge can alter ranking in controlled setup,
  - fsampling metadata is forwarded.
- Missing:
  - parity-grade tests against Dynamo reference behaviors across fsampling/use_CC variants.
- Required action:
  - add parity matrix tests with expected branch behavior and ranking outcomes.

## F-R2-4 (Medium): `wedge_apply_to=auto` decision rule lacks formal contract evidence

- `_resolve_wedge_apply_to` currently infers side from `fs1/fs2`.
- This heuristic is not yet demonstrated as Dynamo-equivalent.
- Required action:
  - document mapping contract and add dedicated tests for auto-resolution logic.

---

## Item status table (audit-corrected)

| Item | Worker round-2 claim | Audit status |
|---|---|---|
| ALN-001 | DONE | DONE |
| ALN-002 | DONE | DONE |
| ALN-003 | DONE | **PARTIAL** |
| ALN-004 | DONE | DONE |
| ALN-005 | DONE | DONE (approximation warning retained) |
| ALN-006 | DONE | DONE (implementation + evidence improved) |
| ALN-007 | DONE | DONE |
| ALN-008 | DONE | DONE |

---

## Required next updates / 下一步必须更新

1. Update progress report status:
   - ALN-003 -> `PARTIAL`.
2. Fix GPU subpixel/wedge objective mismatch (`F-R2-2`).
3. Add parity-grade ALN-003 tests (not just forwarding tests).
4. Add explicit contract tests for `wedge_apply_to=auto`.

---

## Sign-off decision / 签核结论

- **Not approved for full closure yet.**
- **Approved for continuation with targeted fixes above.**

# Worker-001 Fix Plan (Round A)

## Scope

- Input review artifacts:
  - `align-error-list.md`
  - `align-investigation-report.md`
- Priority policy: follow listed order and land P0/P1 items first.

## Selected targets for this round

1. **ALN-001 (S0)**  
   Fix inverse-transform composition order in `apply_inverse_transform`.
2. **ALN-002 (S0/S1)**  
   Replace Euler sign-negation inverse with exact rotation-object inverse.
3. **ALN-004 (S1)**  
   Unify CPU/GPU NCC mask-domain semantics.

## Implementation plan

1. `pydynamo/src/pydynamo/core/average.py`
   - Change inverse pipeline to:
     - inverse shift first,
     - exact inverse rotation second.
   - Use `Rotation.from_euler(...).inv()` instead of `(-tdrot, -tilt, -narot)`.

2. `pydynamo/src/pydynamo/core/align.py`
   - Upgrade CPU `normalized_cross_correlation` to accept optional mask and compute on same support domain as GPU path.
   - Route `_compute_cc_np(..., cc_mode="ncc")` through masked NCC.

3. Tests
   - Add geometry consistency regression test for inverse-transform round-trip against align forward model.
   - Add NCC parity regression test for CPU/GPU masked NCC semantics.

## Acceptance checks

- Focused tests pass for `average`, `align`, `alignment_command`, `reconstruction_command`.
- New tests explicitly cover:
  - transform-chain inversion correctness,
  - CPU/GPU masked NCC parity.

## Deferred items (next rounds)

- ALN-003 wedge/fsampling-aware scoring parity.
- ALN-005 local CC parity calibration.
- ALN-006 subpixel strategy upgrade.
- ALN-007 mask-consistency diagnostics.
- ALN-008 shift-mode semantics expansion.

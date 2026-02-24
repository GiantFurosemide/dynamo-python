# Align Error List / align 错误清单

## Severity legend / 严重度

- `S0`: critical (likely primary root cause)
- `S1`: high (major quality risk)
- `S2`: medium (noticeable but secondary)

| ID | Severity | Type | Code Location | Problem | Likely Effect | Suggested Fix Direction |
|---|---|---|---|---|---|---|
| ALN-001 | S0 | implementation | `pydynamo/src/pydynamo/core/average.py::apply_inverse_transform` | Inverse transform applies inverse rotation then inverse shift; operator order likely inconsistent with align forward model | Severe blur / wrong reconstruction even if pose score seems good | Re-derive transform chain and enforce exact inverse composition order |
| ALN-002 | S0/S1 | implementation | `pydynamo/src/pydynamo/core/average.py::apply_inverse_transform` | Euler inverse built by simple negation `(-tdrot,-tilt,-narot)` may not equal exact inverse | Systematic orientation error in back-transform | Use rotation object inverse (`R.inv()`) or inverse matrix directly |
| ALN-003 | S1 | algorithm | `pydynamo/src/pydynamo/core/align.py` vs `dynamo__align_motor.m` | Missing equivalent per-particle wedge/fsampling scoring policy | Wrong orientation ranking in anisotropic missing-data conditions | Add Dynamo-compatible wedge-aware scoring mode |
| ALN-004 | S1 | implementation/algorithm | `core/align.py::normalized_cross_correlation` vs `_ncc_torch` | CPU and GPU NCC use different mask domains | CPU/GPU non-reproducible best pose | Unify NCC definition and test parity across devices |
| ALN-005 | S1/S2 | algorithm | `core/align.py` (`cc_mode=roseman_local`) | Local CC is approximation, not strict Dynamo-equivalent | Score landscape mismatch on hard data | Document mode as approximate + calibrate against Dynamo references |
| ALN-006 | S2 | algorithm | `core/align.py::_align_single_scale` | Subpixel uses sequential 1D parabola per axis | Shift bias around coupled/anisotropic peak | Upgrade to guarded 3D local fit or validated hybrid method |
| ALN-007 | S2 | implementation | `commands/alignment.py` + `commands/reconstruction.py` | No explicit alignment-scoring vs reconstruction-mask consistency audit | Good score but weak final average consistency | Add validation checks and diagnostics for mask-domain consistency |
| ALN-008 | S2 | algorithm | `core/align.py` shift modes | Current ellipsoid modes are simplified relative to Dynamo `area_search_modus` richness | Missed peaks in constrained-search workflows | Expand/align shift restriction semantics with Dynamo modes |

## Recommended fix order / 建议修复顺序

1. `ALN-001`
2. `ALN-002`
3. `ALN-004`
4. `ALN-003`
5. `ALN-005`
6. `ALN-006`
7. `ALN-007`
8. `ALN-008`

## Minimal acceptance checks / 最小验收检查

- Synthetic known-transform reconstruction returns high correlation to source reference.
- CPU and GPU top-1/top-K alignment results are consistent within tolerance.
- Wedge-heavy dataset shows improved orientation stability after wedge-aware scoring.

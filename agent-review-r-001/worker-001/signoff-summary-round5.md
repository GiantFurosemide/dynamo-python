# Worker-001 Sign-off Summary (Round 5)
# worker-001 可签核摘要（Round 5）

## 1) Purpose / 目的

- Provide a sign-off-ready mapping from audit findings to:
  - implemented code locations,
  - regression/validation evidence,
  - current closure status.
- 提供可签核映射：审计问题 -> 代码落点 -> 测试证据 -> 当前闭环状态。

---

## 2) Audit-item Mapping / 审计项映射

| Audit Item | Code Change Location(s) | Evidence | Status |
|---|---|---|---|
| ALN-001 inverse transform chain | `pydynamo/src/pydynamo/core/average.py` | `test_apply_inverse_transform_recovers_align_forward_model` + full suite pass | DONE |
| ALN-002 inverse Euler handling | `pydynamo/src/pydynamo/core/average.py` | same as above (uses `Rotation.inv()`) | DONE |
| ALN-004 CPU/GPU NCC mask-domain | `pydynamo/src/pydynamo/core/align.py` | `test_ncc_torch_matches_numpy_with_mask` + focused/full pass | DONE |
| ALN-005 roseman-local contract clarity | `pydynamo/src/pydynamo/core/align.py` | runtime one-time warning + tests remain green | DONE (approx contract explicit) |
| ALN-006 subpixel robustness | `pydynamo/src/pydynamo/core/align.py` | `test_subpixel_quadratic3d_non_inferior_to_axis1d_on_fractional_shifts` + metric evidence | DONE |
| ALN-007 mask consistency diagnostics | `commands/alignment.py`, `commands/reconstruction.py`, `commands/classification.py` | command-path regressions green | DONE |
| ALN-008 shift-mode semantics | `pydynamo/src/pydynamo/core/align.py` | shift-mode tests (`center_only`, center vs follow) + full pass | DONE |
| ALN-003 wedge/fsampling parity track | `pydynamo/src/pydynamo/core/align.py`, `commands/alignment.py`, `commands/classification.py` | branch/contract tests added; strict Dynamo `use_CC` matrix equivalence not fully proven | PARTIAL |
| F-R2-2 GPU subpixel objective mismatch | `pydynamo/src/pydynamo/core/align.py` | GPU subpixel now uses same GPU objective path as main search; `test_gpu_subpixel_uses_same_wedge_objective_as_main_search` | DONE |
| F-R2-3 test-strength gap | `pydynamo/test/test_align.py` | parity-oriented branch behavior tests added | DONE |
| F-R2-4 auto-contract test gap | `pydynamo/test/test_align.py` | `_resolve_wedge_apply_to` contract matrix tests added | DONE |

---

## 3) Latest Validation Snapshot / 最新验证快照

- Focused regression:
  - `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_align.py pydynamo/test/test_average.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py pydynamo/test/test_reconstruction_command.py`
  - Result: `50 passed`
- Full suite:
  - `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`
  - Result: `67 passed`

---

## 4) Sign-off Recommendation / 签核建议

- **Worker-001 implementation risk closure:** `PASS with one scoped caveat`.
- **Scoped caveat / 保留项：**
  - `ALN-003` remains `PARTIAL` until strict Dynamo-reference parity is demonstrated across a full `use_CC`/fsampling branch matrix with reference-comparison evidence.

In other words:

- Engineering fixes and regressions are stable and auditable.
- Final “full Dynamo parity” sign-off still needs one dedicated parity-benchmark package for ALN-003.

换句话说：

- 工程实现和回归稳定性已满足交付要求；
- 若要“完全 Dynamo 等价签核”，还需补 ALN-003 的对标矩阵证据包。

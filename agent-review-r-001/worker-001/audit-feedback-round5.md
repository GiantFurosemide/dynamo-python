# Audit Feedback — Round 5
# 第五轮审计反馈

## Scope / 审计范围

- `agent-review-r-001/worker-001/fix-plan-round5.md`
- `agent-review-r-001/worker-001/fix-progress-result-round5.md`
- `agent-review-r-001/worker-001/signoff-summary-round5.md`
- Code/test changes in:
  - `pydynamo/src/pydynamo/core/align.py`
  - `pydynamo/test/test_align.py`

## Reproduced validation / 复现实证

- Focused regression command (as reported) reproduced:
  - `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_align.py pydynamo/test/test_average.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py pydynamo/test/test_reconstruction_command.py`
  - Result: `50 passed`
- Full suite command reproduced:
  - `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`
  - Result: `67 passed`

Reported numbers are accurate.

---

## Findings (ordered by severity) / 发现（按严重度）

1. **[Resolved High] F-R2-2 is fixed in implementation, not only tests.**  
   - In `_align_single_scale_torch_gpu(...)`, subpixel `_cc_at` no longer uses CPU `_compute_cc_np` fallback.
   - Subpixel scoring now stays on GPU and uses the same objective contract as main search:
     - same particle-side preprocessed tensor (`part_eval_t`),
     - same backend family (`_ncc_torch` / `_local_normalized_cross_correlation_torch`),
     - GPU interpolation shift via `grid_sample`.
   - This closes the round-4 blocker.

2. **[Resolved Medium] Round-4 residual test gaps remain covered.**  
   - Contract/branch-strength tests introduced previously are present and passing:
     - `_resolve_wedge_apply_to` auto matrix,
     - fsampling auto-mode parity to expected explicit branch.

3. **[Low residual risk] No dedicated regression for subpixel+`roseman_local` branch.**  
   - Implementation handles `roseman_local` in subpixel path, but the targeted spy test mainly verifies NCC path.
   - Not a release blocker for current closure, but recommended for future hardening.

---

## Verdict / 结论

- **Round-5 verdict: PASS**
- **F-R2-2 status: DONE**
- **Worker sign-off recommendation:** `PASS with one scoped caveat` (ALN-003 parity package still PARTIAL, consistent with signoff summary).

---

## Status snapshot / 状态快照

| Item | Status |
|---|---|
| F-R2-2 (GPU subpixel vs wedge objective consistency) | DONE |
| F-R2-3 (test-strength gap) | DONE |
| F-R2-4 (auto-contract test gap) | DONE |
| ALN-003 strict Dynamo parity package | PARTIAL (known scoped caveat) |


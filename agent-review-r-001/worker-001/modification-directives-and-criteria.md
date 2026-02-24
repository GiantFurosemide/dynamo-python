# Modification Directives and Acceptance Criteria (Align)
# 明确修改意见与达标标准（Align）

## 1) Purpose / 目的

- Define **clear required changes** and **measurable acceptance standards** after audit.
- 本文用于把审计结论转化为可直接执行的修复任务与验收门槛。

---

## 2) Scope / 范围

- Focus only on align-related issues from `ALN-001` to `ALN-008`.
- Current priority is to close residual gaps, especially ALN-003 parity completeness.

---

## 3) Mandatory Modification Directives / 必须执行的修改意见

## D-1 (Highest): Reclassify ALN-003 as partial and complete parity path

- **Required change / 修改要求**
  1. Update worker progress statement: ALN-003 is currently `PARTIAL`, not `DONE`.
  2. Implement Dynamo-equivalent scoring branches for missing-wedge/fsampling semantics:
     - table-driven fsampling-type handling (not only one global wedge mask),
     - explicit data-side and template-side branch behavior where applicable,
     - behavior gates compatible with existing `cc_mode` contracts.
- **Target files / 目标文件**
  - `pydynamo/src/pydynamo/core/align.py`
  - `pydynamo/src/pydynamo/commands/alignment.py`
  - `pydynamo/src/pydynamo/commands/classification.py`
  - related config docs/tests.

## D-2: Add parity-oriented wedge/fsampling tests

- **Required change**
  - Add tests that verify **ranking/selection behavior**, not only argument plumbing:
    - controlled synthetic cases with anisotropic Fourier support,
    - expected orientation ranking changes under wedge constraints,
    - CPU/GPU consistency under wedge-aware scoring.
- **Target files**
  - `pydynamo/test/test_align.py`
  - `pydynamo/test/test_alignment_command.py`
  - `pydynamo/test/test_classification.py`

## D-3: Add quality proof for ALN-006 (subpixel 3D fit)

- **Required change**
  - Add quantitative benchmark tests comparing:
    - old fallback path vs 3D quadratic path,
    - impact on shift error and reconstruction correlation.
  - Keep fallback safety and prove no regression on low-SNR samples.
- **Target files**
  - `pydynamo/test/test_align.py`
  - optional benchmark script + report note.

## D-4: Optimize wedge scoring execution cost

- **Required change**
  - Avoid repeated Fourier support recomputation in inner candidate loops when possible.
  - Cache/precompute particle-side transformed support per particle per stage.
- **Target files**
  - `pydynamo/src/pydynamo/core/align.py`

## D-5: Keep ALN-001/002/004 fixes intact with regression locks

- **Required change**
  - Preserve current correctness fixes and strengthen regression tests for:
    - inverse transform chain correctness,
    - exact inverse rotation handling,
    - CPU/GPU masked NCC consistency.

---

## 4) Acceptance Criteria (DoD) / 达标标准

## A. Functional correctness / 功能正确性

1. ALN-001/002/004 tests remain green without tolerance weakening.
2. New ALN-003 parity tests verify expected wedge/fsampling ranking behavior.
3. Align output quality improves on representative wedge-heavy synthetic cases.

## B. Numerical consistency / 数值一致性

1. CPU/GPU top-1 pose match rate meets predefined threshold on test suite.
2. Score ordering under same mask/wedge setup is stable across devices within tolerance.

## C. Reconstruction quality / 重建质量

1. Controlled known-transform set achieves target reconstruction correlation.
2. Subpixel 3D fit path shows non-inferior and preferably improved metrics vs fallback.

## D. Performance sanity / 性能约束

1. Wedge-aware scoring path does not introduce unacceptable runtime blow-up.
2. Any added caching/precompute logic is covered by tests and does not change outputs.

## E. Reporting discipline / 报告纪律

1. Progress report must distinguish:
   - `DONE` (fully verified),
   - `PARTIAL` (implemented but not parity/quality proven),
   - `TODO`.
2. For each closed item, include:
   - code location,
   - test evidence,
   - metric evidence (if quality/performance claim).

---

## 5) Sign-off checklist / 交付签核清单

- [ ] ALN-003 status corrected to PARTIAL before new parity completion.
- [ ] Dynamo-parity wedge/fsampling behavior implemented and tested.
- [ ] Subpixel 3D path quality evidence added.
- [ ] Wedge-scoring optimization merged without behavior regression.
- [ ] Focused pytest suite passes.
- [ ] Summary report updated with DONE/PARTIAL/TODO clarity.

---

## 6) Recommended execution order / 推荐执行顺序

1. D-1 (ALN-003 parity completion)
2. D-2 (parity test suite)
3. D-3 (subpixel quality proof)
4. D-4 (performance optimization)
5. Final regression + report normalization

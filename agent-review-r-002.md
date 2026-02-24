# Next-Round Plan: Algorithm Improvement Roadmap (R-002)
# 下一轮规划：算法改进路线（R-002）

## 1) Goal / 目标

- **EN:** Define a prioritized, testable algorithm-improvement roadmap after system-map and algorithm-detail baseline is established for Dynamo and pydynamo.
- **中文：** 在完成 Dynamo 与 pydynamo 的 system map 与算法细节基线后，制定可优先级化、可验证的改进路线。

## 2) Inputs used / 依据来源

1. Project-specific behavior from `dynamo-src` and `pydynamo` algorithm paths.
2. Established alignment practices (local normalized CC, angular search/refinement, subpixel registration).
3. Web-search signals on:
   - adaptive/angular sampling in cryoET template matching,
   - robust subpixel peak fitting and phase-correlation refinement,
   - GPU-accelerated 3D template matching pipelines.

## 3) Updated Improvement Directions / 更新后的改进方向

## A. Compatibility-First Execution Mode / 兼容优先执行模式

- **What / 做什么**
  - Build a strict `dynamo_compat_mode` contract for:
    - angle list generation,
    - shift restriction semantics,
    - mask/fmask scoring domain,
    - tie-break policy on equal peaks.
- **Why / 为什么**
  - Current pydynamo is close in many places but still has approximation zones; explicit compatibility mode prevents accidental drift during optimizations.
- **Success metric / 验收指标**
  - On fixed synthetic suite, compatibility mode reduces Euler/shift mismatch vs reference baseline to target thresholds.

## B. Hybrid Search: Discrete + Continuous Refinement / 混合搜索（离散 + 连续细化）

- **What**
  - Keep coarse discrete orientation scan, then refine around top-K candidates using continuous local optimization (gradient-aware or derivative-free local minimizer).
- **Why**
  - Literature indicates gradient-guided refinement can improve precision and runtime compared with purely exhaustive fine-grid scans.
- **Success metric**
  - Better final CC and lower angular error at equal or lower compute budget.

## C. Subpixel Upgrade from 1D to 3D Local Model / 亚像素从 1D 升级到 3D 局部模型

- **What**
  - Replace or augment axis-wise 1D parabola with robust 3D local quadratic fitting near peak.
  - Add fallback when Hessian/peak validity checks fail.
- **Why**
  - 1D sequential updates can be biased under anisotropic CC landscapes.
  - Subpixel literature warns naive quadratic fitting can fail without validity guards.
- **Success metric**
  - Reduced shift bias on controlled translations, lower peak-locking artifacts.

## D. Missing-Wedge-Aware Scoring Contract / 缺失楔形感知评分契约

- **What**
  - Explicitly separate and standardize:
    - data-side wedge support,
    - template-side wedge support,
    - local/global normalization domains.
  - Add clearly documented scoring equations for each mode.
- **Why**
  - Dynamo uses richer wedge/fsampling pathways; parity and reproducibility depend on exact scoring domain definitions.
- **Success metric**
  - Predictable CC scale and stable ranking under wedge variation tests.

## E. Adaptive Angular Scheduling / 自适应角度调度

- **What**
  - Introduce budgeted sampling strategy:
    - broad initial scan,
    - adaptive local densification where correlation landscape is promising.
- **Why**
  - Extensive uniform sampling is accurate but expensive; adaptive allocation improves efficiency.
- **Success metric**
  - Same-or-better best-score/pose with fewer evaluated orientations.

## F. GPU Throughput Optimization / GPU 吞吐优化

- **What**
  - Optimize high-cost kernels:
    - batched rotation/shift evaluation,
    - candidate chunking,
    - minimized host-device synchronization.
  - Optionally add FFT-centered template-matching kernels for large search windows.
- **Why**
  - Web and prior cryoET tools show large gains from batched/FFT/GPU-optimized template matching.
- **Success metric**
  - Significant speedup on representative workloads without parity regression.

## G. Confidence and Uncertainty Output / 置信度与不确定性输出

- **What**
  - Report not only best candidate but also:
    - top-K margin,
    - ambiguity score,
    - optional entropy-like orientation uncertainty.
- **Why**
  - Helps classification/MRA avoid overconfident reassignment in flat/noisy score landscapes.
- **Success metric**
  - Better robustness in multi-reference assignment stability tests.

## H. Benchmark Protocol Hardening / 基准流程硬化

- **What**
  - Standard benchmark spec:
    - fixed seeds,
    - synthetic + real subsets,
    - CPU/GPU paired runs,
    - reproducibility pack (config + summary + hash).
- **Why**
  - Improvement claims are unreliable without strict benchmark discipline.
- **Success metric**
  - Repeatable results across reruns and environments.

## 4) Prioritized execution (proposal) / 优先级执行建议

- **P0 (must first)**
  - A Compatibility-First mode
  - D Missing-wedge-aware scoring contract
  - H Benchmark protocol hardening
- **P1**
  - C Subpixel 3D upgrade
  - B Hybrid discrete+continuous refinement
- **P2**
  - E Adaptive angular scheduling
  - F GPU throughput optimization
- **P3**
  - G Confidence/uncertainty outputs (useful but not blocking parity)

## 5) Risks and mitigations / 风险与缓解

- **Risk:** Faster methods change ranking behavior unexpectedly.  
  **Mitigation:** compatibility mode + dual-run validation.
- **Risk:** Local optimizers become unstable at low SNR.  
  **Mitigation:** guarded fallback to discrete best.
- **Risk:** GPU optimization introduces subtle numeric divergence.  
  **Mitigation:** per-stage CPU/GPU tolerance tests and lockstep probes.

## 6) Definition of Done for R-002 planning / R-002 规划完成标准

- Each direction has:
  - objective,
  - expected gain,
  - measurable metrics,
  - validation method,
  - rollout priority.
- Roadmap can be directly converted into implementation tasks without re-clarifying intent.

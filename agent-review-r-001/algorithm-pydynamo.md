# pydynamo Algorithm Details / pydynamo 算法细节

## 1) Alignment algorithm skeleton / 配准主流程

Core implementation: `pydynamo/src/pydynamo/core/align.py`

1. Parse alignment params (steps/ranges/mask/multigrid/device/correlation mode).
2. Build angle candidates:
   - legacy grid mode, or
   - Dynamo-like aperture/seed mode via `_dynamo_angleincrement2list(...)`.
3. For each candidate Euler:
   - rotate reference volume,
   - evaluate shift candidates under configured shift mode,
   - compute score using selected CC backend.
4. Keep best `(tdrot, tilt, narot, dx, dy, dz, cc)`.
5. Optional subpixel refinement around best shift.
6. Optional multigrid:
   - coarse downsampled pass,
   - fine re-search around coarse best.

## 2) Angle sampling semantics / 角度采样语义

Two modes in code:

- `legacy`:
  - independent loops over tdrot/tilt/inplane ranges.
- `dynamo`:
  - `_normalize_aperture` to derive aperture semantics.
  - `_dynamo_angleincrement2list`:
    - cone aperture around old orientation axis,
    - inplane symmetric scan around narot seed,
    - matrix deduplication tolerance against near-equivalent rotations.

Current command defaults set `angle_sampling_mode` to `"dynamo"` in alignment/classification.

## 3) Correlation backends / 相关性后端

- `ncc`:
  - masked/global normalized cross correlation (CPU and torch path variants).
- `roseman_local`:
  - local normalized correlation approximation:
    - CPU: `_local_normalized_cross_correlation` with uniform local stats.
    - GPU: `_local_normalized_cross_correlation_torch` with `avg_pool3d`.
- Configurable controls:
  - `cc_local_window`
  - `cc_local_eps`

## 4) Shift search and refinement / 位移搜索与细化

- Integer shift candidates from `_iter_integer_shifts(...)`:
  - `cube`
  - `ellipsoid_center` / `ellipsoid_follow` (same geometric filter in iterator layer).
- Subpixel refinement:
  - axis-wise 1D parabolic offset (`_parabolic_subpixel_offset`) around integer best.
  - applied sequentially on x/y/z axes.

## 5) CPU / GPU execution behavior / CPU/GPU 行为

- CPU: scipy-based rotation/shift/interpolation pipeline.
- GPU: torch tensor path for rotate + shift + score, with CPU fallback.
- GPU path is functionally aligned to CPU search logic, including dynamo sampling mode and cc_mode branching.

## 6) Alignment command-level behavior / alignment 命令层行为

`commands/alignment.py`:

- Supports star/tbl input and resolves particle file paths.
- Seeds each particle scan with table Euler (`old_angles`) when available.
- Supports multi-GPU task scheduling (thread pool + device round-robin).
- Produces refined table/star plus optional immediate reconstructed average.
- Uses runtime progress timing (`elapsed`, `eta`, `eta_at`) and rss observability.

## 7) Classification (MRA-like) behavior / classification（类 MRA）行为

`commands/classification.py`:

- Iterative reference competition over particles.
- For each particle, aligns against one/all references depending on `swap`.
- Selects best reference by highest CC.
- Writes per-reference refined tables and updates reference averages each iteration.
- Supports checkpoint/resume.

## 8) Reconstruction behavior / 重建行为

`commands/reconstruction.py` + `core/average.py`:

- Reads Euler+shift from tbl/star.
- Applies inverse rigid transform per particle.
- Optional real-space mask and wedge masking.
- Optional Fourier compensation (`fcompensate`) path for wedge accumulation.
- Final symmetry through `apply_symmetry`.

## 9) Crop behavior / 裁剪行为

`core/crop.py` + `commands/crop.py`:

- Implements Dynamo-like crop behavior with 1-based position interpretation and fill modes `-2/-1/0/1`.
- Command side groups tasks by tomogram to reduce repeated I/O.
- Outputs RELION-style star and cropped subtomograms.

## 10) Runtime and observability / 运行时与可观测性

`runtime.py` provides:

- unified log/error plumbing,
- progress iterator integration,
- timing text with elapsed + ETA + ETA timestamp,
- memory metrics (`rss_cur`, `rss_avg`, `rss_peak`).

## 11) Observed algorithm characteristics / 观察到的算法特点

- Explicitly configurable and test-friendly algorithm knobs.
- Practical CPU/GPU parity intent in alignment core.
- Command orchestration is compact and easier to reason about than Dynamo DB-driven runtime.
- Some semantic choices remain approximation-oriented relative to full Dynamo internals (to be scored in comparison stage).

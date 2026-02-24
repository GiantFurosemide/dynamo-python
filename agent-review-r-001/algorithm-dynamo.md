# Dynamo Algorithm Details / Dynamo 算法细节

## 1) Alignment algorithm skeleton / 配准主流程

Primary execution chain:

1. `pipeline_align_one_particle.m` reads card + project context and particle/reference/mask assets.
2. Builds angle list using `dynamo_angleset(...)` (internally `dynamo_angleincrement2list(...)`).
3. Calls `dynamo__align_motor(...)` for exhaustive angle scan and shift-peak extraction.
4. Runs optional refinement levels by updating cone/inplane sampling.
5. Writes refined particle row (`refined_ptable`) and carries table metadata columns.

## 2) Angle sampling semantics / 角度采样语义

From `dynamo_angleincrement2list.m` + `dynamo_angleset.m`:

- `cone_range` is treated as **aperture**, effectively using `aperture_half = aperture/2`.
- `list_tilt_north` is sampled from `0` to `aperture_half` by `cone_sampling`.
- At each tilt ring, tdrot spacing is adapted from spherical geometry:
  - interval derived from `acos((cos(interval)-c^2)/s^2)`.
- Orientation is composed around `old_angles` (ZXZ convention), not global origin-only scan.
- Inplane scan is symmetric around a `narot_seed`:
  - positive and negative narot lists around seed.
- Near-duplicate orientations are rejected by matrix-difference tolerance.
- Optional flips (`cone_flip`, `inplane_flip`) duplicate transformed orientation set.

## 3) Correlation and scoring / 相关性与评分

Inside `dynamo__align_motor.m`:

- Two main CC paths:
  - local normalized correlation (`localnc` true, Roseman-style).
  - global correlation (`localnc` false).
- Local path uses moving mask statistics and FFT precomputes of particle and squared particle.
- CC peak extraction uses subpixel peak tools (`dynamo_peak_subpixel`).
- Correlation values over 1 are clipped/treated as interpolation artifacts in guard code.

## 4) Shift search constraints / 位移搜索约束

`area_search` + `area_search_modus` drive restricted shift domains:

- unrestricted mode (`modus=0`) uses global peak.
- center-anchored ellipsoid/cylinder modes.
- follow modes where search region moves with previous shifts/orientation.
- special center-only mode (single voxel).

This is not merely a cube-radius scan; it can be orientation-coupled in asymmetric cases.

## 5) Mask and wedge policy / mask 与 missing-wedge 策略

- Particle-side fsampling is read from table columns (types 0..7).
- Depending on `use_CC` path, template and/or particle Fourier support is constrained.
- Bandpass shell is combined with fsampling map for consistent spectral masking.
- Template-side transformed mask/fmask are used through per-angle rotation/filter steps.

## 6) Coarse-to-fine refinement / 多层细化

`pipeline_align_one_particle.m` refinement loop:

- Coarsest alignment is computed first.
- `dynamo__multigrid_convention(...)` shrinks ranges and sampling for next level.
- New angle list is re-centered on previous best Euler output.
- Alignment motor reruns at each refinement level.

## 7) Output semantics / 输出语义

- Best shifts go to columns 4..6.
- Best Euler ZXZ goes to columns 7..9.
- Best correlation goes to column 10.
- Non-alignment metadata columns are preserved from input table where applicable.

## 8) Crop algorithm behavior / 裁剪算法行为

From `dynamo_crop.m` / `dynamo_crop3d`:

- Center uses 1-based voxel convention, floored position.
- Fill behavior:
  - `-1`: shrink + warning
  - `-2`: shrink without warning
  - `0`: fixed-size output with zero-padding
  - `1`: empty output when out-of-scope
- Out-of-scope detection based on size mismatch versus requested sidelength.

## 9) Reconstruction / averaging behavior / 重建平均行为

`dynamo_average.m` family supports:

- transform-aware averaging from tables/models/metadata.
- optional Fourier compensation (`fcompensate`) and fmask controls.
- tag selection/thresholding/restrictor operators.
- practical pipeline usage is orchestrated from `dynamo_iteration_assemble.m`.

## 10) MRA post-processing behavior / MRA 后处理行为

From `dynamo_plugin_post_multireference_tutorial.m`:

- Collects active refined tables per reference.
- Merges tags across references.
- For each tag, picks reference with best score.
- Writes reassigned tables and drives next per-reference average update.

## 11) Observed algorithm characteristics / 观察到的算法特点

- Strong dependence on table-coded geometry metadata.
- Tight coupling between angular sampling and old orientation.
- Rich shift-constraint modes beyond isotropic cube search.
- Mature MRA runtime semantics via database artifacts and plugin flow.

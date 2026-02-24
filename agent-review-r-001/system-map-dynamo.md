# Dynamo System Map (Algorithm Scope) / Dynamo 系统地图（算法范围）

## 1) Entry pipeline / 主执行链

- Main project execution:
  - `dynamo_execute_project.m`
  - Iteration lifecycle: setup -> compute -> assemble.
- Per-particle alignment core:
  - `+dpkproject/pipeline_align_one_particle.m`
  - Alignment motor: `dynamo__align_motor.m`
- Assembly and averaging:
  - `dynamo_iteration_assemble.m`
  - Average generation via `dynamo_average` paths.
- Multireference post-plugin (MRA):
  - `dynamo_plugin_post_multireference_tutorial.m`

## 2) Algorithm subsystems / 算法子系统

- **Angular sampling**
  - `dynamo_angleset.m`
  - `dynamo_angleincrement2list.m`
  - Inputs: `cone_range`, `cone_sampling`, `inplane_range`, `inplane_sampling`, `old_angles`, flips.
- **Alignment correlation & shift**
  - `dynamo__align_motor.m`
  - Local normalized CC (Roseman-style path) and global correlation path.
  - Shift restriction by `area_search` and `area_search_modus` (center/follow/cylinder/single-point behaviors).
- **Pre/post processing**
  - particle normalization, filter mask composition, missing wedge/fsampling handling in alignment flow.
- **Averaging / reconstruction**
  - `dynamo_average.m` family.
  - Uses table transforms, optional Fourier compensation and fmask policies.
- **Cropping**
  - `dynamo_crop.m` (`dynamo_crop3d` / `dynamo_crop2d`).
  - Fill policies: `-2/-1/0/1`.

## 3) Core data contracts / 核心数据契约

- Project runtime structs:
  - `vpr` (virtual project).
  - `scard` (iteration/ref card).
- Particle metadata:
  - Dynamo table columns (tag/alignment/shifts/eulers/cc/wedge fields/reference).
- Runtime file DB model:
  - `dynamo_database_locate` and typed assets (`starting_ptable`, `refined_ptable`, `refined_table`, `average`, masks/logs).

## 4) Runtime execution modes / 运行模式

- CPU paths in MATLAB runtime.
- GPU paths through motor checks and external engine dispatch (project destination modes).
- Parallel execution with processor tables / per-task orchestration.

## 5) High-level data flow / 高层数据流

1. Read card + project context.
2. Resolve particle/table/reference/masks.
3. Build sampled Euler list around prior orientation.
4. For each sampled angle, rotate template/mask and compute CC volume.
5. Apply optional shift restrictions and subpixel peak extraction.
6. Emit refined per-particle row.
7. Assemble per-reference tables and generate averages.
8. In multireference mode, plugin reassigns particles by best correlation and updates averages/references.

## 6) Scope notes / 范围说明

- This map is restricted to algorithm behavior relevant to parity with pydynamo.
- GUI and unrelated package/tooling branches are intentionally omitted.

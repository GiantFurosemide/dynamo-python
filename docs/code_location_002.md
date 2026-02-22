# Code Location — Target Functions for pydynamo

**Purpose:** Map Dynamo MATLAB source to the four pydynamo commands.

---

## 1. Particle Crop

| Role | File | Summary |
|------|------|---------|
| Main crop | `matlab/src/dynamo_crop.m` | `[subvolume,report]=dynamo_crop(volume,sidelength,position,fillOption)`. Crops 3D cube; fillOption: -1=shrink+warning, 0=zeros, 1=empty. |
| 3D crop | `dynamo_crop3d` (same file) | Internal 3D impl. |
| 2D crop | `dynamo_crop2d` (same file) | 2D variant. |
| Table-driven crop | `matlab/src/dynamo_table_place.m` | Uses `dynamo_crop(target_volume,[lx,ly,lz],r,fill_option)` for placing. |
| Data crop | `+dpkdata/+crop` | Package-level crop utilities. |
| I/O | `dynamo_read`, `dynamo_read_subvolume` | Read volumes. tbl/vll via `dynamo_database_*`, `dynamo_read`. |

**Tbl columns used:** 20 (tomo), 24–26 (x, y, z).

---

## 2. Reconstruction (Average)

| Role | File | Summary |
|------|------|---------|
| Main average | `matlab/src/dynamo_average.m` | ~3146 lines. Core averaging with table, tags, fmask, nmask, fcompensate, symmetry. |
| Wedge | `matlab/src/dynamo_applywedgemap.m` | Applies wedge mask. |
| Pipeline | `+dpkproject/pipeline.genericInput2Average` | Project-level averaging. |
| Fast | `dynamo_average_fast.m` | Optimized path. |
| GUI | `dynamo_average_GUI.m` | Wrapper for GUI. |

**Table columns used:** 4–6 (dx,dy,dz), 7–9 (tdrot,tilt,narot), 20 (tomo), tags from col 1.

---

## 3. Alignment

| Role | File | Summary |
|------|------|---------|
| Per-particle align | `matlab/src/+dpkproject/pipeline_align_one_particle.m` | Main alignment: load particle, ref, mask, wedge; search orientations/shifts; output refined row. |
| Iteration compute | `matlab/src/dynamo_iteration_compute.m` | Loops (ref,tag), calls pipeline_align_one_particle or GPU binary. |
| CC / search | `+dpkvol/+align`, `+dpkvol/+aux/+crossCorrelation` | Correlation, nativeCubeRoseman, subpixel. |
| GPU | `dpkproject.runtime.getGPUmotor`, `dynamo_compute_iteration_spp` | External binary for GPU alignment. |
| C++ | `C/src/classes/volume_float*.cpp` | Rotation, shift. |

---

## 4. Classification (MRA)

| Role | File | Summary |
|------|------|---------|
| MRA plugin | `matlab/src/dynamo_plugin_post_multireference_tutorial.m` | Best-ref by CC, re-average per ref, update tables. |
| Embedded MRA | `matlab/src/+dpkproject/pipeline_embedded_multirerefence.m` | Injects MRA into execution when mra_r*==1. |
| Setup | `matlab/src/dynamo_iteration_setup.m` | assign_master, assign_proc; MRA swap → same tag in all refs. |
| Assemble | `matlab/src/dynamo_iteration_assemble.m` | Gather refined tables, average, threshold. |
| Execute | `matlab/src/dynamo_execute_project.m` | Orchestrates setup→compute→assemble→MRA→check. |
| Deps | `dynamo_average`, `dpkproject.recomputeCC`, `setSurvivingReferences` | |

---

## 5. I/O and Conventions (TomoPANDA-pick)

| Need | File (TomoPANDA-pick) |
|------|------------------------|
| tbl↔star | `utils/tbl2star.py`, `utils/io_dynamo.py` |
| tbl read/write | `io_dynamo.read_dynamo_tbl`, `create_dynamo_table` |
| vll read | `io_dynamo.read_vll_to_df` |
| Euler convert | `io_eular.convert_euler` (ZYZ↔ZXZ) |
| dynamo_tbl_vll_to_relion_star | `io_dynamo.dynamo_tbl_vll_to_relion_star` |
| relion_star_to_dynamo_tbl | `io_dynamo.relion_star_to_dynamo_tbl` |

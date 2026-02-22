# Dynamo MATLAB Source — System Map

**Generated from scan of** `dynamo-src/`  
**Focus:** particle crop, reconstruction (average), alignment, MRA (Multireference Analysis)

---

## 1) Entry Points

| Entry | Location | Description |
|-------|----------|-------------|
| **dynamo.m** | `matlab/src/dynamo.m` | Main CLI. Dispatches on `varargin`: no args → `dynamo__start()` menu; `console`/`x`/`X` → `dynamo_console` or `dpkio.console.GUI`; `project_manager`, `gallery`, `average`, `align`, etc. → specific GUIs; `:` in first arg → `dynamo_db`. Deployed: `dynamo "command"` runs via `evalc()`. |
| **dynamo_execute_project.m** | `matlab/src/dynamo_execute_project.m` | Executes a project. Loads vpr via `dynamo_vpr_load`, loops iterations, calls `dynamo_iteration_setup` → `compute` → `assemble`; triggers MRA plugin and post-processing when configured. Supports `matlab`, `standalone`, `matlab_parfor`, `matlab_gpu`, `standalone_gpu`. |
| **dynamo_project_manager** | Via `dynamo project_manager` | GUI for designing/editing projects. |
| **dynamo_db** | Via `dynamo project:element` | Database query CLI. |
| **dynamo_console** | Via `dynamo console` / `dynamo x` | Interactive console for commands. |
| **MacOS bundle** | `MacOS/dynamo_execute_project.app/` | MCR-compiled standalone. |

---

## 2) Module / Package Graph (by Domain)

### Core pipeline / project execution
- **dpkproject** — project execution: `pipeline_align_one_particle`, `pipeline.genericInput2Average`, `pipeline.getSurvivingReferences`, `pipeline.setSurvivingReferences`, `runtime.getGPUmotor`, `runtime.checks.gpu`, `pipeline.tables.gatherGPUforIteRef`
- **dpksta** — project management: `@Project`, `+projects/+iterations/@SingleIteration` (assemble, referenceRectification), `+projects/+rounds/@SingleRound`, `+projects/+tasks/@Task` (alignAndAverage, assembleTask), `+projects/+database/@Database`, `@RuntimeManager`
- **dpkmulticore** — parfor/workers: `singleTask.m`, `run.m`

### Data / I/O
- **dpkdata** — `montage`, `+containers/@ParticleListFile`, `+aux/+gui/@FusedMontage`, `+crop`
- **dpkstar** — `@Block`, `+aux/+datanames` (dnTomogram, dnTiltSeries, dnParticle), `read`, `+gates`
- **dpktbl** — `@StarTable`, `+star/@FullTable`, `+format/all2tbl`
- **dpkio** — console, tomo I/O, `+tomo` (MRC, binary)

### Alignment / matching / averaging
- **dpkvol** — `+align/+aux/+process/@Output`, `+align/+aux/+process/+forDev/@CCBuild` (uses `nativeCubeRoseman`), `+aux/+crossCorrelation` (nativeCubeRoseman, subpixel)
- **dpktomo** — `+match` (dynamo_match, tesselation), `+partition`, `+click`, `+tiltSeries`
- **dpktilt** — tilt-series: `+aux/+align` (aligners, workflows, fitters, CTF), `+aux/+markgold`, `+aux/+reconstructors` (WBP), `+sets`

### Classification / PCA / MRA
- **dpkpca** — eigenvolumes, subaverages
- **dpkmath** — `+pca` (XMatrix, ProjectWorkflow), `+prealign`, `+fsc`

### Geometry / models
- **dpkgeom** — `@dTriRep`, `+graph` (geodesics, mesh, MEX), `+interp`
- **dmodels** — `@curve`, `@membraneByLevels`, `@generalCrystal`

### Compatibility / external tools
- **dpcomp** — chimera, imod, ctffind, motioncor, serialEM, gautomatch, star
- **dpkctf** — CTF conventions

### GUI / dev
- **dpkdev** — `ghost.m` (function list), `+mex/compileItem`, `+pack`, `+commit`
- **dpkpm** — project manager GUI callbacks
- **dpkguis**, **dpkslicer** — UI components

### Other
- **dpksys** — `+categories/+lib/@Project`
- **mbgui** — menus, context
- **mbs**, **mbparse**, **pparse**, **modeltrack**, **row_checkin** — utilities

---

## 3) Core Data Models and Locations

| Model | Location | Description |
|-------|----------|-------------|
| **vpr (Virtual Project)** | `dynamo_vpr_load.m`, `dynamo_vpr_save.m`, `dynamo_vpr_modify.m` | Struct with project name, rounds, destination, masks, templates, tables, plugins. Stored as `.mat` in project folder. |
| **scard (iteration card)** | `dynamo_card_read.m`, `.card` files | Struct per iteration: `database_ite`, `database_ref`, `name_project`, alignment params, `use_CC`, `refine`, `cone_*`, `inplane_*`. |
| **Table (particle metadata)** | `dynamo_table_*`, columns 1–34 | Columns: tag, align flag, shifts (4–6), eulers (7–9), CC (10), fsampling (13–18), class (34). Convention in `dynamo_database_convention.m`. |
| **StarTable / FullTable** | `+dpktbl/@StarTable`, `+dpktbl/+star/@FullTable` | Relational/catalogue tables. |
| **volume_float** | `C/src/classes/volume_float.{h,cpp}` | C++: nx, ny, nz, float *v, bool *m; rotation, shift, mask ops. |
| **orientation, shift_parameters, binary_mask** | `C/src/classes/*.{h,cpp}` | Rigid transform components. |
| **dStar, dProject, dVolumeList** | `@dStar`, `@dProject`, `@dVolumeList` | Object wrappers for data. |

---

## 4) Side Effects

### Filesystem
- **Read:** `dynamo_read`, `dynamo_read_emfile`, `dynamo_read_mrc_simple`, `dynamo_read_subvolume`, `dynamo_database_read`, `dynamo_database_locate` → EM, MRC, `.tbl`, `.card`, `.mat`
- **Write:** `dynamo_write`, `dynamo_write_emfile`, `dynamo_database_locate` → `refined_ptable`, `refined_table`, `average`, `fmask_average`, `starting_reference`, `next_iteration`
- **Copy/Move:** `dynamo_copyfile`, move of "OLD" averages/fmasks in MRA plugin
- **Project DB layout:** `dynamo_database_convention.m` — `cards/`, `results/`, per-iteration and per-reference paths

### Network
- `+dpkio/web.m` — wiki/download
- `+dpkhelp/+wiki/downloadFile.m`
- `dynamo_chimera.m` — may call external Chimera

### Subprocess
- `unix()` in `dynamo_execute_project.m` for GPU motor: `system_gpu_command = sprintf('%s %s', myMotor, cardfile)`
- `dynamo_system.m` — `[status,result]=unix(command)` or `system()` to run external commands
- External tools: ctffind, motioncor, gautomatch, imod, Chimera

### GPU
- `dpkproject.runtime.getGPUmotor(vpr)` — returns path to GPU binary
- `dpkproject.runtime.checks.gpu(vpr)` — checks GPU
- Destinations: `matlab_gpu`, `standalone_gpu` — spawn subprocess with card file

### Environment variables
- **MCR_CACHE_ROOT** — MCR cache (warned if empty/missing in `dynamo.m`, `dynamo_iteration_compute.m`)

### Database (logical)
- No SQL/NoSQL; "database" = folder tree and `.card`/`.mat` files via `dynamo_database_locate`.

---

## 5) Config Surface Area

| Config | Location | Description |
|--------|----------|-------------|
| **Virtual project (vpr)** | `<project>/` | Loaded via `dynamo_vpr_load(name_project)`. Round/ref params: `destination`, `how_many_processors`, `matlab_workers_average`, `mra_r1`.., `plugin_post_r1`.., `plugin_post_order`, `file_mask`, `file_mask_classification`, `folder_data`, etc. |
| **Card files** | `cards/ite_*/card_ite*.card`, `card_iteref_*.card` | Per-iteration params. Read by `dynamo_card_read`, `dynamo_read`. |
| **dynamo_database_convention** | `dynamo_database_convention.m` | Filetypes and identifiers (ref, ite, tag, proc, eig, nex). |
| **dynamo_vpr_convention** | `dynamo_vpr_convention.m` | vpr field conventions. |
| **Env vars** | Shell | `MCR_CACHE_ROOT` (MCR standalone). |
| **GUI settings** | `+dpkpm/*Callback.m`, `pushbutton_save_Callback` | Project manager GUI writes to vpr/cards. |

---

## 6) External Dependencies

### Runtime
- **MATLAB** (or MCR for standalone)
- **MEX:** `+dpkgeom/+graph/mex` (perform_front_propagation_2d/3d, gw toolkit), `dynamo_mex.m`
- **External binaries** (optional): Chimera, IMOD, CTFfind, MotionCor2, gautomatch, jsubtomo, Xmipp
- **GPU binary** — returned by `dpkproject.runtime.getGPUmotor` (exact binary not located in this scan)

### Build
- C/C++: `C/src/classes/` (volume_float, orientation, shift_parameters, binary_mask)
- MEX: `+dpkgeom/+graph/mex/` (C++, gw toolkit)
- MATLAB Compiler for standalone

---

## 7) Hotspots (Large/Complex Files)

| File | Lines | Role |
|------|-------|------|
| `dynamo__help_standalone.m` | ~74,500 | Embedded help for standalone (generated/large). |
| `dynamo_gallery_BACKUP.m` | ~9,027 | Gallery GUI (legacy). |
| `dynamo_project_manager_BACKUP.m` | ~7,426 | Project manager (legacy). |
| `dpkdev/ghost.m` | ~5,128 | Developer function list / scaffolding. |
| `dynamo_tableview_legacy.m` | ~5,047 | Table viewer. |
| `dynamo_gallery.m` | ~4,887 | Gallery. |
| `dynamo__fastmapview.m` | ~4,590 | Map view. |
| `dynamo__project_manager.m` | ~4,481 | Project manager. |
| `dynamo_average.m` | ~3,146 | Averaging (main logic). |
| `dynamo_iteration_compute.m` | ~980 | Iteration compute (alignment orchestration). |
| `pipeline_align_one_particle.m` | ~694 | Per-particle alignment (Roseman, multigrid). |
| `dynamo_iteration_assemble.m` | ~890 | Assemble tables, average, symmetrize, copy to next iteration. |
| `dynamo_plugin_post_multireference_tutorial.m` | ~396 | Native MRA: particle assignment, new averages. |

---

## 8) Suspected Duplication / Repeated Patterns

| Pattern | Evidence |
|---------|----------|
| **Average variants** | `dynamo_average`, `dynamo_average_fast`, `dynamo_average_av3`, `dynamo_average_jsubtomo`, `dynamo_average_xmipp`, `dynamo_average_two`, `dynamo_average_fourier_sampling`, `dynamo_average_GUI` — different backends/inputs. |
| **Align variants** | `dynamo_align`, `dynamo_align_manual`, `dynamo_align_GUI`, `dynamo_align_sequential`, etc. |
| **BACKUP / legacy** | Many `*_BACKUP.m`, `*_backup.m`, `*_legacy.m` — likely superseded versions kept. |
| **Crop 2D vs 3D** | `dynamo_crop2d`, `dynamo_crop3d` — similar logic, different dimensions. |
| **dpktilt vs dpktomo** | Tilt-series vs tomogram tooling; some overlap in alignment/matching. |
| **vpr loading** | `dynamo_vpr_load` used in many places; sometimes passed as `name_project` or `vpr`. |
| **Processor table paths** | `assign_proc`, `starting_ptable`, `refined_ptable` vs `starting_proctable`, `refined_proctable` — two modes. |

---

## 9) Test Status

| Item | Location | Notes |
|------|----------|-------|
| `dynamo_crop_test.m` | `matlab/src/dynamo_crop_test.m` | Cropping border check. |
| `+dpkgeom/+graph/testing_geodesic_triangulation.m` | Geodesic triangulation test. |
| `+dmodels/test.m` | Model tests. |
| `+dpkdev/test.m` | Dev tests. |
| `dynamo__tomoview_test.m` | Tomo view test. |
| `dynamo_jsubtomo_eulers_b2d_test.m` | jsubtomo euler test. |
| **No formal suite** | — | No `runtests`, pytest, or coverage found. Tests appear ad hoc. |
| **How to run** | `dynamo` in MATLAB, then call `dynamo_crop_test(...)` etc. | No documented test runner. |

---

## 10) Unknowns / Inferred Gaps

| Unknown | Notes |
|---------|-------|
| GPU executable name/path | `dpkproject.runtime.getGPUmotor` referenced; implementation not found in scanned paths. |
| MRA vs multireference | `dynamo_plugin_post_multireference_tutorial` is the native MRA; relation to `mra_r1` and other MRA configs not fully traced. |
| `mbparse`, `mbs`, `pparse`, `mbgui`, `row_checkin` | External/private packages; not present in this repo. |
| Exact MCR version | `mcrversion()` checks for 9.1.0 in deployed mode; full MCR matrix unknown. |
| C/MEX build scripts | `+dpkdev/+mex/compileItem.m` exists; full build flow (Makefile, etc.) not located. |
| `dynamoForMatlab.tar` | Archive present; contents and usage not inspected. |
| jsubtomo, Xmipp, AV3 plugin flows | Mentioned in Contents; parsing/integration not fully mapped. |

---

## 11) Target Function Code Location Summary

For pydynamo refactoring, the primary source locations:

| Function | Key Files |
|----------|-----------|
| **Particle crop** | `dynamo_crop.m`, `dpkdata/+crop`, `dynamo_table_place.m` (uses dynamo_crop), tbl/vll via `dynamo_read`, vll via project |
| **Reconstruction (average)** | `dynamo_average.m`, `dynamo_applywedgemap.m`, `dpkproject.pipeline.genericInput2Average` |
| **Alignment** | `dpkproject.pipeline_align_one_particle.m`, `dynamo_iteration_compute.m`, `dpkvol/+align`, GPU binary |
| **MRA** | `dynamo_plugin_post_multireference_tutorial.m`, `pipeline_embedded_multirerefence.m`, `dynamo_iteration_setup.m`, `dynamo_iteration_assemble.m` |

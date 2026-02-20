# Multireference Analysis (MRA) — Technical Specification

**Document version:** 1.0  
**Source:** Dynamo v1.1.558 (dynamoForMatlab.tar)  
**Reference doc:** [Dynamo Wiki - Multireference Analysis](https://www.dynamo-em.org/w/index.php?title=Multireference_Analysis)

---

## 1. Overview

### 1.1 What is MRA?

**Multireference Analysis (MRA)** implements *simultaneous alignment and classification* in subtomogram averaging. Unlike PCA+k-means (which requires pre-aligned data), MRA performs alignment and classification in the same iterative loop.

Key characteristics:
- Each particle is aligned against **all** reference templates in each iteration.
- Each particle "chooses" the reference that gives the best cross-correlation score.
- Particles can **swap** between reference channels across iterations.
- Implemented as a multireference project with the **swap option** (`mra_r* = 1`) enabled.

### 1.2 Why MRA?

- PCA is difficult to scale for large datasets.
- PCA can miss heterogeneity in small subsets.
- PCA assumes correct alignment; if heterogeneity drives alignment to fail, PCA may not recover.

---

## 2. Source Code Architecture

### 2.1 High-Level Flow

```
dynamo_execute_project
    └── For each iteration (ite):
        1. dynamo_iteration_setup        → Creates (ref,tag) task assignations
        2. dynamo_iteration_compute      → Aligns each (ref,tag) pair
        3. dynamo_iteration_assemble     → Gathers refined tables, averages
        4. [if mra_r*==1] dynamo_plugin_post_multireference_tutorial  → MRA: best-ref selection, re-average
        5. dynamo_iteration_check         → Convergence, next iteration
```

### 2.2 Module Map

| Layer | Module | Path | Description |
|-------|--------|------|-------------|
| **Entry** | `dynamo_execute_project.m` | `matlab/src/` | Main execution loop; dispatches compute, assemble, MRA plugin |
| **Setup** | `dynamo_iteration_setup.m` | `matlab/src/` | Creates `assign_master`, `assign_proc`; prepares starting tables per ref |
| **Compute** | `dynamo_iteration_compute.m` | `matlab/src/` | Dispatches alignment tasks to CPU workers or GPU binary |
| **Align** | `dpkproject.pipeline_align_one_particle.m` | `matlab/src/+dpkproject/` | Single-particle alignment against one reference |
| **Assemble** | `dynamo_iteration_assemble.m` | `matlab/src/` | Gathers refined tables, averages, thresholding |
| **MRA** | `dynamo_plugin_post_multireference_tutorial.m` | `matlab/src/` | Best-ref selection, per-ref re-averaging |
| **Embedded MRA** | `pipeline_embedded_multirerefence.m` | `matlab/src/+dpkproject/` | Injects MRA plugin into execution when `mra_r*==1` |
| **GUI** | `setMultireferenceProject.m`, `dcp.m` | `matlab/src/@dcp/` | DCP GUI: nref, radiobutton_mra_swap |
| **Seeds** | `dynamo_write_multireference.m` | `matlab/src/` | Writes multireference seeds (template/table/fmask) |
| **List** | `dynamo_multireference_list.m` | `matlab/src/` | Lists multireference seed files in a folder |
| **Analysis** | `dynamo_multireference_particle_distribution.m` | `matlab/src/` | Particle distribution per reference per iteration |
| **C++** | `volume_float.cpp`, `volume_float_rotate.cpp` | `C/src/classes/` | Low-level 3D volume operations (used by alignment) |
| **GPU** | `dynamo_compute_iteration_spp` | External binary | GPU alignment kernel (per ref channel) |

---

## 3. Core Mechanisms

### 3.1 Task Assignation (Setup)

**File:** `dynamo_iteration_setup.m`

- For each surviving reference `ref`, reads `starting_table` and extracts tags to align (column 2).
- Builds `assign_refpar{ref} = [ref*ones(...), tags_in_ref{ref}]`.
- `assign_master = cat(1, assign_refpar{:})` → matrix `[ref, tag]` for every alignment task.
- **MRA (swap ON):** The same tags appear in multiple references’ starting tables → each particle is aligned against all references. Example: 2 refs, 3 particles → 6 tasks.
- **MRA (swap OFF):** Each particle appears in exactly one reference’s table → standard multireference alignment without swap.

`assign_proc` distributes `assign_master` rows across processors.

### 3.2 Alignment (Compute)

**File:** `dynamo_iteration_compute.m`

- For each `(ref, tag)` in `assign_proc`:
  - Loads `card_iteref` (or `card_itereftag`).
  - Calls `dpkproject.pipeline_align_one_particle(file_card_iteref, tag, ...)`.
- `pipeline_align_one_particle`:
  - Reads particle, reference, mask, wedge.
  - Searches over orientations and shifts (correlation-based).
  - Outputs refined row: Euler angles, shifts, CC score, `grep_average`, `ref` (column 34).

Each particle thus produces multiple refined rows (one per ref), each with its own CC.

### 3.3 Assemble (Default Averaging)

**File:** `dynamo_iteration_assemble.m`

- `dpkproject.gatherTablesAfterProcess`: Collects refined ptables into per-ref `refined_table`.
- For each ref: applies threshold, computes average from particles in that ref’s table.
- Without MRA plugin: average uses all particles in each ref’s table (no best-ref filtering).

### 3.4 MRA Plugin (Particle Swap)

**File:** `dynamo_plugin_post_multireference_tutorial.m`

**Trigger:** `vpr.mra_r{round} == 1` (DCP GUI: “swap particles” enabled).

**Logic:**
1. For each ref, load `refined_table` and restrict to `grep_align==1`.
2. Merge all tables → `table_all`; collect unique `all_tags`.
3. For each tag:
   - Collect CC from each ref’s row (column 10 or 11 depending on mask).
   - If alignment and classification masks differ, optionally recompute CC via `dpkproject.recomputeCC`.
   - `[best_CC, best_ref] = max(score_tag)` → particle assigned to `best_ref`.
4. For each ref:
   - `tags_in_ref{ref} = all_tags(best_reference == ref)`.
   - Re-average only those particles: `dynamo_average(..., 'tags', tags_in_ref{ref}, ...)`.
   - Write new average to `average` and `starting_reference` for next iteration.
   - Update `grep_average` (column 3) and ref membership (column 34) in tables.
5. Update surviving references: `dpkproject.pipeline.setSurvivingReferences`.

---

## 4. Data Conventions

### 4.1 Multireference File Naming

```
<folder_seeds>/template_initial_ref_001.em
<folder_seeds>/table_initial_ref_001.tbl
<folder_seeds>/fmask_initial_ref_001.em
```

Three-digit zero-padded reference index (001, 002, ...).

### 4.2 Table Columns (Conventions)

| Column | Name | Meaning |
|--------|------|---------|
| 1 | tag | Particle identifier |
| 2 | (align) | Include in alignment (1/0) |
| 3 | grep_average | Include in average (1/0) |
| 10/11 | CC | Cross-correlation (10 if align/class masks same; 11 if different) |
| 12 | CPU | Assigned processor |
| 34 | ref | Reference channel membership |

### 4.3 Project Parameters (vpr)

| Parameter | Type | Description |
|-----------|------|-------------|
| `nref_r1`, `nref_r2`, ... | int | Number of references per round |
| `mra_r1`, `mra_r2`, ... | 0/1 | Swap enabled for round |
| `plugin_post_r*` | 0/1 | Post-processing plugin enabled |
| `plugin_post_order_r*` | string | Plugin command (e.g. `dynamo_plugin_post_multireference_tutorial`) |

---

## 5. Database Locations (Conventions)

| Item | Path pattern |
|------|--------------|
| `starting_table` | `{project}/tables/ite_{ite}/ref_{ref}/linked_table_initial_ref_{ref}.tbl` |
| `refined_table` | `{project}/tables/ite_{ite}/ref_{ref}/refined_table_ref_{ref}.tbl` |
| `average` | `{project}/averages/ite_{ite}/average_ref_{ref}.mrc` |
| `starting_reference` | `{project}/references/ite_{ite}/ref_{ref}/starting_reference_ref_{ref}.mrc` |
| `card_ite` | `{project}/cards/ite_{ite}/card_ite_ite_{ite}.card` |
| `card_iteref` | `{project}/cards/ite_{ite}/card_iteref_ref_{ref}_ite_{ite}.card` |
| `assign_proc` | `{project}/cards/ite_{ite}/assign_proc_proc_{proc}.mat` |
| `assign_master` | `{project}/cards/ite_{ite}/assign_master.mat` |

---

## 6. Dependencies

### 6.1 MATLAB

- `dynamo_vpr_load`, `dynamo_vpr_save`
- `dynamo_database_locate`, `dynamo_read`, `dynamo_write`
- `dynamo_table_grep`, `dynamo_table_merge`, `dynamo_tag2subtable`
- `dynamo_average`, `dynamo_crop`
- `dpkproject.pipeline.getSurvivingReferences`, `setSurvivingReferences`
- `dpkproject.recomputeCC`, `dpkproject.addPtable`

### 6.2 External

- GPU: `dynamo_compute_iteration_spp` (C++ binary)

---

## 7. Refactoring Considerations

### 7.1 Porting to Python

| Component | Current | Suggested Python |
|-----------|---------|------------------|
| Project config | vpr struct, .mat | YAML/JSON or dataclass |
| Table format | .tbl | Pandas DataFrame or similar |
| Volume I/O | .em, .mrc | numpy + mrcfile |
| Correlation/search | MATLAB + C/mex | NumPy, CuPy, or custom Cython |

### 7.2 Decoupling Points

1. **Align vs MRA:** Alignment (single particle vs single ref) can stay separate; MRA is a post-assemble step.
2. **Plugin interface:** MRA plugin is standalone; can be a Python function with a clear contract (tables in, averages out).
3. **Parallelism:** `assign_proc` / `assign_master` define tasks; execution can be multiprocessing, Dask, or job arrays.

### 7.3 Critical Logic to Preserve

- Per-particle best-reference selection by CC.
- Per-reference re-averaging using only particles assigned to that ref.
- Table updates: `grep_average`, ref column.
- Surviving references propagation across iterations.

---

## 8. References

- [Dynamo Main Page](https://www.dynamo-em.org/w/index.php?title=Main_Page)
- [Classification](https://www.dynamo-em.org/w/index.php?title=Classification)
- [Multireference Analysis](https://www.dynamo-em.org/w/index.php?title=Multireference_Analysis)
- [Multireference project / alignment](https://www.dynamo-em.org/w/index.php?title=Multireference_project)
- Castaño-Díez et al., J. Struct. Biol. 2012; Acta Cryst D 2017

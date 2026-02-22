# Spec v1 — pydynamo Runnable Specification

**Version:** 1.0  
**Input:** System Map, MRA_SPEC, requirement_002  
**Rules:** No implementation detail; every rule must be testable; unknowns marked TBD.

---

## 1. Interface Contract

### 1.1 CLI Signature

```bash
pydynamo <command> --i <config.yaml> [options]
```

| Command | Description |
|---------|-------------|
| `crop` | Extract subtomograms from tomograms using tbl/vll or star metadata |
| `reconstruction` | Average subtomograms per table to produce density map |
| `alignment` | Align subtomograms against reference(s) |
| `classification` | MRA: align and classify subtomograms across multiple references |

### 1.2 Config File (YAML)

All commands accept `--i config.yaml`. Each command defines its own schema.

**Common fields (optional):**
- `log_level`: debug | info | warning | error
- `output_dir`: base output path
- `num_workers`: parallel workers (0 = auto)
- `device`: cpu | cuda | auto

### 1.3 crop

**Required inputs:**
- `particles`: Path to star or (tbl, vll) pair
- `tomograms`: Path to vll or directory/list of MRC tomograms
- `sidelength`: Integer (cube edge in voxels)
- `output_star`: Path to output star (particle metadata)
- `output_dir` or `output_pattern`: Where to write MRC subtomograms

**Optional:**
- `pixel_size`: For star conversion
- `tomogram_size`: (nx, ny, nz) for coordinate conversion
- `fill`: out-of-scope policy (0=zeros, -1=shrink+warning, -2=shrink, 1=empty)

**Validation:**
- tbl must have columns 1, 2, 3, 20, 24–26 (tag, align, average, tomo, x, y, z)
- vll: one path per line, MRC exists
- sidelength > 0, even (Dynamo convention)

**Output:**
- MRC files per particle
- Updated star with paths to subtomograms
- On error: exit code ≠ 0, message to stderr, JSON error block if `--json-errors`

### 1.4 reconstruction

**Required inputs:**
- `particles`: Path to star or tbl
- `subtomograms`: Path to star (with paths) or directory/pattern of MRC
- `output`: Path to output MRC average
- `sidelength`: Cube size (must match subtomogram size)

**Optional:**
- `fmask`: Fourier mask path
- `nmask`: Real-space mask
- `fcompensate`: 0|1 (Fourier compensation)
- `symmetry`: c1, c2, c4, etc.
- `tags`: Subset of tags to average

**Validation:**
- Table/star columns 4–6 (shifts), 7–9 (Euler angles), 20 (tomo)
- All subtomograms same dimensions
- Output path writable

**Output:**
- Single MRC average volume
- Optional: variance map, FSC if requested

### 1.5 alignment

**Required inputs:**
- `particles`: Star or tbl + vll
- `subtomograms`: Paths to MRC
- `reference`: Path to reference MRC
- `output_table`: Path to refined table/star
- `output_reference`: Path to output refined average (optional)

**Optional:**
- `fmask`, `fmask_classification`
- `cone_*`, `inplane_*`: angular sampling
- `refine`: refinement mode
- `max_iterations`

**Validation:**
- Reference, particles, subtomograms dimensions consistent
- Masks if provided: same size as volumes

**Output:**
- Refined table/star (Euler angles, shifts, CC)
- Optional refined average MRC

### 1.6 classification (MRA)

**Required inputs:**
- `particles`: Star or tbl
- `subtomograms`: Paths to MRC
- `references`: List of reference MRC paths (or folder with template_initial_ref_XXX.mrc)
- `tables`: List of table paths per ref (or folder)
- `output_dir`: Project output root
- `max_iterations`: Iteration count
- `swap`: true|false (MRA swap)

**Optional:**
- `fmask`, `fmask_classification` per ref
- `threshold`: CC threshold for averaging

**Validation:**
- Same number of references and tables
- All refs same dimensions

**Output:**
- Per-iteration: refined tables, averages, starting references
- Final assignment: ref membership (column 34), grep_average (column 3)

### 1.7 Accepted Input Formats

| Format | Extension | Validation |
|--------|-----------|------------|
| Dynamo tbl | .tbl | ASCII, space-separated, ≥35 columns, numeric |
| Dynamo vll | .vll | One path per line, file exists |
| RELION star | .star | starfile-readable, required columns |
| MRC volume | .mrc, .mrcs | mrcfile-readable, 3D |

### 1.8 Output Formats

| Type | Format |
|------|--------|
| Particle metadata | RELION star (preferred) or Dynamo tbl |
| Volumes | MRC |
| Errors | stderr text; `--json-errors` → JSON `{"error": "...", "code": N}` |
| Logs | stdout (info), or file if `--log-file` |

---

## 2. Semantics

### 2.1 crop

**Transformation:** For each particle row, read tomogram at (x,y,z) from tbl/star, extract cube of sidelength centered at that point, write MRC, record path in output star.

**Invariants:**
- Output subtomogram dimensions = (sidelength, sidelength, sidelength)
- Coordinates from tbl col 24–26 (or star rlnCenteredCoordinate + tomogram_size/2)
- Out-of-scope handling per `fill` option

### 2.2 reconstruction

**Transformation:** For each particle, load subtomogram, apply inverse rigid transform (Euler + shift from table), accumulate; divide by count; optionally apply symmetry.

**Invariants:**
- Euler angles ZXZ (Dynamo) or ZYZ (RELION) — conversion defined in io_dynamo
- Wedge/Fourier compensation if fcompensate=1
- Output dimensions = sidelength³

### 2.3 alignment

**Transformation:** For each particle, search over orientations and shifts to maximize cross-correlation with reference; output best Euler, shifts, CC.

**Invariants:**
- CC computed in Fourier space with optional mask
- Same convention as Dynamo: tdrot, tilt, narot (ZXZ degrees)

### 2.4 classification (MRA)

**State machine:** Iteration loop:
1. **Setup:** Build (ref, tag) task list. If swap: each tag appears in all refs.
2. **Compute:** For each (ref, tag), run alignment → refined row per ref.
3. **Assemble:** Gather refined tables, compute per-ref average (default).
4. **MRA (if swap):** For each tag, pick best ref by CC; re-average per ref with assigned particles only; update tables.
5. **Check:** Convergence or max_iterations → stop.

**Invariants:**
- Per-particle best-ref = argmax CC over refs
- grep_average and ref column updated after MRA

---

## 3. Errors

### 3.1 Retry

- File read (transient I/O): retry up to 3 times with backoff. TBD: configurable.

### 3.2 Fallback

- GPU unavailable → fallback to CPU. Log warning.
- MRC read fails → try .mrcs. TBD: document.

### 3.3 Fatal

- Missing required config key
- Invalid dimensions (mismatch ref/particle/sidelength)
- Malformed tbl/star (parse error)
- Output path not writable
- Division by zero (e.g. no particles after threshold)

### 3.4 Exit Codes

- 0: success
- 1: config/input validation error
- 2: runtime error (I/O, memory)
- 3: algorithm error (e.g. convergence failure)

---

## 4. Performance

### 4.1 Constraints

- crop: O(N) in number of particles; I/O bound
- reconstruction: O(N × L³) for N particles, L=sidelength
- alignment: O(N × R × S) where R=refs, S=sampling points; dominant cost
- classification: O(iter × N × R × S)

### 4.2 Targets (TBD)

- Latency: TBD (depends on hardware)
- Throughput: TBD
- Memory: Should support L≤256 on 16GB RAM; document limits

### 4.3 Parallelism

- crop: parallel over particles (multiprocessing or GPU batch)
- reconstruction: parallel over particles
- alignment: parallel over (ref, tag) tasks
- classification: parallel within iteration

---

## 5. Observability

### 5.1 Logs

- Log level configurable
- Each command logs: start, config summary, progress (e.g. N/Total), finish, duration
- Errors with context (file, line, config snippet)

### 5.2 Metrics (TBD)

- Particles processed, CC mean/std, iteration count
- Optional: export to JSON/metrics file

### 5.3 Traces (TBD)

- No distributed tracing initially

---

## 6. Parity Checklist

### 6.1 Crop (10)

- [ ] CROP-1: Read tbl with vll, resolve tomo id → path
- [ ] CROP-2: Read star with rlnMicrographName, resolve to paths
- [ ] CROP-3: Extract cube at (x,y,z) with correct sidelength
- [ ] CROP-4: fill=0: out-of-scope → zeros
- [ ] CROP-5: fill=-1: out-of-scope → shrink + warning
- [ ] CROP-6: fill=-2: out-of-scope → shrink, no warning
- [ ] CROP-7: fill=1: out-of-scope → empty/skip
- [ ] CROP-8: Output MRC compatible with mrcfile
- [ ] CROP-9: Output star has correct particle paths
- [ ] CROP-10: tbl↔star conversion matches TomoPANDA-pick/utils

### 6.2 Reconstruction (12)

- [ ] RECON-1: Read subtomograms by paths from star/tbl
- [ ] RECON-2: Apply Euler (ZXZ) rotation correctly
- [ ] RECON-3: Apply shift (dx,dy,dz) correctly
- [ ] RECON-4: Fourier compensation when fcompensate=1
- [ ] RECON-5: Wedge application from fmask
- [ ] RECON-6: Real-space mask (nmask) if provided
- [ ] RECON-7: Symmetry (c2, c4, etc.) applied
- [ ] RECON-8: Tags filter: average only requested tags
- [ ] RECON-9: Output MRC dimensions correct
- [ ] RECON-10: Variance map if requested
- [ ] RECON-11: FSC computation if requested
- [ ] RECON-12: Parity with dynamo_average on same inputs

### 6.3 Alignment (15)

- [ ] ALIGN-1: Load particle, reference, masks
- [ ] ALIGN-2: Angular search (cone, inplane) matches Dynamo
- [ ] ALIGN-3: Shift search range/config
- [ ] ALIGN-4: CC in Fourier space
- [ ] ALIGN-5: Subpixel refinement
- [ ] ALIGN-6: Output tdrot, tilt, narot (ZXZ degrees)
- [ ] ALIGN-7: Output dx, dy, dz
- [ ] ALIGN-8: Output CC (col 10 or 11)
- [ ] ALIGN-9: grep_average, ref column in output
- [ ] ALIGN-10: Wedge handling
- [ ] ALIGN-11: fmask vs fmask_classification when different
- [ ] ALIGN-12: GPU path when available
- [ ] ALIGN-13: CPU fallback
- [ ] ALIGN-14: Multi-GPU distribution
- [ ] ALIGN-15: Parity with pipeline_align_one_particle on same inputs

### 6.4 Classification / MRA (18)

- [ ] MRA-1: Setup: assign_master with swap ON → all (ref,tag) pairs
- [ ] MRA-2: Setup: assign_master with swap OFF → each tag in one ref
- [ ] MRA-3: Compute: align each (ref,tag)
- [ ] MRA-4: Assemble: gather refined tables per ref
- [ ] MRA-5: Assemble: default average per ref
- [ ] MRA-6: MRA plugin: merge tables, collect all_tags
- [ ] MRA-7: MRA plugin: per-tag best_ref = argmax CC
- [ ] MRA-8: MRA plugin: recomputeCC when align≠class mask (optional)
- [ ] MRA-9: MRA plugin: tags_in_ref{ref} = tags with best_ref==ref
- [ ] MRA-10: MRA plugin: re-average per ref with tags_in_ref only
- [ ] MRA-11: MRA plugin: write new average, starting_reference
- [ ] MRA-12: MRA plugin: update grep_average (col 3)
- [ ] MRA-13: MRA plugin: update ref (col 34)
- [ ] MRA-14: setSurvivingReferences after MRA
- [ ] MRA-15: Iteration loop until convergence or max_iter
- [ ] MRA-16: File naming: average_ref_XXX, starting_reference_ref_XXX
- [ ] MRA-17: Multi-reference seeds: template_initial_ref_XXX, table_initial_ref_XXX
- [ ] MRA-18: Parity with dynamo_plugin_post_multireference_tutorial flow

### 6.5 I/O & Integration (15)

- [ ] IO-1: tbl read with complex token handling (0+1.2e-06i → real)
- [ ] IO-2: vll read, blank/# lines skipped
- [ ] IO-3: star read via starfile
- [ ] IO-4: star write via starfile
- [ ] IO-5: MRC read via mrcfile
- [ ] IO-6: MRC write via mrcfile
- [ ] IO-7: tbl↔star coordinate conversion (Dynamo↔RELION)
- [ ] IO-8: Euler ZXZ↔ZYZ conversion (io_eular / convert_euler)
- [ ] IO-9: Pixel size, tomogram_size for coordinate conversion
- [ ] IO-10: COLUMNS_NAME convention (1–35)
- [ ] IO-11: create_dynamo_table, read_dynamo_tbl parity with TomoPANDA-pick
- [ ] IO-12: dynamo_tbl_vll_to_relion_star parity
- [ ] IO-13: relion_star_to_dynamo_tbl parity
- [ ] IO-14: Config YAML schema validation
- [ ] IO-15: --json-errors output format

### 6.6 System (10)

- [ ] SYS-1: pydynamo --help lists commands
- [ ] SYS-2: pydynamo crop --help shows crop options
- [ ] SYS-3: pydynamo reconstruction --help
- [ ] SYS-4: pydynamo alignment --help
- [ ] SYS-5: pydynamo classification --help
- [ ] SYS-6: Missing --i exits with code 1
- [ ] SYS-7: Invalid YAML exits with code 1
- [ ] SYS-8: Log level respected
- [ ] SYS-9: Output dir created if not exists
- [ ] SYS-10: One-command run (script or make target)

**Total:** 80 checkboxes

# pydynamo

`pydynamo` is a Python CLI toolkit for subtomogram workflows inspired by Dynamo:

- synthetic data generation
- crop
- reconstruction
- alignment
- classification (MRA)

It is designed for local development and HPC/Slurm execution.

---

## 1. Installation

### 1.1 Standard local/dev install

```bash
conda create --name pydynamo python=3.12
conda activate pydynamo
cd pydynamo
pip install -e .
```

### 1.2 Slurm/HPC install (CUDA on compute nodes)

Use this when login nodes do not have visible GPUs, but compute nodes do.

```bash
conda create --name pydynamo python=3.12
conda activate pydynamo
cd pydynamo
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Validate on an allocated compute node:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Notes:

- `torch.cuda.is_available()` may be `False` on login nodes; this is expected.
- Validate CUDA on the real compute node runtime environment.

---

## 2. CLI usage

```bash
pydynamo <command> --i <config.yaml>
```

Available commands:

- `gen_synthetic`
- `crop`
- `reconstruction`
- `alignment`
- `classification`

Common options:

- `--log-level {debug,info,warning,error}`
- `--log-file <path>`
- `--json-errors`

---

## 3. Config templates

Default config templates are under `config/`:

| Command | Example config |
|---|---|
| `gen_synthetic` | `config/synthetic_defaults.yaml` |
| `crop` | `config/synthetic_crop.yaml` |
| `reconstruction` | `config/synthetic_recon.yaml` |
| `alignment` | `config/synthetic_align.yaml` |
| `classification` | `config/synthetic_class.yaml` |

Also available:

- `config/synthetic_align_quicktest.yaml`
- `config/synthetic_recon_withMissingWedge.yaml`
- `config/synthetic_defaults_withMissingWedge.yaml`
- `*_defaults.yaml` for command-level defaults

---

## 4. Quick start

Run a minimal end-to-end smoke flow:

```bash
cd pydynamo
OMP_NUM_THREADS=1 pydynamo gen_synthetic --i config/synthetic_defaults.yaml
pydynamo crop --i config/synthetic_crop.yaml
pydynamo reconstruction --i config/synthetic_recon.yaml
pydynamo alignment --i config/synthetic_align.yaml
pydynamo classification --i config/synthetic_class.yaml
```

---

## 5. Runtime and parallel defaults

`pydynamo` supports conservative auto defaults:

- `crop`
  - `num_workers <= 0` => use all detected CPU cores.
- `alignment` / `classification`
  - `device: auto` => use CUDA if available, otherwise CPU.
  - if CUDA is available and no explicit GPU is set, all detected GPUs are used.
  - multi-GPU task distribution is by particle index round-robin.
  - on CPU, `num_workers > 1` => multi-process alignment/classification (`<=0` => auto: cpu_count-1).
- `reconstruction`
  - `recon_workers > 1` => multi-process per-chunk accumulation (`<=0` => auto: cpu_count-1).

Useful fields:

| Field | Commands | Meaning |
|---|---|---|
| `num_workers` | `crop`, `alignment`, `classification` | CPU worker count (`<=0` for auto) |
| `recon_workers` | `reconstruction` | process workers for recon (`1` = single) |
| `device` | `alignment`,`classification` | `cpu` / `cuda` / `auto` |
| `device_id` | `alignment`,`classification` | single GPU override |
| `gpu_ids` | `alignment`,`classification` | explicit GPU list |
| `progress_log_every` | all major commands | progress log frequency |

`alignment` and `classification` also support wedge-aware scoring and table-driven fsampling controls via YAML.

### Recommended settings for ~60k particles

For large-scale runs (e.g. 60,000 particles), use explicit worker counts to avoid overloading the machine and to get predictable throughput:

| Command         | Parameter          | Recommended        | Notes |
|----------------|--------------------|--------------------|-------|
| alignment      | num_workers        | 8–16 or 0 (auto)   | For 60k, prefer explicit 8–16; auto may use all cores |
| alignment      | multigrid_levels   | 2                  | Keeps coarse-to-fine, controls time per particle |
| alignment      | cone_step / inplane_step | 15       | Finer steps increase runtime significantly |
| alignment      | device             | cuda if available  | GPU reduces wall time when supported |
| reconstruction | recon_workers      | 8–16               | Larger values add scheduling/merge cost; 8–16 is usually sufficient |
| classification | num_workers        | 8–16               | Same as alignment |
| classification | max_iterations     | 3–5                | Per iteration already reads each particle once |
| crop           | num_workers        | 4–16               | Limit by I/O and tomogram count; avoid >> tomogram count × per-tomo parallelism |

- **num_workers / recon_workers**: For 60k, set explicitly to 8–16 instead of relying on auto, to avoid using all cores. Very large worker counts increase IPC and scheduling overhead.
- **crop**: If the number of tomograms is much smaller than num_workers, some workers may sit idle; keep num_workers at or below roughly (tomogram count × desired per-tomogram parallelism) or CPU count.

---

## 6. Output notes

- `crop` writes subtomograms as:
  - `particle_000000000001.mrc` (12-digit zero-padded index)
- `alignment` can output:
  - Dynamo-style refined table (`.tbl`) for internal refined parameters
  - RELION-style `.star` when configured
- `reconstruction` and `classification` write averaged volumes (`.mrc`) and stage outputs.

---

## 7. Logging and observability

Progress logs include:

- elapsed time
- ETA
- ETA wall-clock timestamp
- memory stats (`rss_cur`, `rss_avg`, `rss_peak`)

This is emitted to terminal and log file (if `--log-file` is set).

---

## 8. Slurm usage

See `slurm_example/README.md` for:

- per-stage Slurm scripts
- full pipeline script
- matching config files
- path/resource fields to customize before `sbatch`

---

## 9. Development and tests

Install test tools:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest -q
```

Focused examples:

```bash
python -m pytest -q pydynamo/test/test_align.py pydynamo/test/test_alignment_command.py
python -m pytest -q pydynamo/test/test_crop.py
```

---

## 10. Troubleshooting

### OMP Error #179 / thread oversubscription

Set:

```bash
export OMP_NUM_THREADS=1
```

### CUDA requested but unavailable

- check PyTorch CUDA build/version
- validate inside allocated compute node
- if needed, run with `device: cpu` temporarily

### Shape mismatch errors in alignment

- ensure particle volumes and reference volume have the same shape
- verify mask shapes (`nmask`, wedge settings) are compatible with your workflow

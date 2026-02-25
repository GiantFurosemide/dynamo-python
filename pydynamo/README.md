# pydynamo

Python CLI for Dynamo subtomogram averaging: crop, reconstruction, alignment, classification (MRA).

## Install

### Standard (local/dev)

```bash
conda create --name pydynamo python=3.12
conda activate pydynamo
cd pydynamo && pip install -e .
```

### Slurm/HPC (CUDA 12.1 on compute nodes)

Use this when the login/install node has no GPU, but compute nodes have NVIDIA GPUs.

```bash
conda create --name pydynamo python=3.12
conda activate pydynamo
cd pydynamo
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e . --no-deps
```

Validate on a compute node allocation:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Notes:

- On login nodes without GPU, `torch.cuda.is_available()` can be `False` and this is expected.
- Final CUDA validation should be done on compute nodes under Slurm allocation.

## Run

```bash
pydynamo <command> --i config.yaml
```

| Command | Config example |
|---------|----------------|
| `gen_synthetic` | `config/synthetic_defaults.yaml` |
| `crop` | `config/synthetic_crop.yaml` |
| `reconstruction` | `config/synthetic_recon.yaml` |
| `alignment` | `config/synthetic_align.yaml` |
| `classification` | `config/synthetic_class.yaml` |

## Quick test

```bash
cd pydynamo
OMP_NUM_THREADS=1 pydynamo gen_synthetic --i config/synthetic_defaults.yaml
pydynamo crop --i config/synthetic_crop.yaml
pydynamo reconstruction --i config/synthetic_recon.yaml
```

## Default parallel policy

`pydynamo` uses conservative auto-parallel defaults so a single config file is enough for most runs:

- `crop`:
  - `num_workers <= 0` means auto-detect and use all available CPUs.
- `alignment` and `classification`:
  - `device: auto` means use CUDA when available, otherwise CPU.
  - when CUDA is available, auto mode uses all detected GPUs by default.
  - tasks are distributed across GPUs (round-robin by particle index).

## YAML fields (parallel + device)

These fields are explicitly available in config files under `config/*.yaml`:

| Field | Commands | Type | Meaning |
|------|----------|------|---------|
| `num_workers` | `crop` | int | Number of CPU workers. `<=0` => all detected CPUs. |
| `device` | `alignment`, `classification` | string | `cpu`, `cuda`, or `auto`. `auto` prefers CUDA and falls back to CPU. |
| `device_id` | `alignment`, `classification` | int/null | Optional single GPU override. Example: `0`. |
| `gpu_ids` | `alignment`, `classification` | list/null | Optional explicit GPU list. Example: `[0, 1]`. `null` => all detected GPUs. |

Notes:

- `device_id` and `gpu_ids` override default GPU selection behavior.
- If CUDA is requested but not available, execution falls back to CPU with a warning.
- For reproducible benchmarking, prefer explicit `gpu_ids` and fixed angular search settings.

## Troubleshooting

**OMP Error #179** — Set before running:
```bash
export OMP_NUM_THREADS=1
```

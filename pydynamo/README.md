# pydynamo

Python CLI for Dynamo subtomogram averaging: crop, reconstruction, alignment, classification (MRA).

## Install

```bash
cd pydynamo && pip install -e .
```

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

## Troubleshooting

**OMP Error #179** — Set before running:
```bash
export OMP_NUM_THREADS=1
```

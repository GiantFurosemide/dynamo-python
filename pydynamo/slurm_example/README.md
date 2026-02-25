# slurm_example

Slurm run package for `pydynamo` with:

- one script per stage (`crop`, `reconstruction`, `alignment`, `classification`)
- one full pipeline script (`pipeline_all.slurm`)
- matching YAML configs in `config/`

## Files

- `crop.slurm`
- `reconstruction.slurm`
- `alignment.slurm`
- `classification.slurm`
- `pipeline_all.slurm`
- `config/crop.yaml`
- `config/reconstruction.yaml`
- `config/alignment.yaml`
- `config/classification.yaml`

## Before submit

Edit these variables in scripts if your cluster paths differ:

- `CONDA_BASE` (default `/home/muwang/miniforge3`)
- `CONDA_ENV` (default `pydynamo`)
- `REPO_ROOT` (default `/home/muwang/Documents/GitHub/dynamo-python`)
- `CONFIG_PATH` (or `*_CFG` in `pipeline_all.slurm`)

Adjust `#SBATCH` resources for your partition/GPU type as needed.

## Submit examples

Run single stage:

```bash
cd /home/muwang/Documents/GitHub/dynamo-python/slurm_example
sbatch crop.slurm
sbatch reconstruction.slurm
sbatch alignment.slurm
sbatch classification.slurm
```

Run full pipeline (single job, sequential stages):

```bash
cd /home/muwang/Documents/GitHub/dynamo-python/slurm_example
sbatch pipeline_all.slurm
```

## Output layout

Configured outputs are written under:

- `slurm_example/output/crop/`
- `slurm_example/output/reconstruction/`
- `slurm_example/output/alignment/`
- `slurm_example/output/classification/`

## Notes

- `alignment.slurm` and `classification.slurm` print CUDA availability via PyTorch for quick environment diagnosis.
- Pipeline order in `pipeline_all.slurm`: `crop -> reconstruction -> alignment -> classification`.

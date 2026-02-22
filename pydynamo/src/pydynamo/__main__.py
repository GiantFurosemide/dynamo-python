"""Entry point for python -m pydynamo. Set OMP env vars before any numpy/scipy load."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from .cli import main
main()

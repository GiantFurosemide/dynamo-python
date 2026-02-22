# pydynamo one-click targets
PYDYNAMO = python -m pydynamo.cli
PYDYNAMO_SRC = PYTHONPATH=src $(PYDYNAMO)

.PHONY: gen_data test_crop test_recon test_align test_class run_all help

help:
	@echo "pydynamo one-click targets:"
	@echo "  make gen_data     - Generate synthetic data"
	@echo "  make test_crop    - Run crop on synthetic data"
	@echo "  make test_recon   - Run reconstruction"
	@echo "  make test_align   - Run alignment"
	@echo "  make test_class   - Run classification (MRA)"
	@echo "  make run_all      - gen_data + test_crop + test_recon"
	@echo "  make pytest       - Run unit tests"

gen_data:
	$(PYDYNAMO_SRC) gen_synthetic --i config/synthetic_defaults.yaml

test_crop: gen_data
	$(PYDYNAMO_SRC) crop --i config/synthetic_crop.yaml

test_recon: test_crop
	$(PYDYNAMO_SRC) reconstruction --i config/synthetic_recon.yaml

test_align: test_crop
	$(PYDYNAMO_SRC) alignment --i config/synthetic_align.yaml

test_class: test_crop
	$(PYDYNAMO_SRC) classification --i config/synthetic_class.yaml

run_all: gen_data test_crop test_recon

pytest:
	PYTHONPATH=src python -m pytest test/test_cli.py test/test_crop.py test/test_average.py test/test_align.py -v

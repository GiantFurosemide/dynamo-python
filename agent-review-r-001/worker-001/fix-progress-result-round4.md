# Worker-001 Fix Progress & Result — Round 4

## 1) Round objective

Close residual audit requests from round-2 feedback:

- `F-R2-3`: ALN-003 tests should prove branch behavior, not only argument forwarding.
- `F-R2-4`: `wedge_apply_to=auto` needs explicit contract tests.

## 2) Implemented changes

### Test enhancements in `pydynamo/test/test_align.py`

1. `test_wedge_apply_to_auto_contract_matrix`
   - Validates `_resolve_wedge_apply_to(...)` mapping:
     - `auto + (fs1>0,fs2<=0) -> particle`
     - `auto + (fs1<=0,fs2>0) -> template`
     - `auto + mixed/none -> both`
2. `test_fsampling_auto_mode_matches_expected_explicit_branch`
   - Parity-branch assertion:
     - under `fsampling_mode=table`, output from `wedge_apply_to=auto`
       equals output from expected explicit branch (`particle/template/both`)
       for multiple fsampling cases.

These tests provide behavior-level evidence for auto-branch contract and fsampling-driven side selection.

## 3) Regression evidence

### Focused

Command:

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_align.py`

Result:

- `22 passed`

### Full suite

Command:

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`

Result:

- `67 passed`

## 4) Status discipline update

| Item | Round-4 status |
|---|---|
| ALN-001 | DONE |
| ALN-002 | DONE |
| ALN-003 | PARTIAL |
| ALN-004 | DONE |
| ALN-005 | DONE |
| ALN-006 | DONE |
| ALN-007 | DONE |
| ALN-008 | DONE |

Why ALN-003 remains `PARTIAL`:

- branch behavior and auto-contract evidence is now stronger,
- but strict Dynamo-equivalence across full `use_CC` branch matrix still lacks direct reference-comparison proof.

## 5) Round conclusion

Round-4 closes F-R2-3/F-R2-4 testing gaps at contract/behavior level and keeps the full suite green.

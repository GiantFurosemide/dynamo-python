# pydynamo System Map (Algorithm Scope) / pydynamo 系统地图（算法范围）

## 1) Command entrypoints / 命令入口

- CLI command modules:
  - `pydynamo/src/pydynamo/commands/crop.py`
  - `pydynamo/src/pydynamo/commands/reconstruction.py`
  - `pydynamo/src/pydynamo/commands/alignment.py`
  - `pydynamo/src/pydynamo/commands/classification.py`
- Shared runtime utilities:
  - `pydynamo/src/pydynamo/runtime.py`

## 2) Core algorithm modules / 核心算法模块

- **Alignment core**
  - `pydynamo/src/pydynamo/core/align.py`
  - CPU path + optional PyTorch GPU path.
  - Angle sampling (legacy + Dynamo-like mode), shift search constraints, subpixel refinement, correlation backends.
- **Averaging / transforms**
  - `pydynamo/src/pydynamo/core/average.py`
  - Inverse transform and symmetry application.
- **Wedge tools**
  - `pydynamo/src/pydynamo/core/wedge.py`
- **Cropping**
  - `pydynamo/src/pydynamo/core/crop.py`

## 3) I/O and format conversion / I/O 与格式转换

- Dynamo tbl / vll and RELION star bridges:
  - `pydynamo/src/pydynamo/io/io_dynamo.py`
  - `pydynamo/src/pydynamo/io/io_eular.py`
- Command-layer conversion:
  - STAR <-/-> Dynamo columns, Euler conversion, origin shift conversion.

## 4) Runtime orchestration / 运行编排

- `alignment.py`
  - Reads config and data.
  - Dispatches per-particle alignment.
  - Supports multi-GPU scheduling by task modulo.
  - Writes refined table/star and optional reconstructed average.
- `classification.py`
  - Iterative MRA-like loop over references and particles.
  - Performs best-reference selection and per-reference average update.
  - Checkpoint-based resume support.
- `reconstruction.py`
  - Streamed accumulation with optional wedge/fcompensate.
- `crop.py`
  - Groups tasks by tomogram to avoid repeated I/O loads.

## 5) Observability / 可观测性

- Progress logging hooks with:
  - elapsed
  - ETA
  - estimated finish timestamp
  - RSS memory signals
- Error reporting via runtime error log and optional JSON error output.

## 6) High-level data flow / 高层数据流

1. Parse YAML + resolve paths/devices.
2. Read table/star + volumes + masks.
3. Run algorithmic core per command.
4. Stream/aggregate outputs (tbl/star/mrc).
5. Log progress and diagnostics.

## 7) Scope notes / 范围说明

- This map emphasizes algorithm semantics and command-core coupling.
- Packaging/install/test harness details are omitted here.

# Requirement 005 (Refined) — GPU补全、测试闭环、交付报告

**Source:** `requirements_refined_004.md`, `response_004.md`, `docs/spec_v1.md`  
**Goal:** 继续实现未完成部分，完成测试生成与执行，形成可复现的测试报告

---

## 1. 本轮范围 / Scope

1. 补全未完成实现（重点：alignment/classification 的 GPU 参数链路与可执行路径）
2. 补齐测试用例（含 CPU 与 GPU 可用路径）
3. 在 `conda` 环境 `pydynamo` 中执行测试
4. 输出执行结果与完成度报告到 `response_005.md`
5. 实现多 GPU 并行调度（任务切分到多个 device）
6. 默认并行策略收敛：crop 自动用全部 CPU；alignment/classification 自动用全部可探测 GPU
7. 更新全部 `pydynamo/config/*.yaml`：显式展示并行相关设定，且每个变量均附注释
8. 为全部命令增加进度条与错误日志文件输出能力

---

## 2. 任务拆分 / Task Breakdown

### T1. Alignment GPU执行路径补全（P0）

- 在 `core.align` 中补全 PyTorch CUDA 搜索路径（至少保证 shift + CC 在 GPU 上执行）
- 支持配置 GPU 设备号（`device_id`）
- 保留 CUDA 不可用时回退 CPU 的行为
- 将旋转重采样阶段进一步 GPU 化：使用 CUDA 侧 3D 插值，不再依赖 CPU `scipy/map_coordinates`

### T2. Classification 参数透传补全（P0）

- 在 `commands/classification.py` 增加并使用：
  - `shift_search`
  - `multigrid_levels`
  - `device`
- 确保调用 `align_one_particle()` 时参数完整透传

### T3. Alignment 命令 GPU设备参数（P1）

- 在 `commands/alignment.py` 增加 `device_id` 支持并传入核心对齐函数

### T3.1 多 GPU 调度（Alignment/Classification）（P0）

- `alignment`：
  - 当 `device=auto|cuda` 且检测到多个 GPU 时，按粒子任务切分到多个 GPU
  - 默认使用全部可检测 GPU（`gpu_ids: null`）
- `classification`：
  - 当 `device=auto|cuda` 且检测到多个 GPU 时，按粒子任务切分到多个 GPU
  - 默认使用全部可检测 GPU（`gpu_ids: null`）

### T3.2 默认并行策略（P0）

- `crop`：`num_workers<=0` 时，自动解析并使用全部 CPU
- `alignment` / `classification`：`device=auto` 时，优先 CUDA；若可用则默认使用全部 GPU

### T3.3 YAML 配置显式化与注释（P0）

- 覆盖 `pydynamo/config/*.yaml` 全部文件
- 每个变量必须有注释（含含义、单位或取值范围）
- 并行相关变量显式出现：`num_workers`, `device`, `device_id`, `gpu_ids`

### T3.4 进度与错误日志（P0）

- 命令范围：`crop`, `reconstruction`, `alignment`, `classification`, `gen_synthetic`
- 每个命令具备进度跟踪能力，但默认不在终端打印进度条文本
- 错误信息支持输出到文件（优先 `--log-file`，其次 YAML `error_log_file`/`log_file`）
- 错误输出同时保留 stderr（含 `--json-errors` 行为）
- YAML 中显式提供 `log_file` 与 `error_log_file`
- 当 YAML 中未显式填写路径时，默认输出到与 YAML 同目录（`<config>.log` 与 `<config>.error.log`）
- `crop` 与 `reconstruction` 在运行中按粒子数分段写入进度日志（默认每 10 个粒子记录一次）

### T3.5 Reconstruction 输出 voxel_size 对齐配置（P0）

- `reconstruction` 输出 MRC 的 `voxel_size` 必须显式写为 YAML `pixel_size`
- 不允许依赖输入文件继承或默认值，避免下游尺度不一致

### T3.6 Crop 内存优化（P0，按建议 1/2/3）

- 建议 1：tomogram 读取改为 mmap/非 copy（避免每任务整卷复制）
- 建议 2：按 tomogram 分组调度任务（每个 tomogram 只加载一次）
- 建议 3：单 tomogram 场景使用线程并行（共享同一 volume 视图）
- 目标：在多 worker 下显著降低峰值内存占用，不改变 crop 结果语义

### T3.7 STAR 相对路径解析一致性（P0）

- `alignment` / `classification` 在读取 `.star` 的 `rlnImageName` 为相对路径时：
  - 优先从 `subtomograms` 目录解析
  - 若未命中，再回退到 `particles.star` 所在目录
- 目标：避免路径被错误拼接到 metadata 目录导致粒子全部 skip

### T3.8 Real-space Mask YAML 接入（P0）

- `alignment` / `classification` / `reconstruction` 支持在 YAML 中指定 real-space mask
- 配置键：`nmask`
- 约束：
  - mask 需与参与计算体数据尺寸一致
  - 仅 mask 非零体素参与计算（对齐 NCC、分类平均、重构累积）

### T3.9 命令启动输入日志（P0）

- 目标：所有命令在开始执行时记录“接收到的输入项”
- 覆盖命令：`crop`, `reconstruction`, `alignment`, `classification`, `gen_synthetic`
- 日志内容：
  - command 名称
  - `config_path`
  - YAML 解析后的完整配置
  - 关键 CLI 参数（`log_level`, `log_file`, `json_errors`）
  - 透传附加参数（若存在）

### T3.10 Alignment 直接重建平均体（P0）

- `alignment` 在输出 `star/tbl` 后，直接基于对齐结果重建平均体
- 输出路径：
  - `output_average` 显式指定时按该路径写出
  - 未指定时默认写到 alignment 输出目录下 `average.mrc`
- 支持 `average_symmetry`（默认 `c1`）对平均体做对称化
- 输出 `average.mrc` 的 `voxel_size` 与 `pixel_size` 保持一致

### T3.11 Classification 直接输出最终平均体（P0）

- `classification` 在 MRA 迭代结束后直接输出最终平均体文件，便于直接下游使用
- 输出路径：
  - 单参考（`references` 长度为 1）：
    - `output_average` 显式指定时按该路径写出
    - 未指定时默认写到 `output_dir/average.mrc`
  - 多参考（`references` 长度 > 1）：
    - 输出到 `output_average_dir`（未指定则为 `output_dir`）下的 `average_ref_XXX.mrc`
- 支持 `average_symmetry`（默认 `c1`）对最终平均体做对称化
- 输出 MRC 的 `voxel_size` 与 `pixel_size` 保持一致

### T4. 测试补全（P0）

新增/补强测试，覆盖：

- `core.align`:
  - CPU路径
  - CUDA路径（仅在 `torch.cuda.is_available()` 为 true 时执行）
  - auto 设备解析
  - 旋转阶段 GPU 化验证（GPU 路径不得回落到 CPU `rotate_volume`）
- `commands/alignment`:
  - 最小配置执行并输出 star
  - 对齐完成后自动输出 average.mrc
  - `output_average` 可覆盖默认平均体输出路径
  - auto 模式下设备解析为全部 GPU（可通过 mock 验证）
  - 多 GPU 任务分发（可通过 mock 验证 device_id 分配）
- `commands/classification`:
  - 单迭代可运行并输出 `average_ref_*.mrc` / `refined_table_ref_*.tbl`
  - 迭代结束后自动输出最终平均体（单参考 `average.mrc`；多参考 `average_ref_XXX.mrc`）
  - `output_average` 可覆盖单参考默认最终平均体路径
  - 参数透传验证（`shift_search/multigrid_levels/device`）
  - auto 模式下设备解析为全部 GPU（可通过 mock 验证）
  - 多 GPU 任务分发（可通过 mock 验证 device_id 分配）
- `commands/crop`:
  - `num_workers<=0` 自动解析为可用 CPU 数
- `all commands`:
  - 进度跟踪逻辑存在但默认静默（不输出进度条文本）
  - 触发错误时可在日志文件中看到错误记录
  - 日志与错误日志路径可由 YAML 显式指定，且默认落在 YAML 同目录
  - 启动时会输出接收到的 YAML 输入项
- `crop` / `reconstruction`:
  - 日志中可见周期性进度（如 `10/100, 20/100 ...`）
- `crop`:
  - 在同一 tomogram 多粒子场景下，不再每粒子重复读整卷
  - 单 tomogram + 多 worker 使用线程并行策略
- `scripts/generate_synthetic`:
  - 小规模数据集输出完整性（tomogram/subtomogram/classification）
- `reconstruction`:
  - 输出 `average.mrc` 的 `voxel_size` 与配置 `pixel_size` 一致

### T5. 执行与报告（P0）

- 使用 `conda run -n pydynamo pytest ...` 执行
- 记录：
  - 测试总数、通过数、跳过数、失败数
  - GPU 测试是否实际执行
  - 失败项（若有）与处理结果

---

## 3. 终止条件 / Definition of Done

必须同时满足以下条件：

1. 代码实现完成：
   - [ ] `core.align` 支持 CUDA 路径并可被测试调用
   - [ ] `core.align` 旋转重采样在 GPU 上执行（不依赖 CPU rotate）
   - [ ] `classification` 完整透传 `shift_search/multigrid_levels/device`
   - [ ] `alignment` 支持 `device_id`
   - [ ] `alignment` / `classification` 支持多 GPU 并行任务切分
   - [ ] `crop` 默认 `num_workers<=0` 自动使用全部 CPU
   - [ ] `alignment` / `classification` 默认 `device=auto` 使用全部可探测 GPU
   - [ ] 全部命令具备进度条能力
   - [ ] 全部命令支持错误日志文件输出
   - [ ] `log_file`/`error_log_file` 在 YAML 显式可配置
   - [ ] 未显式配置时日志默认输出到 YAML 同目录
   - [ ] `crop`/`reconstruction` 日志按 `progress_log_every` 周期输出进度（默认 10）
   - [ ] `crop` 已实现 mmap/非 copy + 按 tomogram 分组 + 单 tomogram 线程并行
   - [ ] `reconstruction` 输出 MRC 的 `voxel_size` 等于 YAML `pixel_size`
   - [ ] `alignment` / `classification` 对 STAR 相对 `rlnImageName` 可优先从 `subtomograms` 正确解析
   - [ ] `alignment` / `classification` / `reconstruction` 可通过 YAML `nmask` 启用 real-space mask 并参与计算
   - [ ] `crop` / `reconstruction` / `alignment` / `classification` / `gen_synthetic` 启动时会记录接收到的输入项
   - [ ] `alignment` 可在对齐后直接输出 `average.mrc`（支持 `output_average` 与 `average_symmetry`）
   - [ ] `classification` 可在迭代后直接输出最终平均体（支持 `output_average` / `output_average_dir` 与 `average_symmetry`）
2. 测试完成：
   - [ ] 新增测试文件已加入仓库并可被 pytest 发现
   - [ ] 在 `pydynamo` 环境执行全量测试一次（非 dry-run）
   - [ ] CPU 相关测试全部通过
   - [ ] 若 CUDA 可用，则 GPU 测试执行并通过；若不可用，需明确 skip 证据
   - [ ] 多 GPU 分发逻辑由测试验证（可使用 mock）
   - [ ] 进度与错误日志改动不破坏现有测试（全量通过）
3. 交付文档完成：
   - [ ] `response_005.md` 包含改动清单与测试报告
   - [ ] 报告中给出剩余风险与后续建议（如有）
   - [ ] 全部 `pydynamo/config/*.yaml` 并行配置显式化且变量注释完整

---

## 4. 非目标 / Non-goals

- 不在本轮引入全新算法（如完整 batched 旋转搜索重写）
- 不修改用户未要求的外部接口（CLI command 名称保持不变）
- 不做性能基准承诺（仅给出功能正确性与可执行性）

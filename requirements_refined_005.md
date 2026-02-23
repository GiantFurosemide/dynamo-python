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

### T4. 测试补全（P0）

新增/补强测试，覆盖：

- `core.align`:
  - CPU路径
  - CUDA路径（仅在 `torch.cuda.is_available()` 为 true 时执行）
  - auto 设备解析
  - 旋转阶段 GPU 化验证（GPU 路径不得回落到 CPU `rotate_volume`）
- `commands/alignment`:
  - 最小配置执行并输出 star
  - auto 模式下设备解析为全部 GPU（可通过 mock 验证）
  - 多 GPU 任务分发（可通过 mock 验证 device_id 分配）
- `commands/classification`:
  - 单迭代可运行并输出 `average_ref_*.mrc` / `refined_table_ref_*.tbl`
  - 参数透传验证（`shift_search/multigrid_levels/device`）
  - auto 模式下设备解析为全部 GPU（可通过 mock 验证）
  - 多 GPU 任务分发（可通过 mock 验证 device_id 分配）
- `commands/crop`:
  - `num_workers<=0` 自动解析为可用 CPU 数
- `scripts/generate_synthetic`:
  - 小规模数据集输出完整性（tomogram/subtomogram/classification）

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
2. 测试完成：
   - [ ] 新增测试文件已加入仓库并可被 pytest 发现
   - [ ] 在 `pydynamo` 环境执行全量测试一次（非 dry-run）
   - [ ] CPU 相关测试全部通过
   - [ ] 若 CUDA 可用，则 GPU 测试执行并通过；若不可用，需明确 skip 证据
   - [ ] 多 GPU 分发逻辑由测试验证（可使用 mock）
3. 交付文档完成：
   - [ ] `response_005.md` 包含改动清单与测试报告
   - [ ] 报告中给出剩余风险与后续建议（如有）
   - [ ] 全部 `pydynamo/config/*.yaml` 并行配置显式化且变量注释完整

---

## 4. 非目标 / Non-goals

- 不在本轮引入全新算法（如完整 batched 旋转搜索重写）
- 不修改用户未要求的外部接口（CLI command 名称保持不变）
- 不做性能基准承诺（仅给出功能正确性与可执行性）

# Response 005 — 进度与测试报告

**对应需求:** `requirements_refined_005.md`  
**执行日期:** 2026-02-23  
**环境:** `pydynamo` conda 环境 + CUDA 可用 GPU

---

## 1. 本轮已完成改动

### 1.1 Alignment GPU路径补全

- 文件: `pydynamo/src/pydynamo/core/align.py`
- 完成项:
  - 新增 GPU 单尺度搜索函数 `_align_single_scale_torch_gpu`
  - 新增零填充位移函数 `_shift_tensor_zero`（非循环 roll）
  - 新增 GPU 掩膜 NCC 计算 `_ncc_torch`
  - `align_one_particle()` 新增 `device_id` 参数，并传递到 CUDA 路径
  - CUDA 不可用时继续保持 CPU fallback
  - 旋转阶段进一步 GPU 化：新增 CUDA 3D 旋转插值（`torch.grid_sample`），GPU 路径不再调用 CPU `rotate_volume`

### 1.2 Classification 参数链路补全

- 文件: `pydynamo/src/pydynamo/commands/classification.py`
- 完成项:
  - 增加并透传配置参数: `shift_search`, `multigrid_levels`, `device`
  - 修正 `ref` 索引边界处理，避免越界
  - 修正单粒子时角度数组形状问题（`np.atleast_2d`）

### 1.3 Alignment 命令设备号支持

- 文件: `pydynamo/src/pydynamo/commands/alignment.py`
- 完成项:
  - 支持 `device_id` 配置并传入核心对齐函数
  - 新增设备解析逻辑：`device=auto` 且 CUDA 可用时默认使用全部可检测 GPU
  - 新增多 GPU 调度：按粒子任务切分到多个 device（ThreadPool + round-robin `device_id`）

### 1.4 I/O 稳健性修复

- 文件: `pydynamo/src/pydynamo/io/io_dynamo.py`
- 完成项:
  - `create_dynamo_table()` 中对 `convert_euler` 返回做 `np.atleast_2d`，修复 N=1 时一维返回导致的索引错误

### 1.5 Classification 多 GPU 调度与默认设备策略

- 文件: `pydynamo/src/pydynamo/commands/classification.py`
- 完成项:
  - 新增设备解析逻辑：`device=auto` 且 CUDA 可用时默认使用全部可检测 GPU
  - 新增多 GPU 调度：按粒子任务切分到多个 device（ThreadPool + round-robin `device_id`）
  - 保留 `device_id`/`gpu_ids` 显式覆盖行为

### 1.6 Crop 默认 CPU 并行策略

- 文件: `pydynamo/src/pydynamo/commands/crop.py`
- 完成项:
  - 新增 `_resolve_num_workers()`，当 `num_workers<=0` 时自动使用全部 CPU
  - 默认行为从“串行”改为“自动全 CPU 并行”

### 1.7 YAML 全量更新（显式并行配置 + 全变量注释）

- 目录: `pydynamo/config/*.yaml`（12 个文件）
- 完成项:
  - 所有 YAML 变量均补充注释（含含义/单位/取值）
  - 显式加入并行相关变量：
    - crop: `num_workers`
    - alignment/classification: `device`, `device_id`, `gpu_ids`
  - 明确默认策略：crop 默认全 CPU；alignment/classification 默认自动全 GPU（若可用）

### 1.8 Crop STAR 输出逻辑修复（用户反馈问题）

- 问题：`crop` 在 tbl 输入场景下把 Dynamo tbl 列（`tag/tdrot/x...`）直接写入 `.star`
- 修复文件：
  - `pydynamo/src/pydynamo/commands/crop.py`
  - `pydynamo/src/pydynamo/io/io_dynamo.py`
- 修复内容：
  - 新增 `_build_output_star_df()`，tbl 输入统一转换为 RELION 风格字段（`rln*`）
  - 输出 star 不再暴露原始 tbl 列作为主字段
  - 修复单粒子情况下 Euler 转换返回一维数组导致的索引错误（`np.atleast_2d`）

### 1.9 全命令进度条 + 错误日志文件输出

- 新增通用运行时工具：`pydynamo/src/pydynamo/runtime.py`
  - `configure_logging()`：统一日志初始化（stdout + 可选文件）
  - `progress_iter()`：统一终端进度条（无第三方依赖）
  - `write_error()`：统一错误追加写入日志文件
- 接入命令：
  - `crop`：串行/并行模式均显示进度条
  - `reconstruction`：粒子累积阶段显示进度条
  - `alignment`：单卡与多卡模式显示进度条
  - `classification`：每轮迭代显示进度条
  - `gen_synthetic`：embed/crop/classification 生成阶段显示进度条
- 错误日志输出策略：
  - 优先使用 CLI `--log-file`
  - 否则读取 YAML `error_log_file` 或 `log_file`
  - 保留 stderr 输出与 `--json-errors` 行为

### 1.10 Reconstruction 输出 voxel_size 修复（按 YAML `pixel_size`）

- 文件：`pydynamo/src/pydynamo/commands/reconstruction.py`
- 修复内容：
  - 在写出平均体 `average.mrc` 时，显式设置 `mrc.voxel_size = pixel_size`
  - 使输出 MRC 的尺度与配置文件 `pixel_size` 严格一致

---

## 2. 新增/更新测试

### 2.1 更新

- `pydynamo/test/test_align.py`
  - GPU 测试从固定 skip 改为 `torch.cuda.is_available()` 条件执行
  - 增加 `auto` 设备解析测试
  - 新增 `test_gpu_path_does_not_use_cpu_rotate`：验证 GPU 路径不会回退到 CPU 旋转实现

### 2.2 新增

- `pydynamo/test/test_alignment_command.py`
  - 覆盖 alignment 命令最小可运行路径与输出 star
  - 覆盖 auto 设备解析（CPU fallback / all GPU ids）
  - 覆盖 alignment 多 GPU 任务分发（mock 两张 GPU）
- `pydynamo/test/test_classification.py`
  - 覆盖 classification 单迭代输出
  - 覆盖 `shift_search/multigrid_levels/device` 参数透传
  - 覆盖 classification 在 auto 下的多 GPU 分发（mock 两张 GPU）
- `pydynamo/test/test_generate_synthetic.py`
  - 覆盖小规模 `gen_synthetic` 端到端输出完整性
- `pydynamo/test/test_crop.py`
  - 覆盖 `num_workers<=0` 自动解析为可用 CPU 数
  - 新增 `test_crop_tbl_outputs_relion_star`：验证 tbl 输入时输出为 RELION star 字段且不再含 `tdrot`
- `pydynamo/test/test_reconstruction_command.py`
  - 增加 voxel_size 断言：输出 `average.mrc` 的 `voxel_size.x` 必须等于配置 `pixel_size`
- 说明：
  - 本次“进度条 + 错误日志”属于运行时增强，不改变算法输出；现有测试集全量回归验证通过

---

## 3. 测试执行记录

执行命令（使用 pydynamo 环境解释器）：

```bash
/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q
```

结果：

- Collected: **28**
- Passed: **28**
- Failed: **0**
- Skipped: **0**
- 总耗时: **~8.90s**

关键结论：

- GPU 条件测试实际执行并通过（`test/test_align.py` 显示 `....`）
- 新增“GPU路径不使用CPU旋转”测试通过，确认旋转重采样阶段已GPU化
- 新增多 GPU 分发测试通过（alignment/classification，mock 2 GPU）
- 新增 crop 自动全 CPU 策略测试通过
- 新增 crop tbl->star 格式修复测试通过（输出 RELION 字段）
- 进度条与错误日志接入后全量回归通过，无回归失败
- reconstruction 输出 voxel_size 与配置 pixel_size 一致性测试通过
- command-level 与 synthetic 生成测试全部通过

---

## 4. 与终止条件对照

`requirements_refined_005.md` 终止条件状态：

1. 代码实现完成
   - [x] `core.align` CUDA 路径可执行并被测试调用
   - [x] `core.align` 旋转重采样在 GPU 执行（无 CPU rotate 依赖）
   - [x] `classification` 透传 `shift_search/multigrid_levels/device`
   - [x] `alignment` 支持 `device_id`
   - [x] `alignment` / `classification` 支持多 GPU 并行任务切分
   - [x] `crop` 默认 `num_workers<=0` 自动使用全部 CPU
   - [x] `alignment` / `classification` 默认 `device=auto` 使用全部可探测 GPU
   - [x] 全部命令具备进度条能力
   - [x] 全部命令支持错误日志文件输出
   - [x] `reconstruction` 输出 MRC 的 `voxel_size` 等于 YAML `pixel_size`
2. 测试完成
   - [x] 新增测试可被 pytest 发现
   - [x] 在 `pydynamo` 环境执行全量测试
   - [x] CPU 测试通过
   - [x] CUDA 可用条件下 GPU 测试已执行并通过
   - [x] 多 GPU 分发逻辑测试覆盖（mock）
   - [x] 进度与错误日志改动不破坏现有测试（全量通过）
3. 交付文档完成
   - [x] 输出 `response_005.md`
   - [x] 给出剩余风险与建议
   - [x] 全部 `pydynamo/config/*.yaml` 并行配置显式化且变量注释完整

---

## 5. 剩余风险与建议

1. 当前多 GPU 调度采用 Python 线程 + round-robin，后续可改为批处理/流式调度以进一步提升吞吐
2. GPU 旋转当前为逐角度执行，可进一步引入角度 batch 以减少 kernel launch 开销
3. 测试覆盖已增强，但仍建议补充真实多 GPU 机器上的端到端性能与稳定性压测（非 mock）

# Response 006 — 角度自由度与搜索细化复刻进度（实现中）

**对应需求:** `requirements_refined_006.md`  
**执行日期:** 2026-02-23  
**本轮范围:** 完成 T1 + T2 + T3(首版) + T4 的可运行实现，并完成 T5 回归

---

## 1. 本轮已完成实现

### 1.1 `core.align`（T1 + T2）

已在 `pydynamo/src/pydynamo/core/align.py` 完成以下改造：

1. **3DOF 角度搜索接入（tdrot/tilt/narot）**
   - 新增参数：`tdrot_step`, `tdrot_range`
   - CPU 与 GPU 路径均参与 `tdrot` 扫描，不再固定 `tdrot=0`
   - multigrid 粗细两层都按 3DOF 工作

2. **shift 约束模式（首批）**
   - 新增 `shift_mode`：
     - `cube`
     - `ellipsoid_center`
     - `ellipsoid_follow`
   - 当前版本中 `ellipsoid_*` 使用椭球半径约束生成候选平移集合

3. **subpixel 峰值细化**
   - 新增 `subpixel` 开关（默认 `true`）
   - 在整数峰值邻域进行 1D 抛物线细化，输出浮点 `dx/dy/dz`
   - 失败时自动回退整数峰值（通过数值保护实现）

4. **GPU 路径一致性修复**
   - 修复 `_align_single_scale_torch_gpu` 参数传递冲突（`tdrot_step` 重复传参）
   - 保持 GPU 路径不调用 CPU `rotate_volume`（既有测试覆盖）

### 1.2 命令透传（T4）

1. `pydynamo/src/pydynamo/commands/alignment.py`
2. `pydynamo/src/pydynamo/commands/classification.py`

均已透传并生效：

- `tdrot_step`
- `tdrot_range`
- `shift_mode`
- `subpixel`
- `cc_mode`

### 1.3 YAML 配置更新（T4）

已更新以下配置，新增参数并补注释：

- `pydynamo/config/alignment_defaults.yaml`
- `pydynamo/config/classification_defaults.yaml`
- `pydynamo/config/synthetic_align.yaml`
- `pydynamo/config/synthetic_align_quicktest.yaml`
- `pydynamo/config/synthetic_class.yaml`
- `muwang_test/alignment/alignment_defaults.yaml`
- `muwang_test/alignment/alignment_change_ori.yaml`
- `muwang_test/classification/classification_defaults.yaml`

### 1.4 相关性后端抽象（T3 增强）

已在 `pydynamo/src/pydynamo/core/align.py` 增加 `cc_mode`：

- `ncc`：原始全局归一化相关（默认）
- `roseman_local`：局部归一化相关近似实现（mask-aware 局部均值/方差归一化）

说明：

- CPU 路径与 GPU 路径均支持 `cc_mode` 参数；
- GPU 路径在 `ncc` 与 `roseman_local` 下均走 torch 后端（避免逐候选点 CPU 回落）。
- 新增可调参数：
  - `cc_local_window`（默认 `5`，奇数窗口）
  - `cc_local_eps`（默认 `1e-8`，方差稳定项）

---

## 2. 测试进展（T5）

### 2.1 新增/扩展测试点

1. `pydynamo/test/test_align.py`
   - 新增 `test_align_scans_tdrot_axis`：验证 `tdrot` 参与搜索
   - 新增 `test_align_subpixel_refines_shift`：验证 subpixel 输出非整数 shift
   - 新增 `test_align_supports_roseman_local_cc_mode`：验证 `cc_mode=roseman_local` 可运行
   - 新增 `test_align_roseman_local_accepts_window_and_eps`：验证局部相关窗口参数可配置

2. `pydynamo/test/test_alignment_command.py`
   - 扩展命令层参数透传检查（含 `tdrot_step/tdrot_range/shift_mode/subpixel/cc_mode/cc_local_*`）

3. `pydynamo/test/test_classification.py`
   - 扩展命令层参数透传检查（含 `tdrot_step/tdrot_range/shift_mode/subpixel/cc_mode/cc_local_*`）

### 2.2 本轮执行命令与结果

执行解释器（按用户指定）：

- `/home/muwang/miniforge3/envs/pydynamo/bin/python`

执行命令（focused）：

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_align.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py`

结果（focused）：

- **23 passed in 7.73s**

执行命令（full regression）：

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`（在 `pydynamo/` 目录）

结果（full regression）：

- **45 passed in 7.55s**

### 2.3 小型速度/精度对比（synthetic，CPU）

命令：基于 `align_one_particle` 的固定角度小样本脚本（同一解释器）。

- `ncc` 输出：
  - shift ≈ `(0.449, -0.351, 0.194)`
  - cc ≈ `0.930`
  - 平均耗时 ≈ `85.55 ms`
- `roseman_local` 输出：
  - shift ≈ `(0.013, -0.036, -0.001)`
  - cc ≈ `0.216`
  - 平均耗时 ≈ `224.21 ms`
  - 相对 `ncc` 约 `2.62x` 更慢

结论（当前实现阶段）：

- `roseman_local` 接口与流程已接通，可用于后续对齐复刻迭代；
- 但其数值行为与性能尚未达到 Dynamo 原生 `localnc` 预期，需继续做算法细节对齐与优化。

### 2.4 小型速度/精度对比（synthetic，CUDA）

命令：基于 `align_one_particle` 的固定角度小样本脚本（同一解释器，`device=cuda`）。

- `ncc` 输出：
  - shift ≈ `(0.448, -0.346, 0.195)`
  - cc ≈ `1.000`
  - 平均耗时 ≈ `24.07 ms`
- `roseman_local` 输出：
  - shift ≈ `(0.375, -0.357, 0.147)`
  - cc ≈ `0.228`
  - 平均耗时 ≈ `51.56 ms`
  - 相对 `ncc` 约 `2.14x` 更慢

对比上一版（GPU 回落 numpy）：

- 现已切换为 torch 本地计算，避免每次候选 shift 的 CPU<->GPU 往返；
- 在本 synthetic 样例上，`roseman_local` 速度与稳定性明显优于之前回落方案。

---

## 3. 与 DoD 对照（`requirements_refined_006.md`）

### 3.1 实现完成度

- [x] `core.align` 支持 3DOF 角度搜索（tdrot/tilt/narot）
- [x] 支持 subpixel 平移细化并输出浮点 shift
- [x] 支持受限 shift 搜索模式（`ellipsoid_*` 子集）
- [x] 支持 `cc_mode: ncc|roseman_local`（首版）
- [x] `alignment` / `classification` 配置透传完整
- [x] YAML 示例与注释更新完整

### 3.2 测试完成度

- [x] 新增针对 3DOF + subpixel 的测试
- [x] 关键命令测试通过
- [x] 全量 pytest 回归通过（45 passed）

### 3.3 文档交付

- [x] `response_006.md` 已更新本轮实现与测试进展

---

## 4. 尚未完成 / 下一步

1. **T3 后续增强（可选）**
   - `roseman_local` 当前已 mask-aware，但仍是近似实现，需继续向 Dynamo 原实现细节靠拢
2. **性能与默认参数收敛**待补：
   - 3DOF 搜索空间增大，需在真实数据上评估默认步长与速度


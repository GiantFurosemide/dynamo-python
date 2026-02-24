# Response 007 — cone/inplane 到 Euler 采样复刻进度

**对应需求:** `requirements_refined_007.md`  
**执行日期:** 2026-02-23  
**状态:** 已完成本轮实现与测试

---

## 1. 本轮完成内容

### 1.1 依据 Dynamo 源确认映射逻辑

已对照并确认以下链路：

- `dynamo-src/matlab/src/+dpkproject/pipeline_align_one_particle.m`
- `dynamo-src/matlab/src/dynamo_angleset.m`
- `dynamo-src/matlab/src/dynamo_angleincrement2list.m`

结论：

1. `cone_range` 在 Dynamo 中是 **aperture（锥角开口）**，不是绝对 tilt 区间。
2. `inplane_range` 是围绕 `narot_seed` 的对称扫描范围。
3. 采样以 `old_angles` 为 seed 生成 `tdrot/tilt/narot` 候选集合。

### 1.2 代码实现（已落地）

#### A) `core.align` 角度采样复刻

文件：`pydynamo/src/pydynamo/core/align.py`

新增与改造：

1. 新增 `_dynamo_angleincrement2list(...)` 采样器（Dynamo 语义）
2. 新增 `angle_sampling_mode` 参数：
   - `dynamo`：使用 cone/inplane aperture 采样
   - `legacy`：保留原绝对网格采样路径
3. 新增 `old_angles` 参数，作为 `dynamo` 采样 seed
4. CPU 与 GPU 单尺度搜索都支持 `dynamo` 采样
5. 在 Euler 反解处加入 gimbal-lock 受控处理（避免警告污染）

#### B) 命令层接线

文件：

- `pydynamo/src/pydynamo/commands/alignment.py`
- `pydynamo/src/pydynamo/commands/classification.py`

改造点：

1. 新增并透传 `angle_sampling_mode`（默认 `dynamo`）
2. 从输入 metadata 行提取 `tdrot/tilt/narot` 作为 `old_angles` seed（若缺失则回退 `0,0,0`）

#### C) YAML 配置更新

已更新默认/示例配置并加注释，新增：

- `angle_sampling_mode: dynamo`

涉及：

- `pydynamo/config/alignment_defaults.yaml`
- `pydynamo/config/classification_defaults.yaml`
- `pydynamo/config/synthetic_align.yaml`
- `pydynamo/config/synthetic_align_quicktest.yaml`
- `pydynamo/config/synthetic_class.yaml`
- `muwang_test/alignment/alignment_defaults.yaml`
- `muwang_test/alignment/alignment_change_ori.yaml`
- `muwang_test/classification/classification_defaults.yaml`

---

## 2. 测试结果

执行解释器（按用户指定）：

- `/home/muwang/miniforge3/envs/pydynamo/bin/python`

### 2.1 Focused tests

命令：

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_align.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py`

结果：

- **25 passed in 7.70s**

新增/扩展测试覆盖：

1. `test_dynamo_angleincrement2list_polar_limits`
2. `test_align_dynamo_mode_accepts_old_angles`
3. alignment/classification 透传测试新增 `angle_sampling_mode` 断言

### 2.2 Full regression

命令：

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`（在 `pydynamo/` 目录）

结果：

- **47 passed in 7.70s**

---

## 3. 与 DoD 对照（`requirements_refined_007.md`）

### 3.1 实现完成度

- [x] `core.align` 已支持 Dynamo cone/inplane 采样逻辑
- [x] 命令默认路径采用 `angle_sampling_mode: dynamo`
- [x] 命令可从输入行提取 `old_angles`
- [x] YAML 默认/示例更新完成并有注释

### 3.2 测试完成度

- [x] 新增采样逻辑与透传测试
- [x] focused tests 通过
- [x] 全量 pytest 通过

### 3.3 文档交付

- [x] `response_007.md` 已更新

---

## 4. 备注

1. 按你的最新指示，`roseman_local` 的进一步真实数据评估已暂缓。
2. 当前保留 `legacy` 采样模式用于兼容回退；命令默认使用 `dynamo` 采样模式。

---

## 5. 增量修复（2026-02-24）— `alignment` 输出 RELION STAR

### 5.1 问题复现

你反馈 `muwang_test/alignment/refined.star` 中混入了 Dynamo 内部列（如 `tag/tdrot/dx/...`），这会让输出不是严格的 RELION STAR。

### 5.2 已完成修改

文件：`pydynamo/src/pydynamo/commands/alignment.py`

1. 调整 `.star` 输出路径：不再直接 `starfile.write(out_df, ...)`。
2. 新增 `_build_output_star_df(...)`：
   - **tbl 输入**：调用 `dynamo_df_to_relion(...)`，显式转换为 RELION 字段，并补 `rlnImageName` / `rlnTomoParticleId`；
   - **star 输入**：保留 RELION 坐标/显微图字段，仅覆盖
     - `rlnAngleRot/Tilt/Psi`（由 `tdrot/tilt/narot` 转 ZYZ）
     - `rlnOriginX/Y/ZAngst`（由 `dx/dy/dz * pixel_size` 转换）。
3. 写 star 前过滤字段，仅保留 RELION 语义列，避免输出 `tdrot/dx/cc/ref` 等内部列。
4. 对齐结果行统一写入 `rlnImageName`，避免 tbl 输入路径缺失该列。

### 5.3 测试更新

文件：`pydynamo/test/test_alignment_command.py`

- 更新 `test_alignment_command_emits_output_star` 断言：
  - 期望包含 `rlnAngleRot`、`rlnOriginXAngst`
  - 期望不包含 `tdrot`、`dx`

执行：

- `/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_alignment_command.py`

结果：

- **8 passed**

### 5.4 当前结论

`alignment` 新输出的 `.star` 现在按 RELION 字段组织，不再把 tbl 内部列直接拼到 STAR。

---

## 6. 增量修复（2026-02-24）— 百万粒子内存优化

### 6.1 已完成改造

文件：`pydynamo/src/pydynamo/commands/alignment.py`

1. **移除全量体素预加载**
   - 旧：`tasks` 中持有所有粒子 `np.ndarray`
   - 新：`tasks` 仅保存路径与索引；在 worker 内即时读取体素
2. **平均重建改为在线累加**
   - 旧：依赖常驻粒子数组二次遍历
   - 新：每个粒子对齐后立即得到变换体素并累加到 `avg_acc`
3. **`tbl-only` 流式写出**
   - 条件：`output_table=.tbl` 且未设置 `output_star`
   - 行为：每条 refined 结果实时写入 `.tbl`，避免累计大 DataFrame
4. **保留“双输出”**
   - `output_table=.tbl` + `output_star=.star` 时：
     - `.tbl` 保存真实 refined 内部参数
     - `.star` 保存 RELION schema

### 6.2 测试增强

文件：`pydynamo/test/test_alignment_command.py`

新增/增强：

1. `test_alignment_star_output_stays_relion_schema_for_tbl_input`
2. `test_alignment_writes_tbl_and_star_when_both_configured`

验证：

- `pytest -q pydynamo/test/test_alignment_command.py` → **10 passed**
- `pytest -q pydynamo/test/test_align.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py` → **27 passed**

### 6.3 当前结论

`alignment` 在保持输出语义正确的前提下，已消除“全量体素常驻”这一主要内存瓶颈，并提供 `tbl-only` 低内存执行路径，适配百万级粒子场景。


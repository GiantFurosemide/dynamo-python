# Requirement 007 (Refined) — Dynamo cone/inplane 到 Euler 采样逻辑复刻

**Source:** `requirements_refined_006.md`, `response_006.md`, `docs/spec_v1.md`, `docs/system_map.md`, `dynamo-src/matlab/src/dynamo_angleset.m`, `dynamo-src/matlab/src/dynamo_angleincrement2list.m`, `dynamo-src/matlab/src/+dpkproject/pipeline_align_one_particle.m`  
**Goal:** 明确并复刻 Dynamo 中 `cone/inplane` 到 `tdrot/tilt/narot (ZXZ)` 的采样映射，替换当前简化角度网格在命令默认路径中的行为。

---

## 1. 背景与差距

### 1.1 Dynamo 源逻辑（确认）

1. `pipeline_align_one_particle.m` 使用：
   - `dynamo_angleset(...)`
2. `dynamo_angleset.m` 进一步调用：
   - `dynamo_angleincrement2list(...)`
3. `dynamo_angleincrement2list.m` 核心语义：
   - `cone_range` 是 **aperture**（锥角开口，非绝对 tilt 区间）
   - `cone_sampling` 是锥面轴向采样步长
   - `inplane_range` 是围绕 `narot_seed` 的对称范围
   - `inplane_sampling` 是 inplane 步长
   - 采样围绕 `old_angles`（上一轮姿态）展开

### 1.2 当前 pydynamo 差距

1. 现有实现以“绝对角度区间 + 笛卡尔网格”采样为主。
2. `cone/inplane` 未按 Dynamo aperture 语义驱动 Euler triplet 生成。
3. 命令默认路径未显式采用 Dynamo 采样器。

---

## 2. 本轮范围

### 2.1 必做（P0）

1. 在 `core.align` 增加 Dynamo 风格角度采样器（cone/inplane -> Euler triplets）。
2. 在 `alignment` / `classification` 命令层默认启用该采样逻辑。
3. 命令层从输入表（若存在）读取 `tdrot/tilt/narot` 作为 `old_angles` seed。
4. YAML 显式新增采样模式参数并加注释。
5. 补充测试并完成全量回归。

### 2.2 非目标（本轮不做）

1. 不在本轮继续推进 `roseman_local` 真实数据基准对比。
2. 不追求与 Dynamo 逐 bit 数值一致，仅追求采样逻辑语义一致。

---

## 3. 任务拆分

### T1. 采样逻辑复刻（P0）

- 新增 `core.align` 内部采样函数，复刻 `dynamo_angleincrement2list` 思路：
  - `cone_range` 作为 aperture；
  - `inplane_range` 围绕 seed 对称展开；
  - 基于 seed 组合旋转生成 `tdrot/tilt/narot`。

### T2. 对齐内核接线（P0）

- `align_one_particle` 增加：
  - `angle_sampling_mode: dynamo|legacy`
  - `old_angles`
- CPU/GPU 单尺度搜索都支持 `dynamo` 采样模式。

### T3. 命令层接线（P0）

- `alignment.py` / `classification.py`：
  - 透传 `angle_sampling_mode`
  - 从输入行读取 `old_angles` 作为 seed

### T4. 配置与示例（P0）

- 更新默认与示例 YAML：
  - `angle_sampling_mode: dynamo`（命令默认路径）
  - 注释说明 `dynamo` 与 `legacy` 差异

### T5. 测试与回归（P0）

- 单元测试：
  - Dynamo 采样器输出基本几何约束（tilt 范围）
  - `align_one_particle` 在 `dynamo` 模式可运行
- 命令测试：
  - `angle_sampling_mode` 参数透传
- 全量回归：
  - 全仓 pytest 通过

---

## 4. DoD / 终止条件

必须同时满足：

1. **实现完成**
   - [ ] `core.align` 已支持 Dynamo cone/inplane 采样逻辑
   - [ ] 命令默认路径采用 `angle_sampling_mode: dynamo`
   - [ ] 命令可从输入行提取 `old_angles`
   - [ ] YAML 默认/示例更新完成并有注释
2. **测试完成**
   - [ ] 新增采样逻辑与透传测试
   - [ ] focused tests 通过
   - [ ] 全量 pytest 通过
3. **文档交付**
   - [ ] 进度记录到 `response_007.md`

---

## 5. 风险与缓解

1. **角度表示奇异点（gimbal lock）**
   - 缓解：在 Euler 反解处加入受控处理，避免污染日志与流程。
2. **新旧语义并存导致行为混淆**
   - 缓解：显式 `angle_sampling_mode`，命令默认 `dynamo`，保留 `legacy` 兼容。
3. **采样密度变化导致运行时波动**
   - 缓解：保持 `legacy` 回退路径，逐步在真实数据上校准默认参数。

---

## 6. 补充修复（2026-02-24）— Alignment STAR 输出格式

### 6.1 问题描述

`alignment` 命令在输出 `.star` 时，曾直接将内部 Dynamo/tbl 字段（如 `tag/tdrot/dx/cc/ref`）与 RELION 字段一起写入，导致产物不是“纯 RELION 风格 STAR”。

### 6.2 目标行为（P0）

1. `alignment` 输出 `.star` 时必须输出 RELION 语义字段：
   - 坐标：`rlnCoordinate*` 或 `rlnCenteredCoordinate*`
   - 欧拉角：`rlnAngleRot/Tilt/Psi`（ZYZ）
   - 位移：`rlnOrigin*Angst`
   - 关联信息：`rlnImageName`（以及可选 `rlnMicrographName` / `rlnTomoParticleId`）
2. 不应把 `tdrot/tilt/narot/dx/dy/dz/cc/ref` 作为额外列直接写进 STAR。

### 6.3 实现要求

1. `STAR` 输入路径：
   - 保留输入中的 RELION 坐标/显微图字段；
   - 用对齐结果覆盖 `rlnAngle*` 与 `rlnOrigin*Angst`。
2. `TBL` 输入路径：
   - 通过 `dynamo_df_to_relion(...)` 做显式转换；
   - 再补充 `rlnImageName` 与 `rlnTomoParticleId`（若可得）。

### 6.4 验收（DoD 增补）

- [ ] `alignment` 输出 STAR 不含 `tdrot/dx/...` 内部列
- [ ] `alignment` 输出 STAR 含 `rlnAngleRot` 与 `rlnOriginXAngst`
- [ ] `test_alignment_command` 覆盖上述约束并通过

---

## 7. 补充修复（2026-02-24）— 大规模粒子内存策略（100 万+）

### 7.1 问题描述

在真实场景（`N=10^6` 级别粒子）中，旧版 `alignment` 存在峰值内存风险：

1. 预加载所有 subtomogram 到 `tasks`（体素数组常驻）
2. 依赖常驻体素做平均重建，放大 RSS
3. 缺少 `tbl-only` 低内存输出路径

### 7.2 目标行为（P0）

1. 粒子体素不再全量预加载：按任务即时读取、即时释放
2. 对齐平均使用在线累加（running accumulator），避免存储全体体素
3. 当 `output_table=.tbl` 且未请求 `output_star` 时，启用流式写表，避免累积所有行到内存

### 7.3 验收（DoD 增补）

- [ ] `alignment` 不再预加载全部粒子体素
- [ ] `alignment` 平均重建采用在线累加
- [ ] `tbl-only` 路径支持流式写出
- [ ] 相关回归测试通过（含输出格式与双输出行为）


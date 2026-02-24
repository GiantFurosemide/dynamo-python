# Requirement 006 (Refined) — Alignment 复刻增强：角度自由度 + 相关性/搜索细化

**Source:** `requirements_refined_005.md`, `response_005.md`, `docs/spec_v1.md`, `docs/code_location_002.md`, `dynamo-src/matlab/src/+dpkproject/pipeline_align_one_particle.m`, `dynamo-src/matlab/src/dynamo__align_motor.m`  
**Goal:** 面向 Dynamo 对齐核心行为复刻两项关键能力：  
1) 完整角度自由度搜索；2) 相关性计算与搜索细化（含 subpixel）能力补齐。

---

## 1. 背景与差距 / Background and Gap

### 1.1 Dynamo 侧参考路径（对齐主链）

- `dynamo-src/matlab/src/+dpkproject/pipeline_align_one_particle.m`
  - 负责单粒子对齐流程：预处理 -> 角度集 -> coarse/refine -> 输出表行
- `dynamo-src/matlab/src/dynamo__align_motor.m`
  - 负责角度循环、旋转模板/掩膜、CC 计算、受限 shift 搜索与峰值定位
- `dynamo-src/matlab/src/+dpkvol/+aux/+crossCorrelation/nativeCubeRoseman.m`
- `dynamo-src/matlab/src/+dpkvol/+aux/+crossCorrelation/+subpixel/volume3x3ToPeakBySpline1d.m`

### 1.2 当前 pydynamo 差距（需复刻）

1. **角度自由度不完整**
   - 当前 `pydynamo/src/pydynamo/core/align.py` 在搜索中固定 `tdrot=0`，仅扫描 `tilt/narot`。
   - 与 Dynamo `angles(i,1:3) -> tdrot, tilt, narot` 的完整三角自由度不一致。
2. **搜索细化能力不足**
   - 当前平移网格为整数体素（`dx,dy,dz` 整数网格）。
   - 无等价 `dynamo_peak_subpixel`/局部 3x3 亚像素峰值细化流程。
3. **相关性与约束策略差异**
   - Dynamo 支持 Roseman 局部归一化相关、受限 area-search（椭球/圆柱/跟随旧位移）。
   - 当前为简化 NCC + 立方体全搜索（`[-shift_search,+shift_search]^3`）。

---

## 2. 本轮范围 / Scope

### 2.1 必做（P0）

1. 角度自由度复刻：对齐搜索支持完整 `tdrot/tilt/narot` 三角扫描。
2. 相关性与搜索细化复刻：
   - 亚像素平移峰值细化（subpixel）。
   - 受限 shift 搜索策略（至少一个与 Dynamo 语义一致的 area-search 模式）。
3. YAML/CLI 显式化配置并向后兼容。
4. 增加测试与回归报告（含和旧行为的兼容说明）。

### 2.2 非目标（本轮不做）

- 不追求一次性完全重写为 Dynamo 内部所有分支（如全部 `area_search_modus` 与所有历史兼容分支）。
- 不承诺数值逐 bit 对齐，只追求功能语义与统计行为可比。

---

## 3. 任务拆分 / Task Breakdown

### T1. 角度自由度复刻（P0）

- 在 `core.align` 改造角度采样器：
  - 从当前 2DOF（tilt,narot）扩展到 3DOF（tdrot,tilt,narot）。
  - 新增 `tdrot_range`, `tdrot_step` 配置项（并保留旧配置兼容）。
- GPU 与 CPU 路径统一角度定义，避免 CPU/GPU 行为漂移。
- 保留 multigrid 机制，但在每层都按 3DOF 工作。

**参考：**
- `dynamo-src/matlab/src/dynamo__align_motor.m`（`angles(i,1:3)`）
- `dynamo-src/matlab/src/+dpkproject/pipeline_align_one_particle.m`（`dynamo_angleset` + refine）

### T2. 搜索细化与亚像素峰值（P0）

- 新增亚像素峰值定位流程：
  - 最小实现：在整数峰值邻域 3x3x3 上拟合/插值求 subpixel shift。
  - 输出 `dx/dy/dz` 可为浮点。
- 新增搜索约束模式（先实现一个稳定子集）：
  - `shift_mode: cube|ellipsoid_center|ellipsoid_follow`（命名可微调）。
  - 支持围绕原点或围绕上一轮 shift 的受限区域。
- 在日志中输出“整数峰值 vs 亚像素峰值”用于可观测性。

**参考：**
- `dynamo-src/matlab/src/dynamo__align_motor.m`（search area + `dynamo_peak_subpixel`）
- `dynamo-src/matlab/src/+dpkvol/+aux/+crossCorrelation/+subpixel/volume3x3ToPeakBySpline1d.m`

### T3. 相关性计算策略对齐（P1）

- 抽象相关性后端：
  - `cc_mode: ncc|roseman_local`
- 第一阶段保持默认 `ncc`，并提供 `roseman_local` 的可插拔实现入口。
- 明确 mask 处理和归一化域，减少与 Dynamo 语义偏差。

**参考：**
- `dynamo-src/matlab/src/dynamo__align_motor.m`（`localnc` 分支）
- `dynamo-src/matlab/src/+dpkvol/+aux/+crossCorrelation/nativeCubeRoseman.m`

### T4. 命令层与配置层接入（P0）

- `commands/alignment.py` / `commands/classification.py` 完整透传新增参数。
- 更新 YAML：
  - `pydynamo/config/alignment_defaults.yaml`
  - `pydynamo/config/classification_defaults.yaml`
  - synthetic 与 `muwang_test` 对应示例
- 每个新增参数配备注释（含单位、范围、默认行为）。

### T5. 测试与验收（P0）

- 单元测试：
  - 3DOF 角度扫描确实覆盖 `tdrot`。
  - subpixel 输出非整数且在受控样例上更接近真值。
  - shift 约束模式生效（限制区外峰值被抑制）。
- 命令测试：
  - alignment/classification 参数透传与输出字段正确。
- 回归：
  - 全量 pytest 通过。
  - 与旧默认参数下结果可兼容（误差在容许范围）。

---

## 4. DoD / 终止条件

必须同时满足：

1. **实现完成**
   - [ ] `core.align` 支持 3DOF 角度搜索（tdrot/tilt/narot）。
   - [ ] 支持 subpixel 平移细化并输出浮点 shift。
   - [ ] 支持至少 1 种受限 shift 搜索模式并可配置。
   - [ ] `alignment` / `classification` 配置透传完整。
   - [ ] YAML 示例与注释更新完整。
2. **测试完成**
   - [ ] 新增针对 3DOF + subpixel + shift 约束测试。
   - [ ] 关键命令测试通过。
   - [ ] 全量 pytest 通过。
3. **文档交付**
   - [ ] `response_006.md` 记录差距、方案、进度、风险与测试结果。

---

## 5. 风险与缓解

1. **计算量激增风险（3DOF）**
   - 缓解：默认更保守采样、multigrid 先粗后细、日志提示采样规模。
2. **CPU/GPU 路径结果不一致**
   - 缓解：统一角度网格、统一峰值细化策略、增加 CPU/GPU 一致性测试。
3. **数值稳定性**
   - 缓解：subpixel 邻域容错、异常回退整数峰值。


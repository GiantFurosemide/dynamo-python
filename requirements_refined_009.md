# Requirement 009 (Refined) — Crop 粒子编号从 6 位扩展到 12 位

**Source:** 支线需求（Crop Output 12-Digit Numbering Plan）  
**Goal:** 将 `crop` 产物命名从 `particle_000001.mrc` 统一升级为 `particle_000000000001.mrc`，并保证相关生成链路与测试契约一致。

---

## 1. 背景与问题

当前 `crop` 输出文件名采用 6 位零填充编号。为适配更大规模数据与统一命名规范，需要将粒子编号宽度升级为 12 位。

目标仅变更“编号宽度”，不改变以下语义：

1. 粒子编号来源（`tag`）不变
2. 文件名前缀（`particle_`）与扩展名（`.mrc`）不变
3. 结果排序与匹配逻辑保持兼容（同宽零填充下词典序与数值序一致）

---

## 2. 变更范围

### 2.1 核心命令路径（P0）

文件：`pydynamo/src/pydynamo/commands/crop.py`

1. 将输出文件命名从 `particle_{tag:06d}.mrc` 改为 `particle_{tag:012d}.mrc`
2. 保证写入 STAR 的 `rlnImageName` 与磁盘文件名保持一致

### 2.2 同类生成链路（P1）

文件：`pydynamo/src/pydynamo/scripts/generate_synthetic.py`

1. 同步更新 synthetic crop/classification 产物命名中的 6 位格式为 12 位
2. 包含 `particle_*` 与 `noise_*` 的对应输出与 STAR 行字段

### 2.3 回归测试（P0）

文件：`pydynamo/test/test_crop.py`

1. 增加强约束断言：`rlnImageName` 必须为 12 位编号格式（示例首粒子：`particle_000000000001.mrc`）

---

## 3. 非目标（Out of Scope）

1. 不强制修改所有历史文档中的 6 位示例（除本轮进度文档与需求文档）
2. 不修改与 crop 命名契约无关的测试夹具文件名
3. 不引入新配置项（如可配置编号宽度）

---

## 4. 验收标准（DoD）

1. **实现**
   - [x] `crop` 输出文件名已切换为 12 位编号
   - [x] synthetic 相关命名同步为 12 位编号
2. **测试**
   - [x] `test_crop.py` 包含 12 位命名回归断言并通过
   - [x] 受影响 focused tests 通过
3. **文档**
   - [x] `response_009.md` 记录变更明细、影响面与验证结果

---

## 5. 风险与缓解

1. **外部脚本依赖 6 位命名**
   - 缓解：在 `response_009.md` 显式声明本次行为变更与兼容性影响
2. **生成链路命名不一致**
   - 缓解：同一提交内同时更新 `crop` 与 synthetic 相关命名点

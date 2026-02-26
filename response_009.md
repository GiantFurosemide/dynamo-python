# Response 009 — Crop 12 位编号改造进度

**对应需求:** `requirements_refined_009.md`  
**执行日期:** 2026-02-23  
**状态:** 已完成

---

## 1. 本轮目标

将 crop 粒子文件名编号从 6 位零填充升级到 12 位零填充，并确保：

1. `rlnImageName` 与实际输出文件名一致
2. synthetic 相关输出命名与主流程保持一致
3. 测试具备回归约束，防止编号宽度回退

---

## 2. 已完成改造

### 2.1 Crop 主路径编号升级

文件：`pydynamo/src/pydynamo/commands/crop.py`

- 已将命名格式从 `particle_{tag:06d}.mrc` 修改为 `particle_{tag:012d}.mrc`。

### 2.2 Synthetic 生成链路命名对齐

文件：`pydynamo/src/pydynamo/scripts/generate_synthetic.py`

- 已将相关 `particle_*` 与 `noise_*` 命名由 `:06d` 统一更新为 `:012d`，覆盖：
  - subtomogram 落盘命名
  - synthetic STAR 中 `rlnImageName`
  - classification 数据集中的 real/noise 拷贝与记录

### 2.3 回归测试增强

文件：`pydynamo/test/test_crop.py`

- 在 `test_crop_tbl_outputs_relion_star` 中新增断言：
  - 首粒子 `rlnImageName == "particle_000000000001.mrc"`。

---

## 3. 兼容性说明

本次变更属于输出命名契约变更：

1. 旧格式：`particle_000001.mrc`
2. 新格式：`particle_000000000001.mrc`

如果外部脚本硬编码 6 位编号，需要同步更新匹配规则。

---

## 4. 测试与验证

### 4.1 Crop 回归测试

- 命令：`/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_crop.py`
- 结果：**9 passed**

### 4.2 Lint 检查

- 检查文件：
  - `pydynamo/src/pydynamo/commands/crop.py`
  - `pydynamo/src/pydynamo/scripts/generate_synthetic.py`
  - `pydynamo/test/test_crop.py`
  - `requirements_refined_009.md`
  - `response_009.md`
- 结果：**No linter errors found**

---

## 5. 当前结论

`crop` 12 位编号改造已完成并通过回归验证。当前输出命名规范为：

- `particle_000000000001.mrc`（12 位）

synthetic 相关产物命名已同步到 12 位，测试契约已加固，后续不应回退到 6 位编号。

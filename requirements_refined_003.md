# Requirement 003 (Refined) — Synthetic Data, YAML Config, Testing

**Source:** requirement_003.md  
**Previous:** requirement_002.md, response_002.md  

---

## 1. 目标与范围

在 requirement_002 基础上：
1. 将关键变量、默认值全部显式放入 YAML 配置
2. 生成 synthetic data 用于端到端测试
3. 实现 response_002 未完成功能（wedge/fcompensate, multigrid/GPU）
4. 完成一键启动与文档更新

---

## 2. 任务分解

### 2.1 完善 requirement_refined_003.md ✅

- 本文档即 refined 计划

### 2.2 YAML 配置显式化

- 所有命令的 config schema：必填项、可选项、默认值
- 默认值文件：`pydynamo/config/defaults.yaml` 或各命令的 `*_defaults.yaml`
- 代码中删除硬编码默认值，改为从 config 读取

### 2.3 Synthetic Data 生成

| 步骤 | 输出 | 说明 |
|------|------|------|
| 1 | `synthetic_data/out_tomograms/tomo.mrc` | 2000×2000×800 tomogram，pixel size 同 emd_32820_bin4.mrc |
| 2 | `synthetic_data/out_tomograms/*.tbl`, `.vll`, `.star` | 1000 粒子：随机 Euler、随机坐标 |
| 3 | 模板嵌入 | 模板 emd_32820_bin4.mrc（64×64×64），missing wedge ±48° |
| 4 | `synthetic_data/out_subtomograms/` | 用 crop 提取 1000 个 subtomogram |
| 5 | `synthetic_data/out_tomograms4classification/` | 1000 真实 + 1000 噪声 subtomogram，对应 tbl/vll/star |
| 6 | 后续测试 | 用该数据集测 crop、reconstruction、alignment、classification |

### 2.4 未完成功能实现

- **Reconstruction:** wedge 应用、fcompensate（Fourier 补偿）
- **Alignment:** multigrid 搜索、PyTorch/GPU 路径（本机无 GPU 时仅实现，不强制跑 GPU 测试）
- **Classification:** 确保 MRA 与 alignment 集成无误

### 2.5 测试与完善

- 用 synthetic data 跑 crop → reconstruction → alignment → classification 全流程
- 单元测试覆盖新增功能

### 2.6 一键启动与文档

- `run.sh` 或 `make` 目标：`gen_data`, `test_crop`, `test_recon`, `test_align`, `test_class`
- 更新 README、pyproject.toml、环境说明
- 输出 `response_003.md`

---

## 3. 输入

| 资源 | 路径 |
|------|------|
| 模板 MRC | `synthetic_data/emd_32820_bin4.mrc` |
| Synthetic 根目录 | `synthetic_data/` |

---

## 4. 输出

| 路径 | 描述 |
|------|------|
| requirement_refined_003.md | 本精炼需求 |
| pydynamo/config/*.yaml | 各命令默认配置 schema |
| synthetic_data/out_tomograms/ | tomogram + tbl/vll/star |
| synthetic_data/out_subtomograms/ | 提取的 subtomogram |
| synthetic_data/out_tomograms4classification/ | 真实+噪声 + tbl/vll/star |
| response_003.md | 本轮进度与问题 |

---

## 5. 终止条件

- [ ] YAML 修改完成（关键变量显式化）
- [ ] Synthetic data 生成完成
- [ ] 基于 NumPy + PyTorch 的核心实现与实例测试（GPU 测试可暂不跑）
- [ ] crop, reconstruction, alignment, classification 均支持一键启动

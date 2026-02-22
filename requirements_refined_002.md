# Requirement 002 (Refined) — pydynamo CLI Refactoring

**Document version:** 1.0 (Refined)  
**Source requirement:** requirement_002.md  
**Refined by:** Architect role

---

## 1. 任务需求 (Task Requirements)

### 1.1 目标 (Objective)

将主要基于 MATLAB 的 Dynamo 软件中的以下功能，重构为以 Python 为主的命令行工具：

1. **Particle crop**：从 tomogram 中根据 tbl/vll 提取 subtomogram  
2. **Reconstruction (average)**：根据 tbl 和 subtomogram 生成平均密度图  
3. **Subtomogram Averaging — Alignment**：对 subtomogram 进行 alignment  
4. **Subtomogram Averaging — Classification MRA**：多参考模板的 alignment 与 classification (Multireference Analysis)

### 1.2 目标划分 (Task Breakdown)

#### 1.2.0 完善本需求文档 ✅

- 输出：本 refined 文档 `requirements_refined_002.md`
- 完成后按 refined 文档执行

#### 1.2.1 阅读 Dynamo 源码并输出 System Map

**Input:** `dynamo-src` 目录（包含 matlab/src, C, dynamoForMatlab.tar 等）

**Output:** `docs/system_map/` 下的 System Map 文档

**必须包含：**

1. Entry Points（CLI/main/web handler 等）
2. Module/package graph（按领域分组）
3. Core data models/types 及其位置
4. Side effects inventory：filesystem, network, DB, subprocess, GPU, env vars
5. Config surface area：config 文件、flags、env vars
6. External dependencies（runtime + build）
7. Hotspots：最大文件、最复杂模块（粗略估计）
8. Suspected duplication：重复模式、相似函数
9. Test status：现有测试、运行方式、覆盖率估计
10. Unknowns list：未能推断的内容
11. 输出路径：`docs/system_map`（或 `docs/system_map.md`）

**规则：** 仅阅读与文档化，不 refactor，不改变任何行为。

#### 1.2.2 提炼可执行 Spec

**Input:** System Map 文档

**Output:** 可运行的 Spec（与实现无关、可测试）

**必须包含：**

1. Public interface contract：API/CLI 签名、必选参数、输入格式与校验、输出格式（含错误格式）
2. Core semantics：主要变换与不变量、状态机（如有）
3. Error handling rules：重试、fallback、致命错误
4. Performance expectations：大 O 级约束、latency/throughput（如有）
5. Observability：logs/metrics/traces
6. Parity Checklist：30–100 个 checkbox，按子系统分组

**规则：** 不涉及实现细节，每条规定必须可测试；未知项标 TBD 并说明如何发现。

#### 1.2.3 定位目标功能的代码

**四个功能对应源码定位：**

1. **Particle crop**：tbl + vll → 提取 subtomogram  
   - 关键模块：`dynamo_crop.m`、与 tbl/vll 解析相关的 I/O
2. **Reconstruction (average)**：tbl + subtomogram → 平均  
   - 关键模块：`dynamo_average.m` 及 variants、wedge 处理、FSC
3. **Alignment**：subtomogram 与 reference 的 alignment  
   - 关键模块：`dpkproject.pipeline_align_one_particle.m`、`dynamo_iteration_compute.m`、GPU binary
4. **MRA (Multireference Analysis)**：多 template alignment + classification  
   - 关键模块：`dynamo_plugin_post_multireference_tutorial.m`、`pipeline_embedded_multirerefence.m`、`dynamo_iteration_setup.m`、`dynamo_iteration_assemble.m`  
   - 参考：`docs.re_001/MRA_SPEC.md`

**Output:** 各功能的模块列表、关键文件路径、数据流概述（可在 System Map / Spec 中体现）

#### 1.2.4 重构为以 Python 为主的代码

- **形式：** CLI 命令行工具
- **输出目录：** `pydynamo/src`
- **决策原则：**
  - 核心算法与 Dynamo 保持一致（科学软件要求）
  - 使用 starfile 存储 particle 信息；tbl/vll ↔ star 转换严格参照 TomoPANDA-pick/utils（`tbl2star.py`、`io_dynamo.py`）
  - 体积 I/O 使用 mrcfile，输出 MRC 格式（Dynamo 内部用 .em，需转换）
  - 计算策略：轻量用 NumPy，重度用 PyTorch；自动探测硬件；支持多 GPU 并行

#### 1.2.5 全面测试

- **输出目录：** `pydynamo/test`
- **要求：**
  - 与 Dynamo 行为保持准确性、一致性
  - 每个子功能有具体 test 及 report
  - 最终可一键启动并运行测试

---

## 2. 输入 (Inputs)

| 资源 | 路径 / URL |
|------|------------|
| Dynamo 源码 | `dynamo-src` |
| Dynamo 主文档 | https://www.dynamo-em.org/w/index.php?title=Main_Page |
| Alignment 文档 | https://www.dynamo-em.org/w/index.php?title=Subtomogram_alignment |
| Classification MRA 文档 | https://www.dynamo-em.org/w/index.php?title=Classification |
| starfile | https://teamtomo.org/starfile/ |
| mrcfile | https://mrcfile.readthedocs.io/en/stable/usage_guide.html |
| TomoPANDA-pick 根目录 | `/Users/muwang/Documents/github/TomoPANDA-pick` |
| TomoPANDA-pick utils | `/Users/muwang/Documents/github/TomoPANDA-pick/utils` |
| TomoPANDA-pick notebooks | `/Users/muwang/Documents/github/TomoPANDA-pick/notebooks_template` |
| MRA Spec（已有） | `docs.re_001/MRA_SPEC.md` |

---

## 3. 限制与约束 (Constraints)

### 3.1 CLI 形式

```bash
pydynamo <command> --i config.yaml

# 示例
pydynamo crop       --i config.yaml
pydynamo reconstruction --i config.yaml
pydynamo alignment  --i config.yaml
pydynamo classification --i config.yaml
```

### 3.2 输出路径

| 类型 | 路径 |
|------|------|
| 源代码 | `pydynamo/src` |
| 文档 | `pydynamo/doc` |
| 测试及报告 | `pydynamo/test` |

### 3.3 技术约束

-  core 算法与 Dynamo 行为保持一致
- 计算：轻量用 NumPy，重度用 PyTorch；自动探测 GPU；支持多 GPU 并行
- 体积 I/O：使用 mrcfile，输出 MRC
- Particle 信息：使用 starfile；tbl/vll ↔ star 严格按 TomoPANDA-pick/utils 实现

---

## 4. 输出 (Deliverables)

1. **执行进度报告**：`response_002.md`  
2. **重构代码**：`pydynamo/src`  
3. **测试报告**：`pydynamo/test`，每项 test 有单独报告  

---

## 5. 终止条件 (Exit Criteria)

- 测试通过，与 Dynamo 保持准确性与一致性  
- 支持一键启动与运行  

---

## 6. 执行顺序 (Execution Order)

```
1.2.0 完善需求文档 → requirements_refined_002.md
1.2.1 生成 System Map → docs/system_map
1.2.2 提炼 Spec → 可运行 Spec 文档
1.2.3 定位目标功能代码 → 映射到 crop / reconstruction / alignment / MRA
1.2.4 重构 Python CLI → pydynamo/src
1.2.5 全面测试 → pydynamo/test, response_002.md
```

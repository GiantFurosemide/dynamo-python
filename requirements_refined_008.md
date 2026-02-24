# Requirement 008 (Refined) — 百万级粒子内存与吞吐架构优化计划

**Source:** `requirements_refined_007.md`, `response_007.md`, `docs/spec_v1.md`, `muwang_test` 真实测试反馈  
**Goal:** 在 `N=10^6+` 粒子规模下，将 `crop/reconstruction/alignment/classification` 从“可运行”提升到“可持续稳定运行（内存可控、吞吐可预期、失败可恢复）”。

---

## 1. 背景与核心约束

### 1.1 规模约束

1. 粒子数：100 万 ~ 1000 万
2. 体素边长：常见 64/96/128（单粒子体积内存占用显著）
3. 目标：避免 O(N) 级体素常驻内存

### 1.2 当前状态（基于 007 补丁）

1. `alignment` 已完成：
   - 体素按需读取（非全量预加载）
   - 在线平均累加
   - `tbl-only` 流式写表
2. 尚未系统化：
   - 全命令统一的流式 I/O 策略
   - 大规模执行指标（RSS/吞吐）与回归阈值
   - 中断恢复与断点续跑机制

---

## 2. 架构目标（Architecture Targets）

### 2.1 内存目标（P0）

1. 峰值 RSS 与 `N` 近似解耦（O(1)~O(workers)）
2. 单命令默认不保留全量粒子数组
3. 大结果集优先流式落盘，避免巨型 DataFrame 常驻

### 2.2 吞吐目标（P1）

1. I/O 与计算并行（可配置 worker 数）
2. 顺序路径与并行路径均具备稳定性能
3. 提供在 CPU/GPU 场景下可解释的性能日志

### 2.3 稳定性目标（P1）

1. 粒子级失败隔离（单粒子坏文件不拖垮全任务）
2. 可恢复执行（checkpoint / append-safe 输出）
3. 输出一致性校验（行数、字段、顺序策略）

---

## 3. 分阶段计划

### T1. Alignment 内存路径固化（P0）

1. 将当前流式策略沉淀为“默认行为文档 + 回归测试”
2. 增加超大任务 smoke 测试（mock 10^5 级 metadata，轻量体素）
3. 输出策略明确化：
   - `output_table=.tbl`：流式优先
   - `output_star`：按需启用

### T2. Reconstruction 流式化（P0）

1. 对 `reconstruction` 增加 chunk/batch 处理策略
2. 避免不必要的全量粒子列表驻留
3. 对 wedge/fcompensate 路径做等价流式改造

**本轮落地细化（2026-02-24）:**

- 非 wedge 路径已从 `particles_data` 全量列表改为在线累加（running sum + count）
- 进度日志与 `used` 计数改为基于在线计数 `n_acc`
- 输出语义保持不变（平均结果等价），但峰值内存显著下降

### T3. Classification 大规模路径（P1）

1. MRA 迭代中间结果分块持久化
2. Ref/task 维度分发策略优化（降低峰值内存）
3. 迭代级 checkpoint（ite 粒度恢复）

**本轮落地细化（2026-02-24）:**

- 每轮 `ite_xxx` 目录新增 `checkpoint.yaml`，支持按已完成 iteration 继续执行
- 新增 `resume` / `resume_from_iteration` 配置项
- classification 从“每 ref 全量 `particles_data` 列表平均”改为在线累加 `acc + used`
- 迭代结果先落盘到 `rows_ref_xxx.jsonl`，再按 ref 流式写 `refined_table_ref_xxx.tbl` 与平均体
- 峰值内存从“全轮结果 + 全 ref 粒子变换列表”下降到“单 ref 在线累加 + 行级流式处理”

### T4. 可观测性与基准（P1）

0. 进度日志增加时间可观测字段：`elapsed`、`eta`、`eta_at`
1. 增加内存日志：峰值 RSS、平均 RSS、任务阶段耗时
2. 增加性能基准脚本（small/medium/large profile）
3. 回归阈值：性能退化告警（相对基线）

**本轮落地细化（2026-02-24）:**

- `crop / reconstruction / alignment / classification` 的 progress 日志已统一增加：
  - `elapsed`（已耗时）
  - `eta`（按当前平均每粒子耗时估算的剩余时间）
  - `eta_at`（预计完成时间，wall-clock）
- 该能力同时写入 terminal 输出与 log 文件（沿用现有 logger）
- 在同一 progress 文本中新增内存指标：
  - `rss_cur`（当前 RSS）
  - `rss_avg`（该阶段采样平均 RSS）
  - `rss_peak`（进程峰值 RSS）
- 新增基准脚本：`pydynamo/src/pydynamo/scripts/benchmark_profiles.py`
  - 支持 `small/medium/large` profile
  - 支持 `--baseline` + `--max-slowdown-ratio` 性能退化阈值校验

---

## 4. 验收标准（DoD）

1. **实现**
   - [x] `alignment` 流式行为默认启用并文档化
   - [x] `reconstruction` 完成流式内存改造（非 wedge 路径）
   - [x] `classification` 增加迭代级中间结果持久化/恢复（checkpoint + resume）
2. **测试**
   - [x] 新增大规模路径回归测试（格式 + 内存行为）
   - [x] focused tests 通过
   - [x] 全量 pytest 通过
3. **观测**
   - [x] 关键命令输出进度耗时与预计完成时间（elapsed/eta/eta_at）
   - [x] 关键命令输出内存/吞吐指标（rss_cur/rss_avg/rss_peak + throughput）
   - [x] 提供基准对比结果（至少 2 组规模）

---

## 5. 风险与缓解

1. **流式写出引入顺序变化**
   - 缓解：明确定义顺序语义；必要时提供稳定排序开关
2. **I/O 成为瓶颈**
   - 缓解：增加预取/并行读取参数，默认保守
3. **测试成本上升**
   - 缓解：单元测试用 mock 体素，基准测试分层执行

# Response 008 — 百万级粒子内存优化（阶段进度）

**对应需求:** `requirements_refined_008.md`  
**执行日期:** 2026-02-24  
**状态:** T1/T2/T3/T4 全部完成

---

## 1. 本轮目标

根据你提出的真实规模场景（100 万+ 粒子），本轮先完成 `alignment` 的 P0 内存路径优化与回归固化，确保：

1. 不再全量预加载体素
2. 输出格式与真实 refined 参数同时可用
3. 测试流程可拦截格式/行为回归

---

## 2. 已完成改造

### 2.1 `alignment` 内存路径优化

文件：`pydynamo/src/pydynamo/commands/alignment.py`

1. **按需读取体素**
   - `tasks` 改为仅保存路径/索引
   - 每个 worker 内即时打开 `.mrc`，处理后释放
2. **在线平均累加**
   - 对齐完成后立即生成变换体素并累加到 `avg_acc`
   - 不再依赖“缓存全部粒子体素后再重建”
3. **`tbl-only` 流式输出**
   - 条件：`output_table=.tbl` 且未设置 `output_star`
   - 每条 refined 结果实时写入 tbl，避免大 DataFrame 常驻
4. **双输出语义**
   - `output_table=.tbl` + `output_star=.star`：
     - `.tbl` 输出真实 refined 内部参数
     - `.star` 输出 RELION schema

### 2.2 输出格式防回归

文件：`pydynamo/test/test_alignment_command.py`

新增/增强测试：

1. `test_alignment_star_output_stays_relion_schema_for_tbl_input`
2. `test_alignment_writes_tbl_and_star_when_both_configured`

覆盖点：

- STAR 不得混入 `tdrot/dx/...`
- 同时配置 `tbl + star` 时，两者都必须产出且字段语义正确

### 2.3 `classification`（T3）首轮落地

文件：`pydynamo/src/pydynamo/commands/classification.py`

1. **迭代级 checkpoint + resume**
   - 每轮输出目录 `ite_xxx/` 写入 `checkpoint.yaml`
   - 新增配置：
     - `resume`：自动从最新 completed checkpoint 续跑
     - `resume_from_iteration`：显式指定已完成 iteration 编号
2. **中间结果持久化**
   - 每个 ref 先写 `rows_ref_xxx.jsonl`（每粒子一行）
   - 后续按 ref 流式读取该文件进行表输出与平均构建
3. **内存降压**
   - 移除 `particles_data` 全量列表平均，改为在线累加 `acc + used`
   - tbl 改为流式行写，避免大 DataFrame 常驻

### 2.4 T4 可观测性与基准闭环

文件：`pydynamo/src/pydynamo/runtime.py`

1. **统一进度可观测字段（terminal + log）**
   - 时间：`elapsed` / `eta` / `eta_at`
   - 内存：`rss_cur` / `rss_avg` / `rss_peak`
2. **阶段耗时**
   - 按命令/迭代阶段通过 `elapsed` 持续输出，支持 long-run 在线观察
3. **基准与阈值**
   - 新增脚本：`pydynamo/src/pydynamo/scripts/benchmark_profiles.py`
   - 支持 `small/medium/large` profile
   - 支持 `--baseline` + `--max-slowdown-ratio` 退化阈值检查

---

## 3. 测试结果

执行解释器：

- `/home/muwang/miniforge3/envs/pydynamo/bin/python`

### 3.1 Alignment 命令测试

- 命令：`-m pytest -q pydynamo/test/test_alignment_command.py`
- 结果：**10 passed**

### 3.2 Reconstruction 命令测试

- 命令：`-m pytest -q pydynamo/test/test_reconstruction_command.py`
- 结果：**2 passed**

### 3.3 Focused 回归

- 命令：`-m pytest -q pydynamo/test/test_align.py pydynamo/test/test_alignment_command.py pydynamo/test/test_reconstruction_command.py pydynamo/test/test_classification.py`
- 结果：**29 passed**

### 3.4 Progress ETA 改造回归

- 命令：`pytest -q pydynamo/test/test_crop.py pydynamo/test/test_reconstruction_command.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py`
- 结果：**26 passed**

### 3.5 Classification T3 回归

- 命令：`pytest -q pydynamo/test/test_classification.py pydynamo/test/test_reconstruction_command.py pydynamo/test/test_alignment_command.py`
- 结果：**18 passed**
- 新增测试：
  - `test_classification_resume_from_checkpoint_runs_remaining_iterations_only`

### 3.6 Runtime/T4 回归

- 命令：`/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q pydynamo/test/test_runtime.py pydynamo/test/test_alignment_command.py pydynamo/test/test_classification.py pydynamo/test/test_reconstruction_command.py pydynamo/test/test_crop.py`
- 结果：**32 passed**
- 新增测试：
  - `test_progress_timing_text_includes_eta_and_rss_fields`
  - `test_alignment_tbl_streaming_handles_many_metadata_rows`

### 3.7 基准脚本验证

- 命令：`/home/muwang/miniforge3/envs/pydynamo/bin/python -m pydynamo.scripts.benchmark_profiles --profiles small,medium --output /tmp/pydynamo_bench_baseline.json`
- 输出：small + medium 两组 profile，包含 `elapsed_s`、`throughput_particles_per_s`、`peak_rss_mb`
- 回归阈值校验：
  - 命令：`... --baseline /tmp/pydynamo_bench_baseline.json --max-slowdown-ratio 0.5`
  - 结果：**pass（exit code 0）**

### 3.8 全量回归

- 命令：`/home/muwang/miniforge3/envs/pydynamo/bin/python -m pytest -q`
- 结果：**52 passed**

---

## 4. 对 008 任务映射

### T1. Alignment 内存路径固化（P0）

- [x] 按需读取体素
- [x] 在线平均累加
- [x] `tbl-only` 流式输出
- [x] 输出格式回归测试

### T2. Reconstruction 流式化（P0）

- [x] 非 wedge 路径移除 `particles_data` 全量驻留
- [x] 改为在线累加 `acc + n_acc`
- [x] 回归测试通过（`test_reconstruction_command.py` + focused suite）

### T3/T4

- [x] classification 迭代级持久化/恢复（checkpoint + resume）
- [x] 进度日志统一输出 `elapsed/eta/eta_at`
- [x] 统一内存与吞吐可观测指标（`rss_cur/rss_avg/rss_peak` + benchmark throughput）

---

## 5. 当前结论

`alignment` 与 `reconstruction` 已完成关键内存降压改造：

1. `alignment`：按需读取 + 在线累加 + `tbl-only` 流式落盘
2. `reconstruction`：非 wedge 路径在线累加，移除全量粒子列表驻留

在百万级粒子场景下，`alignment/reconstruction/classification` 的关键内存与可恢复路径已打通（按需读取、在线累加、迭代级 checkpoint/resume）；`crop/reconstruction/alignment/classification` 已统一输出时间与 RSS 指标；并提供了可执行的 profile benchmark + 退化阈值检查，008 目标已完成闭环。

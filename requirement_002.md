
# 1. 任务需求

## 1.1 目标

我想将主要基于Matlab写的dynamo软件的particle crop、 reconstruction （average）、Subtomogram Averaging 中的alignment、Subtomogram Averaging 中 classification 的 Multireference analysis (MRA) ，重构为python为主的命令行工具。

## 1.2 目标划分

### 1.2.0 完善本需求文档

```plaintext
你作为架构师，先要完善本需求文档，保存为requirements_refined_002.md。之后按照你完善后的需求文档进行运行实现。
```

### 1.2.1 阅读dynamo源码

```
Task:
Scan the repository {{dynamo_source_code_path}} and produce a “System Map”.

Output MUST include:
1) Entry points (CLI/main/web handler/etc.)
2) Module/package graph (group by domain)
3) Core data models/types and where they live
4) Side effects inventory: filesystem, network, DB, subprocess, GPU, env vars
5) Config surface area: config files, flags, env vars
6) External dependencies (runtime + build)
7) Hotspots: largest files, most complex modules (rough estimate ok)
8) Suspected duplication: repeated patterns, similar functions
9) Test status: existing tests, how to run, coverage guesses
10) “Unknowns” list: what you couldn’t infer
11) output doc in {{doc_system_map}} = "docs/system_map"

Rules:
- Do NOT refactor anything.
- Do NOT change behavior.
- If you need to read key files, list them and summarize each in <=5 lines.

Output as markdown with headings:
# System Map
## Entry Points
## Module Graph
...
## Unknown
```

### 1.2.2 提炼可执行 Spec

```plaintext
You are a specification engineer.

Using the System Map (paste below), write a runnable-ish specification that defines behavior without referring to source code.

Inputs you have:
{{doc_system_map}} 

Produce:
1) Public interface contract:
   - API/CLI signatures, required args/options
   - accepted input formats and validation rules
   - output formats (including error formats)
2) Core semantics:
   - primary transformations and invariants
   - state machine (if any)
3) Error handling rules:
   - retry, fallback, fatal errors
4) Performance expectations:
   - big-O-ish constraints, latency/throughput targets if implied
5) Observability:
   - logs/metrics/traces expected outputs (if any)
6) A “Parity Checklist” with 30-100 checkboxes grouped by subsystem.

Rules:
- No implementation detail.
- Every rule must be testable.
- If something is unknown, mark it as “TBD” and propose how to discover it.

Output:
# Spec v1
## Interface Contract
## Semantics
## Errors
## Performance
## Observability
## Parity Checklist
```

### 1.2.3  定位目的功能的代码

```plaintext
我主要需要重构四个功能：
1. particle crop: 输入包含颗粒信息的tbl 文件和包含tomogram信息的 vll 文件，提取颗粒的subtomogram.
2. reconstruction （average）: 给 tbl 文件 以及subtomogram，生成construct的
3. Subtomogram Averaging 中 的alignment: 对subtomogram进行alignment
4. Subtomogram Averaging 中 classification 的 Multireference analysis (MRA) ： 对subtomogram进行多template的alignment和classification
```

### 1.2.4 重构为python为主的代码

我不知道具体要求，你要根据我提供的信息决策

### 1.2.5 全面测试

我不知道具体要求，你要根据我提供的信息决策

## 1.3 输入

* dynamo source code path : {{dynamo_source_code_path}} = "dynamo-src"
* dynamo的官方文档网站：https://www.dynamo-em.org/w/index.php?title=Main_Page
* dynamo的 alignment的官方文档网站：https://www.dynamo-em.org/w/index.php?title=Subtomogram_alignment
* dynamo的 classification 的 Multireference analysis ： https://www.dynamo-em.org/w/index.php?title=Classification
* starfile：https://teamtomo.org/starfile/
* mrcfile：https://mrcfile.readthedocs.io/en/stable/usage_guide.html
* tomopanda-pick：{{tomopanda-pick_root}} = ”/Users/muwang/Documents/github/TomoPANDA-pick" , {{tomopanda-pick_util}} = "/Users/muwang/Documents/github/TomoPANDA-pick/utils", {{tomopanda-pick_notebooks}} ="/Users/muwang/Documents/github/TomoPANDA-pick/notebooks_template"

## 1.4 限制

* 我希望最终是是一个命令行工具 ，形式：

```bash
pydynamo <command> --i config.yaml

# for example 
pydynamo reconstruction --i config.yaml
pydynamo alignment --i config.yaml
pydynamo classification --i config.yaml
```

* 新的重构代码输出在 ： {{pydynamo_src}} = "pydynamo/src"
* 新的重构代码的文档输出在 ： {{pydynamo_doc}} = "pydynamo/doc"
* 新的重构代码的测试以及报告输出在 ： {{pydynamo_test}} = "pydynamo/test"
* 需要最终输出一个一键可用版本。
* 因为是科学软件，核心算法一定要与dynamo保持一致
* 关于矩阵运算，自动探测硬件。轻量化的计算用 numpy，重量运算基于pytorch。如可并行，要实现自动探测多GPU并运行。
* subtomogram的实现，dynamo内部是输出和转换 em文件。重构用mrc文件。使用 mrcfile 包操作。
* particle 的信息用 star文件存储，用starfile 包操作。关于tbl vll 和star 文件的转换关系，严格按照 {{tomopanda-pick_util}} 中的 tbl2star.py和io_dynamo.py等文件实现，使用案例在{{tomopanda-pick_notebooks}}

## 1.5 输出

* 执行进度的汇总报告，需生成并保存在response_002.md
* 重构代码
* 测试报告，按照上述步骤，具体每个test也需要生成具体报告并保存在上述的位置

## 1.6 终止条件

* 继续重构、测试，直到满足：完成了test保持准确性和一致性。以及，可以一键启动。

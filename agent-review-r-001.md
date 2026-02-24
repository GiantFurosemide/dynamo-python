# Algorithm Review Package Index (R-001) / 算法审阅包索引（R-001）

## Objective / 目标

- **EN:** Convert this review into a structured document package: first system maps, then detailed algorithm documents for **both** Dynamo and pydynamo, to support high-confidence parity correction.
- **中文：** 将本次审阅升级为结构化文档包：先建立 **双方** system map，再整理 **双方** 算法细节文档，用于后续高置信度修正。

## Output Location / 输出位置

- Main package folder: `agent-review-r-001/`
- This file is only the root index and task contract.

## Deliverables (Updated) / 交付物（已更新）

1. `agent-review-r-001/README.md`
   - Package navigation, reading order, and review contract.
2. `agent-review-r-001/system-map-dynamo.md`
   - Dynamo architecture map focused on crop/reconstruction/alignment/MRA algorithm pipeline.
3. `agent-review-r-001/system-map-pydynamo.md`
   - pydynamo architecture map for the same algorithm scope.
4. `agent-review-r-001/algorithm-dynamo.md`
   - Dynamo algorithm details (angle sampling, CC, shift constraints, multigrid, MRA flow, crop/recon behaviors).
5. `agent-review-r-001/algorithm-pydynamo.md`
   - pydynamo algorithm details aligned to the same decomposition.

## Mandatory Ordering / 强制顺序

1. Build system maps (`system-map-*.md`).
2. Build algorithm detail docs (`algorithm-*.md`).
3. Then enter next-round planning (`agent-review-r-002.md`).

## Quality Gates / 达标标准

- Every algorithm claim must be source-grounded (Dynamo + pydynamo).
- Document must separate:
  - **fact** (what code does),
  - **risk** (what may diverge),
  - **action direction** (what to verify/fix later).
- No code modifications are allowed in this task.

## Next Planning Artifact / 下一轮规划

- After this package is completed, produce `agent-review-r-002.md`:
  - refreshed algorithm improvement directions,
  - finer prioritization and execution plan,
  - inputs from project understanding + common algorithm practice + web research.


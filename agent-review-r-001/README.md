# Agent Review R-001 Package / 审阅包

## What this package contains / 包含内容

- `system-map-dynamo.md`: Dynamo algorithm-side system map.
- `system-map-pydynamo.md`: pydynamo algorithm-side system map.
- `algorithm-dynamo.md`: Dynamo algorithm detail breakdown.
- `algorithm-pydynamo.md`: pydynamo algorithm detail breakdown.
- `agent-review-r-002.md`: pointer to next-round plan file.

## Reading order / 阅读顺序

1. `system-map-dynamo.md`
2. `system-map-pydynamo.md`
3. `algorithm-dynamo.md`
4. `algorithm-pydynamo.md`

## Review contract / 审阅约束

- This package is **review-only**. No code changes are made.
- Focus scope:
  - crop
  - reconstruction / averaging
  - alignment
  - classification / MRA
- Every key statement is intended to be traceable to source files.

## Intended use / 用途

- Build high-confidence parity comparison between Dynamo and pydynamo.
- Provide a fact baseline before writing mismatch severity and correction roadmap in next artifacts.

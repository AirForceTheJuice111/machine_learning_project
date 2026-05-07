# Phase2 Report Supplement

## 1. Objective

Phase2 的目标从 phase1 的“硬件指标感知”切换为“LoRA 算子优化”。

目标算子为：

`Y = W X + A(B^T X)`

本阶段提交的重点不再是 `output.json`，而是一个由 Agent 在运行期间持续维护的、单文件、自包含、可编译的 `optimized_lora.cu`。

## 2. Agent Architecture

Phase2 在 phase1 框架基础上复用了：

- `AgentEngine`
- `ConversationMemory`
- `OpenAILLMClient`
- `ToolRegistry`
- 本地文件读写能力

在此基础上新增：

- `agent_framework/lora_optimize.py`
  - phase2 主编排入口
- `agent_framework/lora_harness.py`
  - correctness + benchmark 本地 harness
- `agent_framework/tools/lora_candidate_tools.py`
  - 候选评测与 best 晋升工具
- `agent_framework/lora_design_guidance.py`
  - phase2 搜索策略与停止条件 prompt
- `agent_framework/lora_candidate_templates.py`
  - seed candidate 模板与模板清单生成

## 3. Candidate Search Workflow

Phase2 采用“候选生成 -> 评测 -> 比较 -> 晋升”的真实 Agent 工作流：

1. 初始化一个可编译的 `optimized_lora.cu` baseline
2. 启动时先清空 `lora_workspace/candidates/`，然后重新写入 baseline seed 与模板 seed
3. 将新候选写入 `lora_workspace/candidates/`
4. 用 `evaluate_lora_candidate` 做 correctness + benchmark
5. 比较 `compile_ok`、`correctness_passed`、`mean_speedup`、`min_speedup`、`score`
6. 对更优候选执行 `promote_lora_candidate`
7. 将 best 持续保存在提交根目录 `optimized_lora.cu`

这样每一轮运行都从干净的候选目录开始，避免把上次搜索遗留的候选文件混入本轮实验。

## 4. Key Engineering Strategies

### 4.1 Correctness First

系统首先确保候选实现：

- 可编译
- 接口符合 `forward(W, X, A, B)`
- 通过 correctness 检查

未通过 correctness 的候选禁止晋升为 best。

### 4.2 Baseline Then Incremental Optimization

为了满足“始终维护可编译版本”的要求，系统会先放置一个基于 ATen GEMM 路径的稳定 baseline。

之后 Agent 再在此基础上：

- 调整访存路径
- 尝试分步与融合策略
- 尝试 tile / shared memory / register blocking
- 针对 `r=16` 做结构化优化

当前代码已经内置三份 seed 模板：

- `seed_aten_mm.cu`
- `seed_addmm_rank16.cu`
- `seed_lowrank_epilogue.cu`

并通过 `seed_manifest.json` 暴露给 Agent 读取。

### 4.3 Multi-Stage Evaluation

系统使用两级评测：

- `quick`：快速筛选候选
- `full`：收尾前严格验证

只有完成 `full` 验证的 best 才允许最终结束。

同时系统会根据报告中的 `score`、`mean_speedup`、`min_speedup` 自动辅助选择当前 best，降低模型在收尾阶段选错候选的概率。

### 4.4 Stable Finalization

phase2 completion checker 采用工程状态判定，而不是只信模型口头回答。

结束条件至少包括：

- 存在 `optimized_lora.cu`
- 比较过多个候选
- 已有 best promotion
- 当前 best correctness 通过
- 当前 best 完成 full 验证

若模型在收尾阶段输出不完整，系统会基于 `best_report.json` 自动合成最终 summary，减少无意义循环。

### 4.5 Best Persistence And Archiving

当前代码中的 best 管理机制已经从“只保留当前 best”扩展为“当前 best + 历史归档”两层结构：

- 当前活跃 best：
  - 根目录 `optimized_lora.cu`
  - `lora_workspace/best/optimized_lora_best.cu`
  - `lora_workspace/best/best_report.json`
- 每次 phase2 成功收尾后会自动归档：
  - `best_YYYYMMDD_HHMMSS_NNN_optimized_lora.cu`
  - `best_YYYYMMDD_HHMMSS_NNN_report.json`

此外，系统还支持集中索引：

- `lora_workspace/best/manifest.json`

`manifest.json` 会记录：

- `archive_id`
- `archived_at`
- 归档 `.cu` 路径
- 归档 report 路径
- 源候选路径
- `shape_preset`
- `compile_ok`
- `correctness_passed`
- `mean_speedup`
- `min_speedup`
- `score`

注意：

- 如果新代码下尚未再次完整跑完一次归档流程，`manifest.json` 可能暂时还不存在
- 但当前代码已经具备自动生成和更新它的能力

## 5. Model Interface

phase2 继续使用 OpenAI-compatible 接口：

- `API_KEY`
- `BASE_URL` / `OPENAI_BASE_URL`
- `BASE_MODEL` / `OPENAI_MODEL`

系统通过环境变量控制：

- `LLM_REQUEST_TIMEOUT_SECONDS`
- `PHASE2_MAX_RUNTIME_SECONDS`

这样可兼容本地开发和服务器评测环境。

## 6. Challenges And Solutions

### 6.1 Challenge: Official Requirement Requires A Single File

官方要求最终只读取 `./optimized_lora.cu`，不能依赖多个源码文件。

解决方案：

- 所有最终实现必须保持单文件
- 候选可以临时存放在 `lora_workspace/candidates/`
- 只有 best 才同步回根目录 `optimized_lora.cu`

### 6.2 Challenge: Need A Real Agent, Not A Hardcoded Kernel

课程要求明确禁止把最终答案硬编码到 Agent 中。

解决方案：

- prompt 明确要求真实候选搜索
- 工具层强制走“评测 -> 晋升”流程
- 停止条件依赖多候选比较与 full 验证

### 6.3 Challenge: Toolchain Instability On Windows

Windows 本地 phase2 在 PyTorch 扩展编译中容易遇到：

- Python 版本不稳定
- CUDA toolkit 与 PyTorch wheel 不匹配
- VS 版本过新导致 CCCL / Thrust / CUB 报错

解决方案：

- 将 Linux 服务器作为主测试环境
- Windows 主要用于开发、静态检查和轻量 smoke test
- 保持 phase2 架构对环境变量和路径配置尽量稳定

### 6.4 Challenge: Ending Cleanly Without Endless Iteration

优化任务天然容易让 Agent 无限试验。

解决方案：

- 明确候选搜索协议
- 强化 prompt 中的“何时停止”
- completion checker 只在工程状态满足时结束
- 允许系统基于 best report 自动收尾

## 7. Current Status

目前 phase2 已完成：

- 主入口
- 本地 harness
- 候选评测工具
- best 晋升工具
- best 历史归档与 `manifest.json` 索引能力
- 候选搜索 prompt
- 停止条件与收尾逻辑
- `run.sh` 提交入口切换

当前 best report 显示：

- `shape_preset = full`
- `compile_ok = true`
- `correctness_passed = true`
- `mean_speedup = 1.0220705710460947`
- `min_speedup = 1.0198499675094959`

这说明当前工程已经具备一套完整、可运行、可归档的 phase2 骨架，但当前 best 仍主要是基于 ATen 调用方式优化得到的版本，性能提升幅度还不大，后续重点仍是继续向更强的自定义 CUDA kernel 演进。

当前下一阶段的重点是：

- 在 Linux 环境中继续真实测试
- 改善候选实现质量
- 推进从 ATen baseline 向更高性能自定义 CUDA kernel 演进


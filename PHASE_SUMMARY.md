# Phase1 And Phase2 Summary

## 1. Project Overview

本项目是一个基于大模型与本地工具调用的 CUDA Agent 框架，当前同时保留两条并存链路：

- **Phase1**：GPU 硬件指标 profiling Agent
- **Phase2**：LoRA 算子优化 Agent

项目根目录当前使用 `/workspace`。在后续协作中，**不要误删 phase1**，因为 phase1 与 phase2 共享一部分底层 Agent 基础设施。

## 2. Phase1 Summary

### 2.1 Goal

Phase1 的目标是：

- 读取 `target_spec.json`
- 根据 target 列表让 Agent 自主生成 CUDA micro-benchmark
- 编译、运行、必要时用 `ncu` profiling
- 输出结构化结果

Phase1 的约束是：

- 禁止外部 benchmark
- 禁止直接查规格表
- 测量必须基于 Agent 在本地生成的 `.cu`

### 2.2 Implemented Files

- `agent_framework/evaluate.py`
- `agent_framework/target_design_guidance.py`
- `agent_framework/tools/cuda_probe_tools.py`
- `agent_framework/core/engine.py`
- `agent_framework/main.py`

### 2.3 Current Status

- 已实现 family-based 调度
- 支持共享 benchmark、多 `mode/program_args`、单编译多运行与结果拆分
- 结果校验已覆盖 family 格式、benchmark 复用、`ncu` 执行与机器可解析输出提取
- 主链路可运行，但新指标的 family 映射与 throughput 类 target 设计仍可继续细化

## 3. Phase2 Summary

### 3.1 Goal

Phase2 不再测硬件指标，而是围绕 LoRA 算子

`Y = W X + A(B^T X)`

构建一个真正的 LoRA CUDA 优化 Agent。

要求：

- 生成候选 `optimized_lora.cu`
- 编译与 correctness 检查
- benchmark 与候选比较
- 将当前最佳版本持续保存在提交根目录 `optimized_lora.cu`

### 3.2 Main Files

- `agent_framework/lora_optimize.py`
- `agent_framework/lora_harness.py`
- `agent_framework/lora_design_guidance.py`
- `agent_framework/lora_candidate_templates.py`
- `agent_framework/tools/lora_candidate_tools.py`
- `run.sh`

### 3.3 Implemented Engineering Workflow

当前代码已经实现了完整的 phase2 工程骨架：

- phase2 主入口为 `agent_framework/lora_optimize.py`
- 启动时会创建 `lora_workspace/{candidates,logs,best,build}`
- 启动时会**先清空本轮的 `lora_workspace/candidates/`**
- 然后重新写入 `seed_baseline.cu` 与三份 seed 模板
- Agent 按“生成候选 -> 评测 -> 比较 -> 晋升”的流程工作
- quick 预设用于快速筛选，full 预设用于最终收尾验证
- completion checker 只在工程状态满足时允许结束

### 3.4 Harness And Evaluation

`agent_framework/lora_harness.py` 已实现：

- synthetic input 生成
- PyTorch reference：
  - `W @ X + A @ (B.transpose(0,1).contiguous() @ X)`
- 使用 `torch.utils.cpp_extension.load(...)` 编译单文件 `.cu`
- correctness 检查
- benchmark：CUDA Event + median

`evaluate_lora_candidate` 返回的关键字段包括：

- `compile_ok`
- `correctness_passed`
- `mean_speedup`
- `min_speedup`
- `score`
- `decision_hint`

### 3.5 Best Promotion And Persistence

当前 best 管理机制包括两层：

- 当前活跃 best：
  - 根目录 `optimized_lora.cu`
  - `lora_workspace/best/optimized_lora_best.cu`
  - `lora_workspace/best/best_report.json`
- 历史归档：
  - 每次 phase2 正常结束或 timeout recovery 成功收尾时，会把当次 best 归档到 `lora_workspace/best/`
  - 文件名格式为：
    - `best_YYYYMMDD_HHMMSS_NNN_optimized_lora.cu`
    - `best_YYYYMMDD_HHMMSS_NNN_report.json`
- 归档索引：
  - `lora_workspace/best/manifest.json`
  - 用于集中记录所有归档 best 的时间、路径与关键指标

注意：

- 如果 phase2 尚未再次完成一次归档流程，`manifest.json` 可能暂时还不存在
- 但当前代码已经具备自动生成与更新 `manifest.json` 的能力

### 3.6 Current Best Result

当前保留在 `lora_workspace/best/` 的 best_report 显示：

- `shape_preset = full`
- `compile_ok = true`
- `correctness_passed = true`
- `mean_speedup = 1.0220705710460947`
- `min_speedup = 1.0198499675094959`

这说明当前 best 是一个**稳定正确、但性能提升仍然较小**的实现。

从代码看，当前 best 仍然主要是 ATen 路径优化，而不是彻底重写的大型自定义 CUDA kernel。

### 3.7 Current Workspace State

为了开始下一轮更干净的实验，当前已经清理掉上一轮运行产物：

- 已删除 `lora_workspace/logs/`
- 已删除 `lora_workspace/candidates/`
- 已删除 `lora_workspace/build/`

保留内容：

- `lora_workspace/best/`
- 根目录 `optimized_lora.cu`

因此，下一次运行 phase2 时会重新创建工作目录并重新生成候选模板。

## 4. Environment Notes

### 4.1 Linux

- **Linux 是 phase2 主测试环境**
- 建议在 Linux 上完成真实 benchmark 与提交前验证

### 4.2 Windows

Windows 当前更适合：

- 写代码
- 做语法检查
- 做轻量 smoke test

Windows 上 phase2 真正编译扩展时，已知风险包括：

- Python 版本不稳定
- Windows Store Python 不适合扩展编译
- CUDA toolkit 与 PyTorch wheel 不匹配
- VS 与 CUDA 组合可能触发 `CCCL / thrust / cub` 编译错误

## 5. Current Risks

### 5.1 Phase1

- 新指标仍可能落到 `generic_family`
- family 校验与 target 映射仍可继续细化

### 5.2 Phase2

- 当前 best 仍偏向 ATen 组合优化，未进入更强的自定义 low-rank kernel 路线
- 本地 Windows phase2 编译环境不稳定
- 候选 prompt 与评测策略后续仍可继续打磨

## 6. Recommended Next Steps

### 6.1 For Phase1

- 按需继续补 target family 映射
- 改善 generic_family 的细分策略

### 6.2 For Phase2

- 以 Linux 为主环境继续调试
- 优先保留当前骨架，不破坏现有 phase1/phase2 双链路
- 在现有正确基线之上，继续推进：
  - 更稳的 low-rank epilogue kernel
  - 针对 `r=16` 的结构化优化
  - 减少转置与中间张量开销
  - 从 ATen baseline 逐步过渡到更强的自定义 CUDA kernel

## 7. Which Document To Give An LLM

如果你只给大模型**一份文档**，优先给：

- `PHASE_SUMMARY.md`

因为它同时说明：

- phase1 和 phase2 的关系
- 当前主入口与关键文件
- 当前代码已经实现到什么程度
- 当前工作区状态与 best 管理机制

如果你要让模型**专注接手 phase2**，建议给：

- `PHASE2.md`

它更适合 phase2 定向开发、调试和优化接力。

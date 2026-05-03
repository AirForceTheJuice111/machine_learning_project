# Phase1 And Phase2 Summary

## 1. Project Overview

本项目是一个基于大模型与本地工具调用的 CUDA Agent 框架，目标分为两个阶段：

- **Phase1**：GPU 硬件指标感知与 profiling Agent
- **Phase2**：LoRA 算子优化 Agent

项目根目录是 `machine_learning_project/`。当前代码库已经同时保留了 phase1 与 phase2 两条能力链路。

## 2. Phase1 Goal And Current Implementation

### 2.1 Phase1 Goal

Phase1 的目标是：

- 读取 `target_spec.json`
- 根据 target 列表让 Agent 自主生成 CUDA micro-benchmark
- 编译、运行、必要时用 `ncu` profiling
- 输出结构化结果

Phase1 强调：

- 禁止外部 benchmark
- 禁止直接查规格表
- 测量必须基于 Agent 自主生成的本地 `.cu`

### 2.2 Phase1 Implemented Files

- `agent_framework/evaluate.py`
- `agent_framework/target_design_guidance.py`
- `agent_framework/tools/cuda_probe_tools.py`
- `agent_framework/core/engine.py`
- `agent_framework/main.py`

### 2.3 Phase1 Key Implementations

- `evaluate.py` 已实现 family-based 调度
- 同一 family 可共享一份 benchmark，并通过多 `mode/program_args` 复用同一 binary
- 支持单编译、多运行、拆回多个 target 结果
- 结果校验包含：
  - family 输出格式校验
  - 共享 benchmark 复用判定
  - `ncu` 执行判定
  - 机器可解析结果行抽取
- `cuda_probe_tools.py` 已支持：
  - 只允许项目内 `generated_cuda/` 下的 `.cu`
  - `skip_compile=true`
  - Windows 下 `ncu/nvcc` 路径兼容
  - `-allow-unsupported-compiler` 自动重试

### 2.4 Phase1 Current Status

- Phase1 主链路可运行
- 本地曾成功生成 `results.json`
- 但新 target 的 family 映射仍可能需要继续细化
- 某些 throughput / counter 类 target 仍可能因 benchmark 设计不稳定而结果波动

## 3. Phase2 Goal And Current Implementation

### 3.1 Phase2 Goal

Phase2 的目标不是再测硬件指标，而是构建一个真正的 LoRA CUDA 优化 Agent，针对算子：

`Y = W X + A(B^T X)`

要求：

- 生成候选 `optimized_lora.cu`
- 编译并测试
- 做 correctness 检查
- 做 benchmark
- 比较多个候选
- 持续将当前最佳版本保存在提交根目录 `optimized_lora.cu`

### 3.2 Phase2 Implemented Files

- `agent_framework/lora_optimize.py`
- `agent_framework/lora_harness.py`
- `agent_framework/lora_design_guidance.py`
- `agent_framework/tools/lora_candidate_tools.py`
- `run.sh`

### 3.3 Phase2 Key Implementations

#### a. Candidate Search Strategy

已实现你要求的候选搜索策略：

- 先评估当前 `optimized_lora.cu` 基线
- 候选统一写入 `lora_workspace/candidates/`
- 每个候选生成后立即评测，不堆积未评测候选
- 先用 `quick` 预设筛选，再用 `full` 预设做收尾验证

此外，phase2 第三轮增强已加入：

- 更具体的 LoRA 优化 playbook
- 更明确的候选命名与逐步变更协议
- 启动时自动写入若干 seed candidate 模板，降低模型从零起步失败率

#### b. Local Harness

`lora_harness.py` 已实现：

- 合成输入生成
- PyTorch reference：
  - `W @ X + A @ (B.transpose(0,1) @ X)`
- `torch.utils.cpp_extension.load(...)` 编译单文件 `.cu`
- correctness 检查
- benchmark（CUDA Event + median）

#### c. Baseline Template

`starter_optimized_lora_source()` 已提供一个更稳的 baseline：

- 单文件
- 自包含
- 导出 `forward(W, X, A, B)`
- `PYBIND11_MODULE(...)`
- 当前基线以 ATen GEMM 路径优先，先保证 correctness，再供 Agent 迭代替换

同时新增了预置模板清单：

- `seed_aten_mm.cu`
- `seed_addmm_rank16.cu`
- `seed_lowrank_epilogue.cu`

这些模板统一写入 `lora_workspace/candidates/`，并通过 `seed_manifest.json` 暴露给 Agent 读取。

#### d. Best Comparison And Promotion

`evaluate_lora_candidate` 返回：

- `compile_ok`
- `correctness_passed`
- `mean_speedup`
- `min_speedup`
- `score`
- `decision_hint`

`promote_lora_candidate` 已增强为：

- 禁止晋升未通过编译的候选
- 禁止晋升未通过 correctness 的候选
- 将 best 同步到：
  - 根目录 `optimized_lora.cu`
  - `lora_workspace/best/optimized_lora_best.cu`
  - `lora_workspace/best/best_report.json`

#### e. Stopping Conditions

`lora_optimize.py` 已实现 phase2 completion checker，只有在以下条件满足时才允许结束：

- 根目录存在 `optimized_lora.cu`
- 至少比较过 2 个候选
- 至少一次晋升 best
- 当前 best 编译成功
- 当前 best correctness 通过
- 当前 best 至少完成一次 `full` 预设验证

#### f. Stable Finalization

为减少模型收尾不稳定导致的空转，系统已支持：

- 若工程状态已满足结束条件，提示模型只输出最终 JSON
- 若模型最终输出不完整，系统可基于 best report 合成 fallback summary

### 3.4 Phase2 Current Status

- Phase2 第一版骨架已经完成
- prompt、候选搜索、停止条件、本地 harness、目录规划已落地
- `run.sh` 已切到 phase2 入口
- 当前 phase2 主要瓶颈不在框架，而在本地 Windows 工具链兼容性

## 4. Environment Notes

### 4.1 Linux

Phase2 官方环境更接近 Linux，因此：

- **Linux 是 phase2 主测试环境**
- 建议在 Linux 上完成真实 benchmark 与提交前验证

### 4.2 Windows

Windows 当前主要适合：

- 开发代码
- 做语法检查
- 做部分轻量 smoke test

Windows 上 phase2 真正编译扩展时，当前已知风险包括：

- Python 版本不稳定
- Windows Store Python 不适合扩展编译
- CUDA toolkit 与 PyTorch wheel 不匹配
- VS 2026 / CUDA 13.2 组合容易触发 `CCCL / thrust / cub` 编译错误

## 5. Current Known Risks

### 5.1 Phase1 Risks

- 某些新指标仍会落到 `generic_family`
- family 校验与 target 映射仍可继续细化
- throughput 类 target 可能需要更针对性的 benchmark 设计

### 5.2 Phase2 Risks

- 本地 Windows phase2 环境不稳定
- baseline 目前以 correctness 为主，性能尚不是最终版本
- 候选 prompt 虽已加强，但后续仍可能继续需要微调
- LoRA 高性能自定义 kernel 还需要后续继续迭代

## 6. Recommended Next Steps

### 6.1 For Phase1

- 按需继续补 target family 映射
- 改善 generic_family 的细分策略
- 对不稳定 target 增加更明确约束

### 6.2 For Phase2

- 以 Linux 为主环境继续调试
- 先确保 baseline + candidate + promotion + full 验证完整跑通
- 再逐步引导 Agent 从 ATen baseline 走向更强的自定义 CUDA kernel
- 针对 `r=16` 与 `d in [3584, 4608]` 做更有针对性的 tile/fusion 优化

## 7. Suggested Prompt For Next Conversation

如果要开启新对话并让模型快速接手，建议先贴这段说明：

> 这是一个分为 phase1 和 phase2 的 CUDA Agent 项目。  
> phase1 是 GPU 硬件指标 profiling Agent，入口是 `agent_framework/evaluate.py`，已实现 family-based 调度、共享 benchmark、多 mode、多次运行与结果拆分。  
> phase2 是 LoRA 算子优化 Agent，入口是 `agent_framework/lora_optimize.py`，已实现 local harness、candidate evaluation、promotion、completion checker、best report 持久化和 `run.sh` 提交入口。  
> 当前 phase2 第一版工程骨架已完成，但高性能 kernel 仍需继续优化；Linux 是主测试环境，Windows 主要用于开发和静态检查。  
> 请基于当前代码继续协助：优先保持现有 phase1/phase2 架构不被破坏，再在 phase2 上做候选 prompt、baseline、best 比较逻辑或 LoRA kernel 优化。

## 8. Final Note

当前项目状态可以概括为：

- **Phase1：可运行、可继续细化**
- **Phase2：骨架已成、下一步重点转向性能优化与 Linux 环境验证**

新对话中应优先告诉模型：

- phase1 和 phase2 是两条并存能力链路
- 不要误删 phase1
- phase2 当前的主入口是 `agent_framework/lora_optimize.py`
- phase2 的关键约束是：
  - 单文件 `optimized_lora.cu`
  - correctness first
  - candidate search
  - full validation before stop

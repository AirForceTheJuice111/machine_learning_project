# Phase2 Workflow

## 1. Phase2 现在在做什么

Phase2 的目标是围绕算子：

`Y = W X + A(B^T X)`

构建一个真实的优化 Agent，而不是提交一份写死的最终 CUDA 代码。

重构后的系统遵循以下基本原则：

- 根目录始终维护一个可编译的 `optimized_lora.cu`
- Agent 必须真实地产生候选、评测候选、比较候选，再决定是否晋升 best
- 优化重点优先放在 rank=16 的低秩路径，而不是一开始就重写整个 `W @ X`
- 停止条件由工程状态决定，不再只依赖模型“口头说完成了”

## 2. 启动链路

评测入口仍然是：

```bash
bash run.sh
```

启动后实际链路如下：

1. `run.sh`
   - 检查 `openai`、`rich`、`torch`
   - 设置 `PYTHONPATH`
   - 启动 `python3 -m agent_framework.lora_optimize`
2. `agent_framework/lora_optimize.py`
   - 创建 `lora_workspace/{candidates,logs,best,build}`
   - 确保根目录存在一个可编译的 `optimized_lora.cu`
   - 刷新 framework 自带的 seed 模板与 `seed_manifest.json`
   - 启动 LLM + ToolRegistry + completion checker 的优化循环
3. `agent_framework/tools/lora_candidate_tools.py`
   - 负责候选生成、修复、评测、晋升
4. `agent_framework/lora_harness.py`
   - 负责编译单文件 CUDA 扩展
   - 跑 correctness
   - 跑 benchmark
   - 输出评测报告 JSON

## 3. 这次重构改了什么

这次重构重点不是“再塞更多功能”，而是把 Phase2 变回一个清晰、可维护的系统：

- `agent_framework/lora_optimize.py`
  - 只保留编排、收尾、completion checker 主逻辑
- `agent_framework/lora_artifacts.py`
  - 专门管理目录布局、best 固化、归档、starter candidate
- `agent_framework/lora_search_policy.py`
  - 专门管理 best 选择、停止条件、generation prompt 覆盖度判断
- `agent_framework/tools/lora_candidate_tools.py`
  - 强化生成源码的静态校验
  - 允许优先通过 `prompt_id` 选内置 prompt，而不是把整段 prompt 文本手工塞给工具
- `agent_framework/lora_candidate_templates.py`
  - 把过于激进、不现实的 prompt 改成更可收敛的低秩路径优化 prompt
- `agent_framework/lora_design_guidance.py`
  - 把主 prompt 改成更像“执行协议”，减少无意义的大而空指令

此外还有两个关键行为变化：

- Phase2 默认不再注册 Phase1 的 `compile_and_run_cuda_source`
  - 只有显式设置 `PHASE2_ENABLE_PHASE1_PROBE=1` 才启用
  - 这样可以避免 Agent 在 Phase2 里跑偏到 Phase1 的 `generated_cuda/` 路线
- Phase2 默认不再在启动时清空旧的 `candidates/` 和 `logs/`
  - 这样更利于复盘、调试和人工检查搜索过程

## 4. Agent 会读哪些文件

### 4.1 启动阶段必读

- 根目录 `optimized_lora.cu`
  - 当前 baseline
- `lora_workspace/candidates/seed_manifest.json`
  - 模板索引和 generation prompt 索引

### 4.2 搜索阶段常读

Agent 在搜索过程中通常会继续读取这些文件：

- `lora_workspace/candidates/seed_baseline.cu`
  - 当前 baseline 的候选副本
- `lora_workspace/candidates/seed_aten_mm.cu`
  - 最稳的 ATen baseline
- `lora_workspace/candidates/seed_addmm_rank16.cu`
  - `W@X` 走 ATen，低秩路径走 `addmm`
- `lora_workspace/candidates/seed_lowrank_epilogue.cu`
  - `W@X` 走 ATen，低秩累加交给自定义 kernel
- 某个已有候选 `cand_*.cu`
  - 当 Agent 想继续修复或微调已有候选时会读取
- 最新评测报告 `lora_workspace/logs/*.json`
  - 用来决定某个候选是否值得继续修复或晋升
- `lora_workspace/best/best_report.json`
  - 用来和当前候选比较

## 5. Agent 会产生哪些 CUDA 代码

Phase2 现在会稳定地产生 3 类 CUDA 代码。

### 5.1 基线与模板候选

这些文件由系统直接生成或刷新：

- `optimized_lora.cu`
  - 提交根目录的当前 best
- `lora_workspace/candidates/seed_baseline.cu`
  - `optimized_lora.cu` 的镜像副本
- `lora_workspace/candidates/seed_aten_mm.cu`
  - 完整走 ATen 的 correctness-first baseline
- `lora_workspace/candidates/seed_addmm_rank16.cu`
  - `WX` 用 `at::mm`，低秩项走 `addmm`
- `lora_workspace/candidates/seed_lowrank_epilogue.cu`
  - `WX` 用 `at::mm`，最终低秩加和交给自定义 CUDA kernel

### 5.2 基于 generation prompt 生成的新候选

现在内置的 generation prompt 不是要求模型“一步写出超大 fused GEMM”，而是要求它生成更现实、可收敛的候选：

- `epilogue_rank16`
  - 保持 `W @ X` 在 ATen 上
  - 优化 rank=16 的最终 epilogue kernel
- `lowrank_outer`
  - 把低秩更新写成 16 个 outer product 的结构化累加
- `split_btx`
  - 只对 `BTX` 或 `BTX` 的消费路径做定制，不碰完整主 GEMM

这些候选会写成类似：

- `lora_workspace/candidates/cand_v3_epilogue_rank16_*.cu`
- `lora_workspace/candidates/cand_v4_lowrank_outer_*.cu`
- `lora_workspace/candidates/cand_v5_split_btx_*.cu`

### 5.3 基于失败信息修复出的候选

当某个候选：

- 编译失败
- correctness 失败
- 速度没有优势

Agent 会调用 `revise_candidate`，它会读取原始候选源码，再结合错误/瓶颈描述，写出新的修订候选，例如：

- `cand_v4_lowrank_outer_rev1_*.cu`
- `cand_v4_lowrank_outer_rev2_*.cu`

## 6. Agent 如何判断一个结果好不好

判断标准统一来自 `evaluate_lora_candidate` 返回的 report。

### 6.1 先看是否可用

第一层只看两件事：

- `compile_ok`
- `correctness_passed`

如果其中任意一个失败，这个候选就不能晋升为 best。

### 6.2 再看性能

如果候选已经正确，再看：

- `mean_speedup`
- `min_speedup`
- `mean_student_ms`
- `score`

其中：

- `mean_speedup`
  - 多个 shape 的平均加速比
- `min_speedup`
  - 多个 shape 里的最差加速比
- `score`
  - 一个用于排序的综合分数
  - 正确候选优先
  - `full` 评测优先于 `quick`
  - 然后再综合 speedup 和耗时

### 6.3 quick 和 full 怎么用

- `quick`
  - 快速筛选候选
  - 适合在搜索阶段高频使用
- `full`
  - 收尾验证
  - 当前 best 在结束前必须至少完成一次 `full`

所以一个候选真正能成为最终提交，至少要满足：

- `compile_ok = true`
- `correctness_passed = true`
- `shape_preset = full`

## 7. Agent 一轮完整优化是怎么跑的

一轮推荐流程如下：

1. 读取当前 `optimized_lora.cu`
2. 对它执行一次 `quick` 评测，得到基线 report
3. 读取 `seed_manifest.json`
4. 选一个模板候选或一个内置 `prompt_id`
5. 生成一个新的 `cand_*.cu`
6. 立刻调用 `evaluate_lora_candidate`
7. 如果失败：
   - 读取失败 report
   - 调用 `revise_candidate`
   - 再次评测
8. 如果成功：
   - 与当前 best report 比较
   - 只有更优时才调用 `promote_lora_candidate`
9. 当 best 基本稳定后：
   - 对当前 best 补一次 `full`
10. 满足结束条件后：
   - 写出 `final_summary.json`
   - 归档 best 到 `lora_workspace/best/`

## 8. Agent 如何决定继续优化还是停止

重构后，停止条件不再只写死在 prompt 里，而是由代码中的 search policy 统一判断。

主要看这些条件：

- 是否已经比较过至少 2 个候选
- 当前 best 是否已经通过 correctness
- 当前 best 是否已经完成 `full`
- 是否已经尝试完内置 generation prompts
- 是否已经连续多轮没有显著提升

默认还有一个“建议目标”：

- `PHASE2_TARGET_SPEEDUP`
  - 默认值为 `1.10`
  - 它是搜索建议值，不是 correctness 之上的硬门槛

也就是说：

- 如果已经正确、已完成 `full`、尝试覆盖也足够、并且搜索已经明显停滞
- 那么系统会倾向于结束，而不是为了追一个不现实的 speedup 无止境空转

## 9. 如何本地跑通

最小运行方式：

```bash
export API_KEY=你的密钥
export BASE_URL=你的兼容接口地址
export BASE_MODEL=你要用的模型
bash run.sh
```

可选环境变量：

```bash
export PHASE2_MAX_RUNTIME_SECONDS=1740
export PHASE2_TARGET_SPEEDUP=1.10
export PHASE2_ENABLE_PHASE1_PROBE=0
```

运行后重点看这些产物：

- 根目录 `optimized_lora.cu`
- `lora_workspace/logs/*.json`
- `lora_workspace/logs/final_summary.json`
- `lora_workspace/best/best_report.json`
- `lora_workspace/best/manifest.json`

## 10. 当前推荐的理解方式

如果你下次要继续改 Phase2，建议按下面的顺序看代码：

1. `run.sh`
2. `agent_framework/lora_optimize.py`
3. `agent_framework/lora_design_guidance.py`
4. `agent_framework/lora_search_policy.py`
5. `agent_framework/tools/lora_candidate_tools.py`
6. `agent_framework/lora_harness.py`
7. `agent_framework/lora_candidate_templates.py`
8. `optimized_lora.cu`

按照这个顺序读，你能比较快地看明白：

- Agent 为什么会产生某个候选
- 它为什么会读某个 `.cu`
- 它依据什么报告判断候选是好是坏
- 它在什么条件下晋升 best
- 它在什么条件下停止

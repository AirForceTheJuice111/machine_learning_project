# GPU Hardware Profiling Agent Report

## 1. Agent Overall Architecture

本项目构建了一个面向 GPU 硬件指标测量的本地 Agent。其核心目标是：读取 `target_spec.json` 中给定的目标指标，利用大模型自主生成 CUDA micro-benchmark，随后在本地完成编译、运行、性能分析与结果汇总，并最终输出结构化测量结果。

整体架构可分为以下几层：

### 1.1 Interaction and Entry Layer

- `agent_framework/main.py` 提供交互式入口，负责读取 `API_KEY`、`BASE_MODEL`、`BASE_URL` 等环境变量，并初始化 Agent 运行所需的 LLM 客户端、记忆模块、工具注册表和执行引擎。
- `agent_framework/evaluate.py` 提供自动评测入口，用于批量读取 `target_spec.json` 中的目标指标，并按照评测模式自动运行 Agent，最终将结果输出为 `output.json`。

### 1.2 Planning and Execution Layer

- `agent_framework/core/engine.py` 实现了 completion-driven 的 ReAct 主循环。
- 与传统固定轮数对话不同，本项目采用“完成条件 + 时间预算”的控制方式：每轮由模型决定是否调用工具，工具执行完毕后根据当前观察结果判断是否已满足结束条件；若超出时间预算则安全退出，避免无限循环。
- 这种设计兼顾了 Agent 的自主性与工程可控性。

### 1.3 Memory Layer

- `agent_framework/core/memory.py` 维护 system prompt、user/assistant 消息以及 tool observation。
- 所有工具返回结果都会写回对话记忆，使模型在后续轮次中能够基于编译输出、运行结果和 profiling 信息继续推理，而不是“无状态”地重复生成。

### 1.4 Tool Layer

本项目主要使用三类工具：

- `read_file`：读取本地文件，帮助 Agent 理解已有代码或输入文件。
- `write_file`：将 Agent 生成的 CUDA 源码写入本地。
- `compile_and_run_cuda_source`：编译 Agent 自主生成的 `.cu` 文件，并可执行程序运行和 `ncu` profiling。

其中，`compile_and_run_cuda_source` 是整个系统的核心工具。它负责：

- 限制源码和二进制必须位于项目根目录下的 `generated_cuda/` 中，避免使用外部 benchmark。
- 调用 `nvcc` 编译生成的 CUDA 程序。
- 运行生成的程序并收集标准输出。
- 可选调用 `ncu` 对当前二进制做 profiling。
- 支持 `skip_compile=true`，从而允许“编译一次，运行多次”的复用模式。

## 2. Key Innovations and Optimization Strategies

### 2.1 Target-Specific Prompting and Design Guidance

项目没有把指标测量硬编码为固定 benchmark，而是通过 `target_design_guidance.py` 向模型注入“设计约束”而不是“代码模板”。

例如：

- latency 类指标要求使用 pointer chasing 和工作集控制；
- bandwidth 类指标要求使用大数组 streaming benchmark；
- shared memory / bank conflict 类指标要求构造可归因的访问模式。

这种做法的优点是：

- 保留了 Agent 自主设计 benchmark 的能力；
- 避免了直接查询规格表或硬编码答案；
- 对新 target 具有一定泛化能力。

### 2.2 Family-Based Batch Scheduling

本项目后期最重要的结构优化是从“逐 target 单独评测”升级为“按测量 family 批处理，共享 benchmark”的方案。

具体做法是：

- 在 `target_design_guidance.py` 中引入 `TARGET_FAMILY_MAP`、`FAMILY_DESIGN_CONSTRAINTS`、`group_targets_by_family()`。
- 在 `evaluate.py` 中先对 targets 做 family 分组，再按 family 启动 Agent。
- 要求同一 family 生成一份共享 `.cu`，并通过不同 `mode` 或 `program_args` 覆盖多个 target。
- 首次运行完成编译，后续运行通过 `skip_compile=true` 复用已有 binary。
- 最终再把 family 级结果拆回各个 target。

这一优化显著减少了：

- LLM 会话次数；
- CUDA 编译次数；
- 同类指标重复生成近似 benchmark 的开销。

同时，它仍保留了“每个 target 可以有独立测量逻辑”的优点，因为不同 target 可以通过不同运行模式共享同一份 probe。

### 2.3 Completion-Driven Stopping Instead of Fixed Iteration Count

项目没有采用简单的固定轮数上限，而是通过 `CompletionCheckResult` 驱动终止条件。评测模式下，只有满足以下条件后才允许结束：

- 已得到目标结果；
- 已完成至少一次有效 `ncu` profiling；
- 输出格式满足结构化 JSON 要求；
- 在 family 模式下，还要求共享 benchmark、多次运行与结果拆分逻辑成立。

此外，系统还设置了总时间预算，防止 Agent 因错误推理或接口异常而无限循环。

### 2.4 Error Recovery and Safe Failure Handling

项目对工程错误做了多层容错：

- LLM 请求失败时，`engine.py` 会安全退出，而不是整个进程崩溃。
- `llm_client.py` 对 LLM 请求增加了显式超时与连接失败提示，避免长时间无响应。
- `evaluate.py` 会把失败原因写入输出文件，而不是静默失败。
- 对 Windows 下常见的工具链兼容问题做了特殊处理，例如 `ncu.bat` 路径解析、`nvcc` 对新版本 MSVC 的 `-allow-unsupported-compiler` 自动重试。

## 3. Model Interface and Invocation Method

本项目采用 OpenAI-compatible 接口调用大模型，封装在 `agent_framework/core/llm_client.py` 中。

主要调用方式如下：

- 使用 `OpenAI(api_key=..., base_url=..., timeout=...)` 初始化客户端；
- 调用 `chat.completions.create(...)` 发送消息与工具 schema；
- 使用 `tool_choice="auto"` 让模型自主决定何时调用工具；
- 通过 `temperature` 控制生成稳定性；
- 通过环境变量 `LLM_REQUEST_TIMEOUT_SECONDS` 控制单次请求超时时间。

环境变量设计兼容了多种部署方式：

- `API_KEY`
- `BASE_MODEL` 或 `OPENAI_MODEL`
- `BASE_URL` 或 `OPENAI_BASE_URL`

这样既支持本地开发环境，也支持服务器提交环境。

## 4. Challenges and Solutions

### 4.1 Challenge: Windows Toolchain Compatibility

在 Windows 本地调试时，`nvcc` 依赖 MSVC 的 `cl.exe`，而 `ncu` 常常以 `ncu.bat` 的形式存在于 PATH 中。直接调用时，Agent 很容易出现：

- 找不到 `cl.exe`
- 找不到 `ncu`
- `unsupported Microsoft Visual Studio version`

对应解决方案：

- 在工具层使用 `shutil.which()` 和多后缀解析来兼容 `.exe/.bat/.cmd`。
- 对 `unsupported compiler` 错误自动追加 `-allow-unsupported-compiler` 重试。
- 在文档和启动流程中明确要求先加载 Visual Studio 的开发环境。

### 4.2 Challenge: Avoiding External Benchmark Dependence

课程要求严禁使用外部 benchmark 或直接查询硬件规格，因此不能简单调用现成工具得到答案。

解决方案：

- 在 system prompt 和 tool policy 中显式禁止外部 benchmark、下载行为和第三方资源。
- 强制所有测量基于 Agent 自主生成的本地 CUDA 代码完成。
- 编译工具只允许访问项目内 `generated_cuda/` 下的 `.cu` 文件和对应二进制。

### 4.3 Challenge: Long Runtime and Excessive Compilation

逐 target 生成、逐 target 编译虽然简单，但在 target 较多时会导致总耗时过高。

解决方案：

- 引入 family-based scheduling。
- 用共享 benchmark + 多 mode 运行替代多个独立 benchmark。
- 通过 `skip_compile=true` 实现 binary 复用。

这是本项目从“能工作”走向“更可提交、更可扩展”的关键优化。

### 4.4 Challenge: LLM Instability and Non-Structured Output

大模型在多轮工具调用后，可能出现：

- 不输出最终 JSON；
- 输出格式不完整；
- 已完成实验但没有显式收尾。

解决方案：

- 在 prompt 中明确要求结构化输出。
- 在 completion checker 中校验必要字段。
- 在 `evaluate.py` 中实现 fallback extraction：若模型未显式收尾，但工具输出中已包含可识别结果，则自动提取并合成结果。

### 4.5 Challenge: Server Submission Reliability

在服务器评测环境中，常见问题包括：

- 缺少 Python 依赖；
- 输出路径不符合要求；
- 失败时未生成 `output.json`；
- `submit-test` 和 `submit` 调试成本较高。

解决方案：

- 编写 `run.sh`，统一从 `/target/target_spec.json` 读取输入，并输出到 `/workspace/output.json`。
- 在 `run.sh` 中增加对 `openai`、`rich` 的依赖安装兜底。
- 在评测失败时也强制写出结构化错误信息，方便服务器端定位问题。

## 5. Summary

总体而言，本项目构建了一个面向 GPU 硬件测量任务的本地 Agent 框架。它并不是简单的脚本拼接，而是结合了：

- ReAct 风格的多步推理与工具使用；
- 面向 CUDA benchmark 生成的 prompt 约束；
- completion-driven 的执行控制；
- family-based 的共享 benchmark 调度优化；
- 面向 Windows 与服务器提交环境的工程化容错机制。

该 Agent 的核心价值在于：将“读取 target -> 设计 benchmark -> 编译运行 -> profiling 分析 -> 汇总结果”这一完整流程自动化，并在尽量减少硬编码和外部依赖的前提下，提高测量效率、可解释性与提交稳定性。

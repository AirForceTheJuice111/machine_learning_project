**memxlife**
**机器学习系统，2026 春季**
**课程项目第二阶段：对 LoRA 算子的智能体式优化**
**截止日期：2026 年 5 月 12 日 上午 8:00**
**评估设备：NVIDIA GeForce RTX 3090**

在第一阶段中，你已经构建了一个能够检查 GPU 属性并推理性能的智能体式分析工具。
在第二阶段，你将为一个 LoRA 风格算子构建一个**优化智能体**。目标不是提交一个手工编写的静态内核，而是构建一个能够迭代生成、测试、分析并改进 CUDA 实现的智能体系统。

---

## 1. 任务
你的目标算子是

\[
Y = W X + A(B^T X)
\]

其中：

- \(W \in \mathbb{R}^{d \times d}\)
- \(X \in \mathbb{R}^{d \times d}\)
- \(A \in \mathbb{R}^{d \times r}\)
- \(B \in \mathbb{R}^{d \times r}\)
- \(r = 16\)
- 所有张量以 `.pt` 文件存储，并可通过 `torch.load` 加载。
- 所有隐藏的评估张量均使用 `float32`。

**形状范围**
评估时将会选取若干不同尺寸，隐藏维度 \(d\) 在 [3584, 4608] 范围内选择。

对于本项目，请使用以下公开范围：
\[
d \in [3584, 4608]
\]
评估中我们会在此区间内选择多个测试用例进行测试。
因此，你需要设计智能体以及生成的 CUDA 代码，使其能够处理该区间内的多种尺寸，而不是仅针对某一个特定的矩阵形状过拟合。

---

## 2. 你需要提交的内容
你的提交必须至少包含：

**`run.sh`**

在运行期间，你的系统必须维护一个文件：

**`optimized_lora.cu`**

**提交约定**
评估系统将进入提交根目录。
它会执行：
```bash
bash run.sh
```
随后，它会读取同一目录下的：
`./optimized_lora.cu`

**时间预算**
你的智能体最多可以运行 **30 分钟**。

在超时或正常结束时，评估系统会读取提交根目录下最终的 `optimized_lora.cu`，并使用官方测试框架对该文件进行基准测试。

因此：

- 你必须始终维护一个可编译的最新版 `optimized_lora.cu`
- 不要等到最后关头才生成第一个可工作的版本

---

## 3. 你的智能体应该做什么
你的提交必须是一个**实际的优化智能体**。

一个有效的智能体应具备以下能力：

- 生成候选 CUDA 实现
- 编译并测试它们
- 对它们进行基准测试
- 在不同的备选方案之间进行比较
- 迭代改进当前最佳实现
- 将当前最佳版本保持在 `optimized_lora.cu` 中

由于在官方测试中你的智能体无法接触隐藏的评估张量，你的智能体应在公开尺寸范围内自行生成合成的测试张量，并用于本地搜索。

**重要说明**
本项目**不是**要求你提交一个单一的静态内核。
课程团队期望看到的是一个**真正的智能体式工作流**，而不是一次性、固定不变的解决方案。

---

## 4. 官方评估环境
官方评估环境如下：

- **GPU**：NVIDIA GeForce RTX 3090
- **操作系统**：Ubuntu 22.04.4 LTS
- **Python**：3.10.12
- **PyTorch**：2.3.0a0+6ddf5cf85e.nv24.04
- **CUDA 工具包**：12.4
- **GCC**：11.4.0
- 环境中 `nvcc` 可从 CUDA 工具包安装路径获得。
- 必要时，你也可以依赖标准的 PyTorch 扩展工具链。

---

## 5. 官方 Python 测试框架
课程团队将使用一个 Python 测试框架来：

- 加载隐藏的 `W.pt`、`X.pt`、`A.pt`、`B.pt`
- 编译 `optimized_lora.cu`
- 调用导出的 CUDA 实现
- 将其输出与标准 PyTorch 参考实现比较
- 测量运行时间并进行基准测试
- 计算相对于标准 PyTorch 实现的加速比
- 将结果写入 `result.out`

以下是一个简化的参考版本：

```python
import torch
from pathlib import Path
from torch.utils.cpp_extension import load


def load_inputs(base_dir: str):
    base = Path(base_dir)
    W = torch.load(base / "W.pt", map_location="cpu").contiguous().cuda()
    X = torch.load(base / "X.pt", map_location="cpu").contiguous().cuda()
    A = torch.load(base / "A.pt", map_location="cpu").contiguous().cuda()
    B = torch.load(base / "B.pt", map_location="cpu").contiguous().cuda()
    return W, X, A, B


def reference_impl(W, X, A, B):
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def build_module(cu_path: str):
    module = load(
        name="optimized_lora_ext",
        sources=[cu_path],
        verbose=False,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    )
    return module


def check_correctness(y, y_ref):
    diff = (y - y_ref).float()
    max_abs_err = diff.abs().max().item()
    rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
    passed = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    return passed, max_abs_err, rel_l2_err


def benchmark(fn, W, X, A, B, warmup=10, iters=50):
    for _ in range(warmup):
        _ = fn(W, X, A, B)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(W, X, A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # 毫秒

    times.sort()
    return times[len(times) // 2]


def main():
    input_dir = "./hidden_inputs"
    cu_path = "./optimized_lora.cu"
    result_path = "./result.out"

    W, X, A, B = load_inputs(input_dir)
    module = build_module(cu_path)

    with torch.no_grad():
        y_student = module.forward(W, X, A, B)
        y_ref = reference_impl(W, X, A, B)

    passed, max_abs_err, rel_l2_err = check_correctness(y_student, y_ref)

    if passed:
        student_ms = benchmark(module.forward, W, X, A, B)
        torch_ms = benchmark(reference_impl, W, X, A, B)
        speedup = torch_ms / student_ms
    else:
        student_ms = None
        torch_ms = None
        speedup = 0.0

    with open(result_path, "w") as f:
        f.write(f"correct: {passed}\n")
        f.write(f"max_abs_err: {max_abs_err}\n")
        f.write(f"rel_l2_err: {rel_l2_err}\n")
        f.write(f"student_median_ms: {student_ms}\n")
        f.write(f"torch_median_ms: {torch_ms}\n")
        f.write(f"speedup: {speedup}\n")


if __name__ == "__main__":
    main()
```

**建议**：在你自己的智能体工作流中使用一个兼容的本地测试框架，以减少环境差异。

---

## 6. `optimized_lora.cu` 的接口要求
你最终提交的 `optimized_lora.cu` 必须满足：

- 是**单一文件**
- **自包含**
- 能够被官方 Python 测试框架**直接编译**
- 能够导出一个**可调用的入口点**

期望的可调用接口为：

```cpp
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B);
```

并且模块必须通过 `PYBIND11_MODULE(...)` 暴露该接口，以便测试框架可以调用：

```python
module.forward(W, X, A, B)
```

**允许的依赖项**
你可以使用：

- 标准 CUDA 头文件
- 标准 C/C++ 库头文件
- 系统环境中已存在的标准 PyTorch 扩展头文件

你**不得**依赖除 `optimized_lora.cu` 之外的额外源码文件。

这意味着在最终的 `optimized_lora.cu` 之外，不得有任何额外的：
- `.cu`
- `.cuh`
- `.h`
- `.cpp`

等文件。

---

## 7. 必须仔细阅读的专项规则

### 7.1 输入和输出格式
- 隐藏输入以 `.pt` 文件存储
- 它们通过 `torch.load` 加载
- 所有隐藏评估张量使用 `float32`
- 测试的算子是：
  \[
  Y = W X + A(B^T X)
  \]
- 隐藏维度 \(d\) 并非固定不变
- 评估会使用 \(d \in [3584, 4608]\) 范围内的多种尺寸
- 低秩维度固定为 16
- 你生成的 CUDA 实现必须接受给定的张量，并返回输出张量

### 7.2 评分策略与标准
**正确性**是硬性要求。

未通过正确性检查的提交将不会获得性能分数。

正确性通过与 PyTorch 参考实现：
\[
Y_{\text{ref}} = W X + A(B^T X)
\]
进行对比，使用以下标准：
```python
torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)
```
我们还会记录：
- `max_abs_err`
- `rel_l2_err`

对于通过正确性检查的提交，最终得分由以下部分组成：

- **70% 加速比**
- **30% 智能体实现 / 工程方法论**

**加速比**
加速比计算方式为：
\[
\text{加速比} = \frac{\text{标准 PyTorch 实现的运行时间中位数}}{\text{你的 CUDA 实现的运行时间中位数}}
\]
运行时间测量使用：
- 先进行预热
- CUDA 事件
- 多次重复运行的中位数延迟

**智能体实现 / 工程方法论**
该部分奖励那些真正实现了优化智能体的提交，考量因素包括：
- 迭代改进工作流
- 候选方案的生成与比较
- 使用基准测试/性能分析来辅助决策
- 代码组织性与可复现性
- 智能体系统的整体工程质量

### 7.3 你必须使用自己的 API 密钥
如果你的智能体依赖于外部模型 API，你必须使用**自己的 API 密钥**。

课程团队不会为你们提供 API 额度。

你应该设计你的系统，使得自己的 API 密钥可以通过安全且清晰的方式提供，例如通过环境变量或智能体所使用的本地配置文件。

### 7.4 严格禁止
以下行为被禁止：

- **只提交一个静态内核**
  本项目要求的是一个智能体，而不是一个固定的手工最终内核。
- **在智能体内部硬编码最终的 CUDA 代码**
  不要将预先写好的最终 `optimized_lora.cu` 作为一个固定的字面量/模板/字符串嵌入在智能体中，然后在运行时简单将其转储出来。
  你的智能体应当真正执行优化，而不是仅仅展示一个隐藏的最终答案。
- **依赖额外的源文件来生成最终的实测实现**
  最终被评测的实现必须是单文件的 `optimized_lora.cu`。
- **破坏官方 I/O 约定**
  评估系统会在提交根目录执行 `bash run.sh`，并从同一目录读取 `./optimized_lora.cu`。

---

## 8. 实用建议
一份优秀的提交通常会：

- 生成候选的 CUDA 代码变体
- 自动编译并测试它们
- 对照本地 PyTorch 参考实现验证正确性
- 预热后进行多次基准测试
- 将当前最佳的有效实现持续保持在 `optimized_lora.cu` 中
- 避免对某一个确切的矩阵尺寸过拟合

---

## 9. 总结
在本项目中，你将为一个 LoRA 算子构建一个智能体式 CUDA 优化系统。

你的智能体应当：
- 通过 `bash run.sh` 启动运行
- 迭代改进 CUDA 代码
- 始终维护一个可用的 `optimized_lora.cu`
- 产出一个单文件、自包含的最终 CUDA 实现
- 针对算子
  \[
  Y = W X + A(B^T X)
  \]
  在 [3584, 4608] 范围内的多种隐藏测试尺寸上进行优化

只有正确的实现才会被排名，而在正确的提交中，最终评估基于：
- **70% 加速比**
- **30% 智能体实现 / 工程方法论**
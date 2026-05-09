# TVM + OpenCL — 深度解析

## 一句话

**Apache TVM 是一个深度学习编译器：输入 PyTorch/ONNX 模型 → 输出目标平台的优化原生代码（OpenCL/CUDA/Metal/...）。**

## 做什么的？

```
你的 PyTorch Demucs 模型
         │
         ▼
  TVM Relay IR (中间表示)
         │
         ├── 图级优化: 算子融合、常量折叠、死代码消除
         ├── 张量化: 将 Reshape/Transpose 折叠进计算
         └── AutoTVM/AutoScheduler: 每种算子搜索最优 OpenCL workgroup
         │
         ▼
  TVM TIR (张量中间表示) → OpenCL Kernel 代码
         │
         ▼
  编译为 .so/.dll (动态库)
         │
         ▼
  你的 C++ 程序 dlopen → 调用 → 得到结果
```

## 完整工作流

**第 1 步：用 Python 做编译（一次性）**
```python
import tvm
from tvm import relay
import torch
import demucs

# 1. 加载模型
model = demucs.pretrained.get_model("htdemucs_6s")
model.eval()

# 2. 导出为 TVM Relay IR
input_shape = (1, 2, 441000)  # batch, channels, samples
scripted = torch.jit.trace(model, torch.randn(input_shape))
mod, params = relay.frontend.from_pytorch(scripted, [("audio", input_shape)])

# 3. 设定目标 + 编译
target = tvm.target.Target("opencl")  # 生成 OpenCL kernel
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

# 4. 导出
lib.export_library("demucs_opencl.so")         # 动态库
with open("demucs_graph.json", "w") as f:      # 图描述
    f.write(lib.get_graph_json())
with open("demucs.params", "wb") as f:         # 权重
    f.write(relay.save_param_dict(params))
```

**第 2 步：C++ 加载 + 推理**
```cpp
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <dlfcn.h>

// 1. 加载动态库
tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("demucs_opencl.so");

// 2. 加载图 + 参数
auto graph_json = read_file("demucs_graph.json");
auto params_bytes = read_file("demucs.params");
mod = tvm::runtime::GraphExecutor::Create(graph_json, mod, 
    {tvm::Device{"opencl", 0}});

// 3. 设置输入
auto input = tvm::runtime::NDArray::Empty({1, 2, 441000}, 
    tvm::DataType::Float(32), tvm::Device{"opencl", 0});
input.CopyFromBytes(audio_data, audio_size);

// 4. 推理！
mod.SetInput("audio", input);
mod.Run();

// 5. 取结果
auto output = mod.GetOutput(0);  // stems
```

## 为什么比手写 OpenCL 好？

### 1. 算子融合：手写需要 20+ 次 kernel，TVM 可以融合到 5 次

```
手写 OpenCL:                     TVM 融合后:
  Conv1d  ──→ kernel launch       ┌─ Conv1d ──┐
  GLU     ──→ kernel launch       │  + GLU    │ ──→ 1 次 kernel
  ReLU    ──→ kernel launch       │  + ReLU   │
  LayerN  ──→ kernel launch       └───────────┘
                     ↓
  每次 launch 有 OpenCL API 开销 (~10μs) → 20×10μs = 200μs 浪费
  融合后只需 5 次 launch → 50μs
```

### 2. AutoTVM：自动调优 workgroup size

手写 OpenCL 你需要手工尝试 `__attribute__((work_group_size_hint(64,1,1)))` 的各种值。TVM 的 AutoScheduler 会用 ML 模型**自动搜索**最优配置。

### 3. 混合精度选择

```python
target = "opencl -device=amd_radeon"  # AMD GPU
target = "opencl -device=nvidia_gpu"  # NVIDIA GPU  
target = "opencl -device=intel_gpu"   # Intel iGPU

# TVM 会根据设备能力自动选算子实现 (FP16/FP32/INT8)
```

### 4. 仍然可以写 custom kernel

如果 TVM 自动生成的 Attention kernel 不够快，你可以手写那个特定 kernel 作为 TVM 的 external op 挂上去：

```python
@tvm.register_func("demucs.attention")
def custom_attention(q, k, v):
    # 你的手写 OpenCL kernel
    ...
```

## 权衡

| | TVM + OpenCL | ONNX Runtime OpenCL EP | 纯手写 OpenCL |
|---|:---:|:---:|:---:|
| OpenCL Kernel 质量 | ⭐⭐⭐ Auto-tuned | ⭐⭐⭐ 手工优化 | ⭐→⭐⭐⭐⭐ 看水平 |
| 开发时间 | ~2 周 | ~3 天 | ~5 个月 |
| 算子覆盖率 | 接近 100% | ~95% (Transformer 可能缺) | 自己决定 |
| 性能天花板 | 较高 (融合+调优) | 中等 (无融合) | 最高 (极限手写) |
| 依赖链 | TVM (C++ runtime, ~5MB) | ONNX Runtime (~30MB) | 无 |
| 社区 | Apache 顶级项目, 活跃 | Microsoft, 稳定 | --- |

## 在项目中如何集成

TVM 编译输出 3 个文件：
```
models/
  demucs_opencl.so        ← OpenCL kernel 动态库 (~5MB)
  demucs_graph.json       ← 计算图描述 (~50KB)
  demucs.params           ← 模型权重 (~600MB)
```

你的 C++ 项目只需要链接 TVM C++ Runtime (`libtvm_runtime.so`, ~5MB)，不需要完整的 TVM 编译链。

CMakeLists.txt 加一行：
```cmake
find_package(TVM REQUIRED)
target_link_libraries(rmms-ai-server-onnx PRIVATE tvm_runtime)
```

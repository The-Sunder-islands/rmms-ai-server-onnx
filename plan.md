# RMMS AI Server ONNX — 项目计划 v1.0

## 1. 项目概述

### 定位
创建 rmms-ai-server 的 **ONNX Runtime + 现代 C++（C++20）** 高性能衍生版本。
零 Python 运行时开销，完全兼容 RMMS AI 服务端协议 v1.0.0-draft。

### 核心决策一览

| 领域 | 决策 | 理由 |
|------|------|------|
| **C++ 标准** | C++20 | 协程/concepts/span，未来可能另建 C++23 实验仓库 |
| **ONNX 导出** | `torch.export` + `dynamo_export` | PyTorch 2.x 新方案，动态形状友好 |
| **STFT** | 混合模式 | 模型内置 STFT，C++ 也提供替换实现 |
| **精度** | FP32 + FP16 双版本 | |
| **模型分发** | 魔搭社区 | 需要时告知授权 |
| **ORT 集成** | 纯 C API | 最稳定，无 C++ ABI 问题 |
| **HTTP** | Drogon | 内置 SSE + 协程支持 |
| **JSON** | glaze | C++20 反射，编译期序列化 |
| **音频** | FFmpeg (最新，x86_64_v3) | 全格式，优化指令集 |
| **构建** | CMake + Conan | Windows + Linux |
| **并发** | C++20 协程 | Drogon 原生支持 |
| **版本协商** | 宽松兼容 | 只返回头，不拒绝 |
| **测试** | Google Test | |

### 阶段路线图

| 阶段 | 目标 | 状态 |
|------|------|------|
| **Phase 0** | 模型导出脚本 (PyTorch → ONNX) | ⏳ |
| **Phase 1** | C++ onnxruntime 单文件推理 PoC | ⏳ |
| **Phase 2** | HTTP 服务端骨架 + REST API | ⏳ |
| **Phase 3** | 完整协议实现 (SSE/Capabilities/Tasks) | ⏳ |
| **Phase 4** | 多设备后端 + EP 优化 | ⏳ |
| **Phase 5** | Docker + 文档 + 基准测试 | ⏳ |

---

## 2. 架构设计

### 目录结构
```
rmms-ai-server-onnx/
├── CMakeLists.txt
├── conanfile.txt
├── README.md
├── plan.md
├── src/
│   ├── main.cpp
│   ├── server/          # HTTP 服务端 (Drogon controllers)
│   │   ├── health.cpp
│   │   ├── capabilities.cpp
│   │   ├── tasks.cpp
│   │   ├── events.cpp      # SSE
│   │   └── files.cpp
│   ├── engine/          # ONNX 推理引擎
│   │   ├── session.h/cpp    # ORT Session 封装
│   │   ├── split.h/cpp      # 音频分轨
│   │   └── preprocess.h/cpp # 音频预处理
│   ├── device/          # 设备后端
│   │   ├── backend.h        # 抽象接口
│   │   ├── registry.h/cpp   # 动态库加载 + 注册
│   │   ├── cpu.cpp
│   │   ├── cuda.cpp
│   │   ├── directml.cpp
│   │   └── opencl.cpp
│   ├── protocol/        # 协议模型 (glaze)
│   │   ├── models.h        # 数据结构
│   │   ├── errors.h        # 错误码
│   │   └── serialize.h     # JSON 序列化
│   └── util/            # 工具
│       ├── audio.h/cpp
│       ├── config.h/cpp    # TOML 配置
│       └── logging.h
├── tests/
├── models/              # .onnx (gitignored)
├── scripts/             # Python 导出脚本
│   └── export_demucs.py
├── docker/
└── docs/
```

### 数据流
```
Client (RMMS DAW)
  │  POST /api/v1/tasks (multipart audio)
  ▼
┌─ Drogon HTTP Server ──────────────────────────┐
│  Controller → TaskManager → PipelineRunner     │
│    │                                              │
│    ├─ progress/partial_result → SSE broadcast  │
│    │                                              │
│    ▼                                              │
│  Engine::Split                                  │
│    ├─ FFmpeg 解码 → float32[]                   │
│    ├─ 分段 (Sectioner, 保守自适应)               │
│    ├─ ORT Session::Run() per section            │
│    ├─ 拼接 (crossfade overlap)                  │
│    └─ FFmpeg 编码 → .wav/.flac/.mp3             │
└────────────────────────────────────────────────┘
```

---

## 3. 关键模块详设

### 3.1 音频分段器 (Sectioner)

**策略：保守自适应 + 核显修正**

```
启动时:
  probe_memory() → available_mb
  
  对核显 (Intel iGPU / AMD APU):
    探测的是系统内存而非专用显存
    effective_mb = available_mb * 0.75  ← 额外修正因子
    
  对独立显卡:
    effective_mb = available_mb

  max_section_samples = (effective_mb - model_size_mb) / peak_memory_per_sample
  max_section_seconds  = max_section_samples / sample_rate
  max_section_seconds  = clamp(max_section_seconds, 30, 600)

推理时:
  total_seconds = audio_duration
  段数 N = ceil(total_seconds / max_section_seconds)
  每段实际长度 = total_seconds / N
  overlap = base_overlap * (1.0 + (1.0 / N) * 0.5)   ← 段越多 overlap 越大
  
  crossfade 拼接:
    窗口函数: 三角窗 (简单高效)
    每段首尾 overlap/2 长度做 linear crossfade
    中间区域直接拼接
```

### 3.2 设备后端 (Device Backend)

**EP 优先级：CUDA > DirectML > OpenCL > CPU**

```
auto-detect 流程:
  1. 遍历可用 EP
  2. 每个 EP 创建临时 ORT Session，做一次 dummy 推理
  3. 记录延迟 → 按延迟排序
  4. 选最快的作为主 EP
  5. 缓存结果 (启动时一次)

降级回退:
  try: CUDA EP 推理
  catch OOM: 减半分段 → CUDA 重试一次
  catch OOM again: 回退 CPU EP, 分段切到极大值 (600s)

动态库加载:
  lib/rmms_backend_cuda.so   → 导出 create_backend() → CUDABackend*
  lib/rmms_backend_dml.so    → 导出 create_backend() → DMLBackend*
  lib/rmms_backend_opencl.so → 导出 create_backend() → OpenCLBackend*
  
  启动时 dlopen 扫描 lib/ 目录，自动注册
```

**device info 抽象 (variant 模式):**

```cpp
using DeviceBackend = std::variant<CPUBackend, CUDABackend, DMLBackend, OpenCLBackend>;

// 编译期多态, 零虚函数开销
auto info = std::visit([](auto& b) { return b.get_info(); }, backend);
```

### 3.3 内存管理 (std::pmr)

```cpp
// 推理专用内存池
class InferenceArena {
    std::pmr::monotonic_buffer_resource pool{256 * 1024 * 1024}; // 256MB
    std::pmr::vector<float> input_buffer{&pool};
    std::pmr::vector<float> output_buffer{&pool};
};
// 单次任务完成后 pool.release() 统一释放
```

### 3.4 与 Python 版共存

| 方面 | 策略 |
|------|------|
| **部署** | 并存部署，不同端口 |
| **端口** | Python: 8420 / C++: 8421 |
| **能力声明** | 如实返回，`not_implemented` 的能力标注为禁用 |
| **模型** | 各管各的，独立模型目录 |
| **mDNS** | 同 `_rmms-ai._tcp`，不同实例名 |

### 3.5 客户端发现

```
mDNS 广播:
  RMMS AI Server (Python)._rmms-ai._tcp.local  → port 8420
  RMMS AI Server (ONNX/C++)._rmms-ai._tcp.local → port 8421

客户端列表:
  ┌─────────────────────────────────┐
  │ 发现以下 AI 服务端:              │
  │                                 │
  │ ○ RMMS AI Server (Python)       │
  │   192.168.1.100:8420            │
  │   能力: 分轨/MIDI/检测          │
  │                                 │
  │ ○ RMMS AI Server (ONNX/C++)     │
  │   192.168.1.100:8421            │
  │   能力: 分轨 (ONNX 加速)        │
  │                                 │
  │ [手动添加] [自动连接]           │
  └─────────────────────────────────┘
  
  用户手动选择 → 确认 → 连接
```

---

## 4. Phase 0: 模型导出

> **注意**：此阶段用 Python 写脚本，但脚本本身属于项目的一部分。
> 导出完成后，后续所有代码都是 C++。

### 4.1 导出脚本 `scripts/export_demucs.py`

```python
# 伪代码
import torch
import demucs

model = demucs.pretrained.get_model("htdemucs_6s")
model.eval()

# 通过 torch.export + dynamo 追踪
example = torch.randn(1, 2, 44100 * 10)  # 10s stereo
exported = torch.export.export(model, (example,),
    dynamic_shapes=({"input": {2: torch.export.Dim("time")}},))

torch.onnx.dynamo_export(exported, "htdemucs_6s.onnx",
    input_names=["audio"],
    output_names=["vocals", "drums", "bass", "other", "guitar", "piano"],
    dynamic_axes={"audio": {2: "time"}, 
                  "vocals": {2: "time"}, "drums": {2: "time"}, ...})
```

### 4.2 推荐 ORT 配置

```
SessionOptions:
  graph_optimization_level = ORT_ENABLE_ALL
  intra_op_num_threads     = CPU 核心数
  inter_op_num_threads     = 1
  execution_mode           = ORT_SEQUENTIAL
  enable_profiling         = false
```

---

## 5. OpenCL 补充说明

### 泛用性

OpenCL 确实是 ONNX Runtime 中**平台覆盖面最广**的 GPU EP：

| GPU 厂商 | CUDA EP | DirectML EP | OpenCL EP |
|----------|---------|-------------|-----------|
| NVIDIA   | ✓ 首选  | ✓ (Win)     | ✓         |
| AMD      | ✗       | ✓ (Win)     | ✓ (主要路径) |
| Intel    | ✗       | ✓ (Win)     | ✓ (iGPU)  |
| Apple    | ✗       | ✗           | ✗ (已弃用)  |

**OpenCL 是 AMD GPU 在 Linux 下的唯一 GPU 加速路径。**

### 已知限制

- Transformer MultiheadAttention 算子可能需要 fallback 到 CPU
- 需要 OpenCL 1.2+ runtime（Intel NEO / AMD ROCm / NVIDIA OpenCL）
- 部分算子不支持 FP16，只能用 FP32

### 建议优先级

```
CUDA > DirectML > OpenCL > CPU
  │        │          │        │
  │        │          │        └─ 所有平台回退
  │        │          └─ AMD Linux / Intel iGPU
  │        └─ Windows 通用 GPU
  └─ NVIDIA 首选
```

---

## 6. 下一步行动

1. 创建 CMake + Conan 项目骨架
2. 创建目录结构和初始头文件
3. 实现 Phase 1: onnxruntime C API 最简推理 PoC
4. Phase 0 的模型导出脚本留空框架（等你准备好 PyTorch 环境）

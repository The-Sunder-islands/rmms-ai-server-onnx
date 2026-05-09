# RMMS AI Server ONNX — 项目计划

## 1. 项目概述

### 你的诉求
- **目标**：创建 rmms-ai-server 的 ONNX + 现代 C++ 衍生版本
- **性能**：极致性能，零 Python 运行时开销
- **依赖**：最小化依赖，脱离 PyTorch 生态
- **兼容**：完全兼容 RMMS AI 服务端协议 v1.0.0-draft
- **模型**：导出 PyTorch Demucs (htdemucs/htdemucs_6s) → ONNX 格式
- **路线**：直接 C++ 起步（非 Python 过渡）
- **定位**：实验性重构，长期独立项目
- **协议**：REST + SSE 完全兼容现有客户端

### 技术栈 （✅ = 已确定）
| 组件 | 选定 | 说明 |
|------|------|------|
| C++ 标准 | **C++20** | 未来可能再建 C++23 实验仓库 |
| ONNX Runtime | Microsoft onnxruntime | 纯 C API |
| HTTP 框架 | **Drogon** | 内置 SSE/协程 |
| JSON 解析 | **glaze** | C++20 反射，编译期序列化 |
| 音频 I/O | **FFmpeg (最新)** | x86_64_v3 优化编译 |
| 构建系统 | **CMake** | |
| 包管理 | **Conan** | Windows + Linux |
| 配置 | TOML (toml++) | |
| 日志 | spdlog | |
| 测试 | Google Test | |
| 并发模型 | **C++20 协程** | Drogon 内置 |

### 阶段路线图
| 阶段 | 目标 | 状态 |
|------|------|------|
| **Phase 0** | 模型导出 (PyTorch → ONNX) + 验证 | ⏳ |
| **Phase 1** | C++ onnxruntime 单文件推理 PoC | ⏳ |
| **Phase 2** | HTTP 服务端骨架 + REST API | ⏳ |
| **Phase 3** | 完整协议实现 (SSE/Capabilities/Tasks) | ⏳ |
| **Phase 4** | 多设备后端 + 性能优化 | ⏳ |
| **Phase 5** | Docker 部署 + 文档 | ⏳ |

### 项目目录结构
```
rmms-ai-server-onnx/
├── CMakeLists.txt
├── conanfile.txt
├── src/
│   ├── main.cpp
│   ├── server/          # HTTP 服务端 (Drogon)
│   ├── engine/          # ONNX 推理引擎
│   ├── protocol/        # 协议模型 (glaze)
│   ├── device/          # 设备后端
│   └── util/            # 工具函数
├── include/             # 公共头文件
├── tests/
├── models/              # .onnx 文件 (gitignored)
├── scripts/             # 模型导出脚本 (Python)
├── docker/
├── docs/
├── README.md
└── plan.md
```

---

## 2. 已决策项汇总

### Phase 0: 模型导出
| Q | 决策 |
|---|------|
| Q1 导出策略 | `torch.export` + `torch.onnx.dynamo_export` (PyTorch 2.x) |
| Q2 STFT 位置 | 混合模式：模型内做 STFT，C++ 也提供替换实现 |
| Q3 精度 | FP32 + FP16 双版本 |
| Q4 分发 | 上传到**魔搭社区** (需要时告知，你去授权) |

### Phase 1: C++ PoC
| Q | 决策 |
|---|------|
| Q5 C++ 标准 | C++20 (未来可能再建 C++23 实验仓库) |
| Q6 ORT 集成 | 纯 C API (`OrtCreateSession` 等) |
| Q7 音频库 | 最新 FFmpeg，x86_64_v3 指令集编译优化 |
| Q8 分段策略 | **待展开追问** ↓ |

### Phase 2-3: HTTP + 协议
| Q | 决策 |
|---|------|
| Q9 HTTP 框架 | Drogon |
| Q10 并发模型 | C++20 协程 |
| Q11 JSON 库 | glaze |
| Q12 版本协商 | 宽松兼容（只返回头，不拒绝） |

### Phase 4: 设备
| Q | 决策 |
|---|------|
| Q13 EP 策略 | **待展开追问** ↓ |
| Q14 性能 vs 扩展 | **待展开追问** ↓ |

### 工程化
| Q | 决策 |
|---|------|
| Q15 包管理 | Conan (Windows + Linux) |
| Q16 目录结构 | 采用建议结构 |
| Q17 测试 | Google Test 单元测试 |

### 迁移兼容
| Q | 决策 |
|---|------|
| Q18 共存 | **待展开追问** ↓ |
| Q19 客户端发现 | **待展开追问** ↓ |

---

## 3. 展开追问

### Q8: 长音频分段策略 — 如何更动态？

你说"希望更加动态"。当前 Python 版的策略是：

```
固定阈值 (300s) → 超过则分割为 N 段 (每段 ≤ 300s) → GPU 不限段数 / CPU 限 8 段
```

问题是这很"静态" —— 不管 GPU 有多少显存都硬编码 300s。

**更动态的方案：**

<details open>
<summary><b>Q8a: 显存自适应分段</b></summary>

在推理前探测可用显存，根据模型大小动态计算最长段长：
```
max_section_seconds = (available_memory - model_size) / (peak_memory_per_second)
```

- 192GB H100 → 可能整首歌不分段
- 4GB GTX 1050 → 可能每段 30s

你倾向于：
- [ ] **完全自适应**：每次推理前探测显存
- [ ] **保守自适应**：启动时探测一次，缓存结果
- [ ] **用户可控**：提供参数让用户设段长上限
- [ ] **基准测试驱动**：预置不同 GPU 的推荐配置表

</details>

<details open>
<summary><b>Q8b: 分段重叠与拼接质量</b></summary>

分段边界处的音频拼接可能出现不连续。当前用 `overlap` 参数处理。动态分段下：
- [ ] **固定 overlap 比例**：不管段多段少都用同一个 overlap
- [ ] **动态 overlap**：段越短 overlap 比例越大（短段更易边界失真）
- [ ] **交叉淡入淡出**：在拼接处做 crossfade

你倾向哪个？

</details>

<details open>
<summary><b>Q8c: 尾部处理</b></summary>

音频长度未必是段长的整数倍。最后一段可能很短：
- [ ] **补零**：pad 到标准长度
- [ ] **反射填充**：mirror padding 保持连续性
- [ ] **自适应缩短**：最后一段独立处理（可能质量下降）
- [ ] **交给模型处理**：让 ONNX 动态轴自然处理短输入

你倾向哪个？

</details>

---

### Q13: ONNX Execution Provider — 解释 + 追问

**什么是 EP？** ONNX Runtime 的 Execution Provider 是把 ONNX 算子映射到具体硬件的后端。不同 EP 对算子的支持程度不同。

```
ONNX 模型图
  ├── CPU EP       ← 所有算子都支持（回退方案）
  ├── CUDA EP      ← NVIDIA GPU，需 cuDNN/TensorRT
  ├── DirectML EP  ← Windows GPU (DirectX 12)，AMD/NVIDIA/Intel 通用
  ├── CoreML EP    ← macOS (Apple Silicon / AMD GPU)
  ├── OpenVINO EP  ← Intel CPU/GPU/iGPU
  ├── TensorRT EP  ← NVIDIA 最快，但需额外构建 ort
  └── ROCm EP      ← AMD GPU (Linux)
```

**Drogon 集成要点：** Drogon 的协程模型下，ORT 推理是阻塞操作。需要在独立线程池中执行，协程 await 其结果。

<details open>
<summary><b>Q13a: EP 优先级策略</b></summary>

多 EP 可用时（如同时有 CUDA 和 CPU），选择策略：
- [ ] **自动检测择优**：启动时测试每个 EP 延迟，选最快的
- [ ] **用户指定**：配置文件/API 参数强制选择
- [ ] **固定优先级**：CUDA > DirectML > CoreML > CPU
- [ ] **每个模型可不同**：split 用 CUDA，midi 用 CPU

你倾向哪个？

</details>

<details open>
<summary><b>Q13b: EP 回退</b></summary>

如果 CUDA EP 因 OOM 失败：
- [ ] **自动回退 CPU**：静默切换，任务继续
- [ ] **失败 + 提示**：返回错误，让用户切换设备
- [ ] **降级重试**：同一 EP 用更小分段重试，不行再回退

你倾向哪个？

</details>

<details open>
<summary><b>Q13c: 先做哪些 EP？</b></summary>

- [ ] **Phase 1 只做 CPU**（最简 PoC）
- [ ] **Phase 1 CPU + CUDA**（覆盖主要场景）
- [ ] **Phase 1 CPU + CUDA + DirectML**（Windows 也能 GPU）

你倾向哪个？

</details>

<details open>
<summary><b>Q13d: CoreML 的价值</b></summary>

CoreML EP 在 macOS 上可以利用 Apple Silicon 的 ANE（神经网络引擎）。但 Demucs 模型可能不完全被 CoreML 支持。
- [ ] **值得做**：Apple Silicon 用户是重要群体
- [ ] **先不做**：等用户反馈再决定
- [ ] **用 MPS 替代**：PyTorch MPS 后端对应 ORT 的什么等价物？

你倾向哪个？

</details>

---

### Q14: 性能 vs 可扩展性双赢

这是一个经典的架构权衡。传统的"抽象层"虽然可扩展但会引入虚函数/间接调用开销。

<details open>
<summary><b>Q14a: 编译期多态 vs 运行期多态</b></summary>

```cpp
// 方案 A: 传统虚函数 (运行期多态, 可扩展)
class IInferenceBackend {
    virtual std::vector<Tensor> run(const Tensor& input) = 0;
};
class CUDABackend : public IInferenceBackend { ... };
class DirectMLBackend : public IInferenceBackend { ... };

// 方案 B: 模板 + Concept (编译期多态, 零开销)
template<InferenceBackend B>
class Pipeline { ... };
```

方案 B 性能更好但编译慢、二进制大。方案 A 可扩展性好但有虚函数开销。

现在 C++20 有了 `std::variant` + `std::visit`：
```cpp
// 方案 C: variant (编译期多态, 无虚函数, 类型安全)
using Backend = std::variant<CPUBackend, CUDABackend, DirectMLBackend>;
```

你倾向于：
- [ ] **方案 A**：传统 OOP，简单清晰
- [ ] **方案 B**：模板，极致性能
- [ ] **方案 C**：variant，平衡折中
- [ ] **混合**：公开接口用 variant，内部实现用模板

</details>

<details open>
<summary><b>Q14b: EP 注册机制</b></summary>

当前 Python 版有装饰器式注册：
```python
@register_backend
class CUDABackend(DeviceBackend): ...
```

C++ 中类似设计：
- [ ] **手动注册**：`registry.add<CUDABackend>()`，简单粗暴
- [ ] **自注册 static 变量**：利用静态初始化自动注册
- [ ] **CMake 生成**：构建时扫描目录，生成注册代码
- [ ] **动态库加载**：每个后端是 `.so/.dll`，运行时 `dlopen`

你倾向哪个？

</details>

<details open>
<summary><b>Q14c: 内存管理</b></summary>

ONNX 推理涉及大量张量内存分配。C++ 下的策略：
- [ ] **Ort::Allocator**：使用 ORT 自带分配器
- [ ] **自定义 Arena**：预分配大块内存池，减少碎片
- [ ] **std::pmr**：C++17 polymorphic memory resource
- [ ] **先简单后优化**：初期用 ORT 默认分配，性能瓶颈时再优化

你倾向哪个？

</details>

---

### Q18: 与 Python 版共存的细节

两个服务端实现同一份协议，可能会在同一个网络中共存。

<details open>
<summary><b>Q18a: 部署形态</b></summary>

- [ ] **互斥部署**：同一台机器只运行其中一个
- [ ] **并存部署**：不同端口运行两个实例，客户端选择
- [ ] **级联部署**：C++ 作主服务，Python 作 fallback（处理 ONNX 不支持的 capability）

你倾向哪个？

</details>

<details open>
<summary><b>Q18b: 能力声明差异</b></summary>

如果 C++ 版暂时只支持 `split`（分轨），不支持 `midi`/`detect`。capabilities 端点应：
- [ ] **如实返回**：`split: implemented`, `midi: not_implemented`, `detect: not_implemented`
- [ ] **隐藏未实现**：只返回 split

你倾向哪个？

</details>

<details open>
<summary><b>Q18c: 模型文件共享</b></summary>

`.onnx` 文件格式和 PyTorch `.pth` 不同，无法共享。但如果未来两个版本都支持 `.onnx`：
- [ ] **独立模型目录**：各管各的
- [ ] **共享模型目录**：`AI_SERVER_MODEL_CACHE_DIR` 指向同一位置
- [ ] **HTTP 模型服务**：独立模型服务，两个版本都 HTTP 调它

你倾向哪个？

</details>

---

### Q19: 客户端发现 — 你的看法？

目前 Python 版通过 mDNS 广播 `_rmms-ai._tcp`。两个版本共存时：

<details open>
<summary><b>Q19a: mDNS 实例区分</b></summary>

- [ ] **相同服务类型，不同实例名**：
  - `RMMS AI Server (Python)._rmms-ai._tcp`
  - `RMMS AI Server (ONNX/C++)._rmms-ai._tcp`
  - 客户端在列表中看到两个，由用户选择

- [ ] **不同服务类型**：
  - Python: `_rmms-ai._tcp`
  - ONNX:  `_rmms-ai-onnx._tcp`
  - 客户端不需要改动就能区分

- [ ] **TXT 记录标记**：
  - 相同 `_rmms-ai._tcp`，TXT 加 `runtime=onnx-cpp`
  - 客户端读取 TXT 后自动选择

你倾向哪个？

</details>

<details open>
<summary><b>Q19b: 客户端自动选择</b></summary>

如果客户端看到两个服务端都可用：
- [ ] **让用户选**：弹窗列出服务端，标注性能特点
- [ ] **自动选 C++ 版**：因为更快
- [ ] **自动选延迟最低的**：ping 测试
- [ ] **这个你不管**：由客户端开发者决定

你倾向哪个？

</details>

<details open>
<summary><b>Q19c: 手动配置兼容</b></summary>

如果用户在 RMMS 配置文件里硬编码了 Python 版地址：
```xml
<ai><server_url>http://192.168.1.100:8420</server_url></ai>
```

切换到 C++ 版是否需要改配置？
- [ ] **同端口替换**：停 Python 启 C++，客户端无感
- [ ] **不同默认端口**：C++ 版用 8421，避免冲突
- [ ] **端口可配**：用户自己选

你倾向哪个？

</details>

---

## 4. 下一步

等你回答完上述追问后，我会：
1. 更新 plan.md 为最终版本
2. 画出 Phase 0 的模型导出脚本架构
3. 建立 CMake + Conan 项目骨架
4. 开始写第一个 ONNX 推理 PoC

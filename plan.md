# RMMS AI Server ONNX — 项目计划

## 1. 项目概述

### 你的诉求（已明确）
- **目标**：创建 rmms-ai-server 的 ONNX + 现代 C++ 衍生版本
- **性能**：极致性能，零 Python 运行时开销
- **依赖**：最小化依赖，脱离 PyTorch 生态
- **兼容**：完全兼容 RMMS AI 服务端协议 v1.0.0-draft
- **模型**：导出 PyTorch Demucs (htdemucs/htdemucs_6s) → ONNX 格式
- **路线**：直接 C++ 起步（非 Python 过渡）
- **定位**：实验性重构，长期独立项目
- **协议**：REST + SSE 完全兼容现有客户端

### 技术栈框架 (TBD = 待确定)
| 组件 | 候选 | 选定 |
|------|------|------|
| 构建系统 | CMake / Meson / xmake | TBD |
| C++ 标准 | C++20 / C++23 | TBD |
| ONNX Runtime | Microsoft onnxruntime | 确定 |
| HTTP 框架 | cpp-httplib / Drogon / Boost.Beast / uWebSockets | TBD |
| JSON 解析 | nlohmann/json / simdjson / glaze | TBD |
| 音频 I/O | libsndfile / dr_libs / miniaudio | TBD |
| 日志 | spdlog / fmt | TBD |
| 测试 | Google Test / Catch2 / doctest | TBD |
| 包管理 | vcpkg / Conan / FetchContent | TBD |
| 配置 | TOML (toml++) / JSON / YAML | TBD |

### 阶段路线图
| 阶段 | 目标 | 状态 |
|------|------|------|
| **Phase 0** | 模型导出 (PyTorch → ONNX) + 验证 | ⏳ |
| **Phase 1** | C++ onnxruntime 单文件推理 PoC | ⏳ |
| **Phase 2** | HTTP 服务端骨架 + REST API | ⏳ |
| **Phase 3** | 完整协议实现 (SSE/Capabilities/Tasks) | ⏳ |
| **Phase 4** | 多设备后端 + 性能优化 | ⏳ |
| **Phase 5** | Docker 部署 + 文档 | ⏳ |

---

## 2. Phase 0: 模型导出

### 需要决定的问题

<details open>
<summary><b>Q1: Demucs 模型导出策略</b></summary>

Demucs 的 HTDemucs 架构包含：
- CNN 编码器/解码器（标准 Conv1d/ConvTranspose1d）
- Transformer 层（Multi-head Attention）
- LSTM 层（部分版本）
- 动态输入长度（变长音频）

ONNX 导出面临的核心挑战：
- **动态形状**：输入音频长度可变，需要 `dynamic_axes`
- **Transformer**：PyTorch 的 MultiheadAttention 在 ONNX 导出时有时需要特殊处理
- **复数运算**：Demucs 内部可能涉及 STFT 复数值处理

导出方案选择：
- [ ] **torch.onnx.export**：标准方案，可能需要手动处理 Transformer 算子
- [ ] **torch.export + torch.onnx.dynamo_export**：PyTorch 2.x 新方案，对动态形状支持更好
- [ ] **分阶段导出**：先导出编码器+Transformer+解码器为独立子图
- [ ] **ONNX 模型仓库**：导出后托管在 Hugging Face / GitHub Releases

你的想法？
</details>

<details open>
<summary><b>Q2: STFT/ISTFT 处理</b></summary>

Demucs 内部使用 STFT 将音频转为频谱。ONNX 原生支持 STFT 算子（opset 17+），但：
- 窗口函数需要预计算
- 需要在 C++ 侧做预处理还是在模型内部？
- 或者完全在 C++ 侧做 STFT，模型只负责分离？

选项：
- [ ] **模型内 STFT**：所有处理在 ONNX 图中，单次推理调用
- [ ] **C++ 侧 STFT**：模型输入/输出为频谱，C++ 负责音频↔频谱转换
- [ ] **混合**：模型内做 STFT，但 C++ 提供替换实现作为优化

你的想法？
</details>

<details open>
<summary><b>Q3: 模型精度</b></summary>

原始 Demucs 使用 FP32。对 ONNX 推理：
- [ ] **FP32**：最大兼容性，但体积大（~330MB/4轨，~600MB/6轨）
- [ ] **FP16**：体积减半，需要 GPU 支持（CUDA FP16 / CoreML）
- [ ] **INT8 量化**：最小体积，但质量可能下降
- [ ] **多精度发布**：FP32 + FP16 双版本

你的想法？
</details>

<details open>
<summary><b>Q4: 模型分发方式</b></summary>

- [ ] **GitHub Releases**：随代码发布 .onnx 文件
- [ ] **Hugging Face Hub**：独立模型仓库
- [ ] **首次运行下载**：服务启动时自动下载
- [ ] **内嵌**：随 Docker 镜像打包

你的想法？
</details>

---

## 3. Phase 1: C++ 单文件推理 PoC

### 需要决定的问题

<details open>
<summary><b>Q5: C++ 标准</b></summary>

- [ ] **C++20**：广泛支持（GCC 10+, Clang 10+, MSVC 2019 16.11+），有 coroutine、concepts、ranges
- [ ] **C++23**：更新特性（std::expected, std::flat_map），但编译器支持有限
- [ ] **C++17**：最广泛兼容，牺牲一些现代特性

你的偏好？
</details>

<details open>
<summary><b>Q6: ONNX Runtime 集成方式</b></summary>

- [ ] **C API**：`OrtCreateSession/OrtRun`，纯 C，最稳定
- [ ] **C++ API**：`Ort::Session` 等，RAII 包装，需 `#include <onnxruntime_cxx_api.h>`
- [ ] **封装层**：自己写一个轻量封装，隔离 ONNX Runtime 细节

你的偏好？
</details>

<details open>
<summary><b>Q7: 音频预处理路径</b></summary>

当前 Python 版支持：wav, mp3, flac, ogg, m4a, aac, wma, aiff, opus。

C++ 版的音频加载策略：
- [ ] **libsndfile**：支持 WAV/FLAC/OGG，不支持 MP3/AAC（需额外 libmpg123/libfaad）
- [ ] **miniaudio**：单头文件库，支持 WAV/MP3/FLAC，解码+重采样一体化
- [ ] **FFmpeg (libav\*)**：全格式支持但依赖重
- [ ] **dr_libs**：单头文件，dr_wav + dr_mp3 + dr_flac，轻量
- [ ] **先只支持 WAV/FLAC**，后续按需扩展

你的偏好？
</details>

<details open>
<summary><b>Q8: 长音频分段策略</b></summary>

当前 Python 版通过 `SectionConfig` 按设备和时长自动分段处理。C++ 版：
- [ ] **沿用相同策略**：阈值 300s，每段 300s，GPU 不限段数
- [ ] **流式处理**：不预先加载全部音频，边读边处理
- [ ] **固定段长**：简单起见，每段固定 30s/60s

你的偏好？
</details>

---

## 4. Phase 2-3: HTTP 服务端 + 协议实现

### 需要决定的问题

<details open>
<summary><b>Q9: HTTP 框架选型</b></summary>

核心需求：REST API + SSE（Server-Sent Events）+ multipart 文件上传。

| 框架 | 优点 | 缺点 |
|------|------|------|
| **cpp-httplib** | 单头文件，极简，支持 SSE/multipart | 性能中等，不支持 HTTP/2 |
| **Drogon** | 高性能，内置 SSE/WebSocket，协程 | 依赖较多，学习曲线 |
| **Boost.Beast** | Boost 生态，低层可控，HTTP/1.1+WebSocket | 冗长，需要自己实现 SSE |
| **uWebSockets** | 极高吞吐，内置 SSE/WS | C API 风格，文档较少 |
| **Pistache** | 现代 C++，REST 友好 | 社区较小，文档不足 |

你的偏好？或者让我帮你选一个？
</details>

<details open>
<summary><b>Q10: 并发模型</b></summary>

- [ ] **线程池**：类似 Python 版的 ThreadPoolExecutor，简单直接
- [ ] **协程 (C++20 coroutines)**：配合 ASIO，高并发低开销
- [ ] **事件驱动 (libuv/libevent)**：经典的 Node.js 风格
- [ ] **混合**：HTTP 层用协程，推理用线程池

你的偏好？
</details>

<details open>
<summary><b>Q11: JSON 库选型</b></summary>

协议大量使用 JSON。性能关键路径是 SSE 事件序列化和请求解析。

| 库 | 亮点 |
|------|------|
| **nlohmann/json** | 最流行，STL-like API，易用 |
| **simdjson** | 解析极快(GB/s)，只读 |
| **glaze** | C++20 反射，编译期序列化，极快 |
| **rapidjson** | 老牌高性能，但 API 较底层 |
| **boost.json** | Boost 生态，较新但稳定 |

你的偏好？
</details>

<details open>
<summary><b>Q12: 协议版本发送</b></summary>

当前协议规定客户端发送 `X-Protocol-Version` 头，服务端返回。C++ 版：
- [ ] **严格版本检查**：拒绝不兼容的主版本
- [ ] **宽松兼容**：只做头返回，不校验（当前 Python 版策略）

你的偏好？
</details>

---

## 5. Phase 4: 多设备后端

### 需要决定的问题

<details open>
<summary><b>Q13: ONNX Execution Provider 策略</b></summary>

onnxruntime 支持的 EP：
- CPU (默认)
- CUDA (需 CUDA/cuDNN)
- DirectML (Windows GPU)
- CoreML (macOS)
- OpenVINO (Intel)
- TensorRT (NVIDIA，需额外构建)
- ROCm (AMD)

策略：
- [ ] **编译时选择**：不同 CMake target 编译不同的 EP
- [ ] **运行时动态加载**：自动检测可用 EP 并选择
- [ ] **先只做 CPU + CUDA**，后续扩展

你的偏好？
</details>

<details open>
<summary><b>Q14: 设备管理</b></summary>

当前 Python 版有完整的设备抽象层（DeviceBackend + register + acquire/release）。C++ 版：
- [ ] **复刻同样设计**：类似接口的 C++ 实现
- [ ] **简化**：直接按 EP 枚举，不搞抽象层
- [ ] **Plugin 模式**：动态库加载各后端

你的偏好？
</details>

---

## 6. 工程化

### 需要决定的问题

<details open>
<summary><b>Q15: 包管理</b></summary>

- [ ] **vcpkg**：微软维护，包多，Windows 首选
- [ ] **Conan**：跨平台，支持二进制缓存
- [ ] **FetchContent**：CMake 内置，直接从 GitHub 拉源码编译
- [ ] **手动管理**：git submodule 或系统包

你的偏好？
</details>

<details open>
<summary><b>Q16: 项目目录结构</b></summary>

```
rmms-ai-server-onnx/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── server/          # HTTP 服务端
│   ├── engine/          # ONNX 推理引擎
│   ├── protocol/        # 协议模型
│   ├── device/          # 设备后端
│   └── util/            # 工具函数
├── include/             # 公共头文件
├── tests/
├── models/              # .onnx 模型文件 (gitignored)
├── scripts/             # 模型导出脚本 (Python)
├── docker/
└── docs/
```

这样合理吗？还是你有其他偏好？
</details>

<details open>
<summary><b>Q17: 测试策略</b></summary>

- [ ] **单元测试**：每个模块独立测试（GTest/Catch2）
- [ ] **集成测试**：启动 HTTP 服务器，用 Python 脚本测试协议一致性
- [ ] **基准测试**：与 Python 版对比延迟和吞吐
- [ ] **CI/CD**：GitHub Actions 自动构建+测试

你的偏好？
</details>

---

## 7. 迁移与兼容

<details open>
<summary><b>Q18: 与 Python 版共存</b></summary>

- [ ] **独立运行**：两个项目互不依赖
- [ ] **共享配置**：`.env` 格式兼容，可互换部署
- [ ] **共享测试套件**：用同一套协议测试脚本验证两个实现

你的想法？
</details>

<details open>
<summary><b>Q19: 客户端发现</b></summary>

mDNS 广播时，两个服务端共享 `_rmms-ai._tcp` 服务类型：
- [ ] **同类型不同实例名**：客户端可选择
- [ ] **不同子类型**：如 `_rmms-ai-onnx._tcp` 区分

你的想法？
</details>

---

## 8. 你最优先回答的问题

如果问题太多，请至少回答这 **5 个最关键** 的：

1. **Q5**: C++ 标准？（C++20 是自然选择，但确认一下）
2. **Q6**: ONNX Runtime 集成方式？（C API vs C++ API）
3. **Q9**: HTTP 框架选型？（影响最大，决定开发体验）
4. **Q10**: 并发模型？（线程池 vs 协程）
5. **Q13**: EP 策略？（先做 CPU + CUDA？）

先回答了这 5 个我就能开始画更详细的架构图，其他的可以边做边定。

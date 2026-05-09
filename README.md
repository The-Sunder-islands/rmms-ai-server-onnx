# RMMS AI Server ONNX

基于 **ONNX Runtime** + **现代 C++** 的高性能音频分轨推理服务端。是 [rmms-ai-server](https://github.com/The-Sunder-islands/rmms-ai-server) 的实验性衍生项目。

## 目标

- **性能极致**：C++ 原生实现，零 Python 运行时开销
- **依赖最小**：仅需 onnxruntime + HTTP 库，无需 PyTorch 生态
- **跨平台优先**：Linux / Windows / macOS，CPU / CUDA / DirectML / CoreML
- **协议兼容**：完全兼容 [RMMS AI 服务端协议](https://github.com/The-Sunder-islands/rmms-ai-protocol) v1.0.0-draft

## 技术路线

| 阶段 | 目标 | 状态 |
|------|------|------|
| **Phase 0** | 从 PyTorch Demucs 导出 .onnx 模型 | ⏳ |
| **Phase 1** | C++ onnxruntime 单文件推理 PoC | ⏳ |
| **Phase 2** | C++ HTTP 服务端骨架 + REST API | ⏳ |
| **Phase 3** | 完整协议实现 (SSE/Capabilities/Tasks) | ⏳ |
| **Phase 4** | 多设备后端 + 性能优化 + 基准测试 | ⏳ |

## 技术选型 (TBD)

| 组件 | 候选 |
|------|------|
| 构建系统 | CMake |
| HTTP 框架 | cpp-httplib / Drogon / Boost.Beast |
| JSON 解析 | nlohmann/json / simdjson |
| ONNX Runtime | Microsoft onnxruntime (C API) |
| 音频 I/O | libsndfile / dr_libs |
| 测试 | Google Test / Catch2 |

## 与 rmms-ai-server 的关系

- **rmms-ai-server**：Python + FastAPI + Demucs PyTorch，功能完整，适合快速部署
- **rmms-ai-server-onnx**：C++ + ONNX Runtime，性能极致，适合资源受限环境

两者实现相同的 RMMS AI 协议，客户端可无缝切换。

## License

MIT

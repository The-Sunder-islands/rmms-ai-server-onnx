# 纯 OpenCL 推理 Demucs — 可行性分析

## 目标

完全脱离 ONNX Runtime / PyTorch / 任何 ML 框架，用**纯 OpenCL C Kernel** 实现 Demucs (htdemucs / htdemucs_6s) 的推理。

## Demucs 模型算子拆解

```
输入: audio [1, 2, T]  (T = 采样点数, 如 441000)

Stage 1: Encoder (4 层)
  Conv1d (stride=4) → GLU → Conv1d → GLU → ... 
  每层输出 channel 增加, 时间维度减半

Stage 2: Transformer (5 层 x 2 pass = 10 个 transformer block)
  Multi-head Self-Attention (8 heads)
  Layer Normalization
  Feed-Forward (Conv1d + ReLU + Conv1d)
  Cross-attention (between time and frequency domain)

Stage 3: Decoder (4 层)
  ConvTranspose1d (stride=4) + Skip connections from Encoder
  逐层恢复到原始时间维度

Stage 4: ISTFT
  复数 → 时域重建

输出: stems [1, K, T]  (K=4 或 6)
```

## 所需 OpenCL Kernel 清单

### 必须实现的算子

| 算子 | OpenCL 实现难度 | 是否有现成库 | 备注 |
|------|:---:|------|------|
| **Conv1d** | ⭐⭐⭐ | clBlast (GEMM-based) | 需 im2col 转换，或用 clBlast::Gemm |
| **ConvTranspose1d** | ⭐⭐⭐ | clBlast (GEMM-based) | 同上，反向 |
| **GLU** | ⭐ | 自写 | 简单的 split + multiply + sigmoid |
| **LayerNorm** | ⭐⭐ | 自写 | reduce_mean → reduce_var → normalize |
| **Multi-head Attention** | ⭐⭐⭐⭐⭐ | **无现成** | Softmax + MatMul + Transpose + Scale |
| **Feed-Forward (Linear)** | ⭐⭐ | clBlast::Gemm | MatMul + bias + ReLU |
| **Softmax** | ⭐⭐ | 自写 | reduce_max → exp → reduce_sum → div |
| **ReLU / GELU** | ⭐ | 自写 | element-wise |
| **STFT / ISTFT** | ⭐⭐⭐ | clFFT | 复数 FFT + 窗口函数 |
| **复数运算** | ⭐⭐ | 自写 | magnitude, phase, complex mul |

### 模型加载

| 需求 | 难度 | 说明 |
|------|:---:|------|
| 权重文件解析 | ⭐⭐ | 需解析 PyTorch `.pth` state_dict 或转换后的二进制格式 |
| 权重内存管理 | ⭐⭐⭐ | ~600MB FP32 权重需分配在 GPU 显存中 |
| 多 buffer 管理 | ⭐⭐⭐ | 推理中间张量 ~2-4GB 峰值 |

## 可行性评估

### ✅ 能做的

1. **Conv1d/ConvTranspose1d → clBlast GEMM**：卷积可以通过 im2col + GEMM 实现，clBlast 提供高性能 GEMM
2. **STFT/ISTFT → clFFT**：有成熟库，但需要处理复数
3. **逐元素算子**：GLU/ReLU/GELU/LayerNorm 等写 OpenCL kernel 相对直接
4. **Softmax**：标准 kernel，网上有大量参考实现

### ⚠️ 有挑战的

5. **Multi-head Attention**：这是最复杂的算子。需要实现：
   - Q/K/V 投影 (3x GEMM)
   - Q*K^T / sqrt(d_k)
   - Softmax
   - Attention * V
   - Output projection
   
   这些子操作可以拆成多个 GEMM + 逐元素 kernel，但 GPU 内核启动开销会很大。**融合 attention kernel** 需要手写，难度高。

6. **Demucs 的 Cross-Attention**：Demucs 在时域和频域之间做 cross-attention，比标准 self-attention 更复杂

7. **权重转换**：PyTorch state_dict → 自定义二进制格式，需要写一个 Python 脚本做转换

### ❌ 非常困难的

8. **性能调优**：H100 上的 GEMM 是 989 TFLOPS，手写 kernel 能跑到 10-30% 就很不错了。需要：
   - 每个 kernel 做 workgroup 调优
   - 不同 GPU 微架构不同 (NVIDIA vs AMD vs Intel)
   - Memory coalescing / bank conflict 优化
   - 全局内存 → 局部内存传输优化

9. **不同 GPU 的 OpenCL 编译器差异**：NVIDIA/AMD/Intel 的 OpenCL 编译器行为不同，kernel 可能在 A 上跑在 B 上 crash

10. **没有自动微分**：虽然推理不需要，但如果未来想微调模型就完全不行

## 工作量估算

| 阶段 | 内容 | 预估工时 |
|------|------|---------|
| 1 | OpenCL 基础设施：context/queue/buffer 管理 | 1 周 |
| 2 | 逐元素 kernel (GLU/ReLU/LayerNorm/Softmax) | 2 周 |
| 3 | Conv1d (im2col + clBlast GEMM) | 2 周 |
| 4 | Multi-head Attention (6 kernel 组合) | 4 周 |
| 5 | STFT/ISTFT (clFFT 集成) | 2 周 |
| 6 | 权重加载 + 模型图编排 | 2 周 |
| 7 | 验证正确性 (逐层对比 PyTorch 输出) | 3 周 |
| 8 | 性能调优 | 4 周+ |
| **总计** | | **~20 周 (5 个月)** |

## 更现实的替代方案

### 方案 A: TVM + OpenCL (推荐)

Apache TVM 是一个深度学习编译器，可以将 PyTorch/ONNX 模型**编译为 OpenCL 代码**：

```python
import tvm
mod, params = tvm.relay.frontend.from_pytorch(model, input_shape)
target = tvm.target.Target("opencl")
lib = tvm.build(mod, target, params=params)
lib.export_library("demucs_opencl.so")
```

- TVM 自动生成 OpenCL kernel，自动调优 (AutoTVM/AutoScheduler)
- 支持 Fusion pass（合并多个算子减少 kernel 启动次数）
- 仍需要 OpenCL runtime 来执行，但不用手写 kernel
- 学习曲线中等

### 方案 B: ONNX Runtime OpenCL EP + custom ops

ONNX Runtime 的 OpenCL EP 处理大部分算子，只对 ORT 不支持的 Demucs 特定算子写自定义 OpenCL kernel 作为 custom op 插件。

- 兼顾生态和定制
- 只需要写少量 OpenCL kernel
- 仍然依赖 ONNX Runtime

### 方案 C: 纯 OpenCL (你提出的)

- 完全控制、零框架依赖
- 工作量极大 (~5个月)
- 性能未必比 ORT OpenCL EP 好

## 建议

如果目标是"学习和实验 OpenCL" → 建议先用 TVM 生成 OpenCL 代码作为参考，然后逐步替换关键算子。

如果目标是"最快得到可用产品" → 建议方案 B (ORT + 少量 custom OpenCL ops)。

如果目标是"长期打造自己的推理引擎" → 方案 C 可行但需要充足的时间投入。

---

**你的意图是哪一种？**

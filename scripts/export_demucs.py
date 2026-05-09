"""
Phase 0: Export Demucs models to ONNX format.

Usage:
    python scripts/export_demucs.py --model htdemucs_6s --output models/ --fp16

Requires: PyTorch 2.x, demucs, onnx

Strategy:
    torch.export + torch.onnx.dynamo_export (PyTorch 2.x new path)
    - Handles dynamic shapes natively (variable audio length)
    - Better Transformer op coverage

Architecture trace:
    Input:  audio  [B=1, C=2, T=samples]
    Output: stems  [B=1, K=4/6, T=samples]  (vocals,drums,bass,other[,guitar,piano])

Dynamic axes:
    T dimension = variable (any audio length), marked as Dim("time")

Precision:
    --fp32 : standard, larger file, max compatibility
    --fp16 : half-size file, requires GPU EP with FP16 support

Output:
    models/htdemucs_6s_fp32.onnx  (~600MB)
    models/htdemucs_6s_fp16.onnx  (~300MB)
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Export Demucs model to ONNX")
    parser.add_argument("--model", default="htdemucs_6s", choices=["htdemucs", "htdemucs_6s"])
    parser.add_argument("--output", default="models/", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Export FP16 version")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    args = parser.parse_args()

    print(f"[TODO] Export {args.model} to ONNX (opset {args.opset})")
    print("This requires a PyTorch environment with demucs installed.")
    print()
    print("Steps:")
    print("  1. import torch, demucs")
    print("  2. model = demucs.pretrained.get_model(args.model)")
    print("  3. model.eval()")
    print("  4. exported = torch.export.export(model, (dummy_input,), ...)")
    print("  5. torch.onnx.dynamo_export(exported, f'{args.model}.onnx', ...)")
    print()
    print("See plan.md §4 for details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

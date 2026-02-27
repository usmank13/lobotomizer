#!/usr/bin/env python3
"""ResNet50 pruned + quantized for edge deployment.

Downloads the pretrained ResNet50 from torchvision (~98 MB) and compresses it
using L1 structured pruning (30%) + dynamic int8 quantization. The result is
a smaller, faster model suitable for edge devices like Jetson Nano or RPi.

Expected results:
  - Parameter count: ~25.6M → ~25.6M (pruning zeroes weights, doesn't remove params)
  - Size on disk:    ~98 MB → ~25-35 MB (quantization shrinks storage significantly)
  - Latency:         ~2x faster on CPU (dynamic int8 speeds up Linear layers)

Usage:
    python examples/resnet50_edge.py
"""

import torch
import torchvision.models as models

import lobotomizer as lob

# --- Load pretrained ResNet50 ---
# This downloads ~98 MB on first run (cached afterwards).
print("Loading pretrained ResNet50...")
model = models.resnet50(weights="DEFAULT")
model.eval()

# --- Compress: structured pruning + dynamic quantization ---
# L1 structured pruning removes entire channels with the smallest L1 norms,
# which actually reduces compute (unlike unstructured pruning which just zeroes weights).
# Dynamic int8 quantization converts Linear layers to int8 at inference time.
print("Compressing...")
result = lob.compress(model, recipe=[
    lob.Prune(method="l1_structured", sparsity=0.3),
    lob.Quantize(method="dynamic", dtype="qint8"),
])

# --- Print before/after summary ---
print("\n" + result.summary())

# --- Save compressed model ---
output_dir = "compressed_resnet50_edge"
result.save(output_dir)
print(f"\nCompressed model saved to {output_dir}/")
print("Load it later with: model = torch.load(f'{output_dir}/model.pt')")

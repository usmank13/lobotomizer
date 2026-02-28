#!/usr/bin/env python3
"""Compress ResNet50 for edge deployment using Lobotomizer.

Requires: torchvision (ships with torch)
"""

import torch
import torchvision.models as models

import lobotomizer as lob

# --- Load pretrained ResNet50 ---
print("Loading pretrained ResNet50...")
model = models.resnet50(weights="DEFAULT")
model.eval()

# --- Compress: structured pruning + dynamic quantization ---
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

#!/usr/bin/env python3
"""Compress MobileViT for ultra-constrained edge devices using Lobotomizer.

Requires: pip install timm
"""

import sys

try:
    import timm
except ImportError:
    print("This example requires timm: pip install timm")
    sys.exit(1)

import lobotomizer as lob

# --- Load model ---
print("Loading MobileViT-v2-050 from timm...")
model = timm.create_model("mobilevitv2_050", pretrained=True)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

# --- Compress: light pruning + quantization ---
print("\nCompressing (light pruning + quantization)...")
result = lob.compress(model, recipe=[
    lob.Prune(method="l1_unstructured", sparsity=0.15),
    lob.Quantize(method="dynamic", dtype="qint8"),
])

# --- Print before/after summary ---
print("\n" + result.summary())

# --- Save ---
output_dir = "compressed_mobilevit_xxs"
result.save(output_dir)
print(f"\nCompressed model saved to {output_dir}/")

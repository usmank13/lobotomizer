#!/usr/bin/env python3
"""Compress YOLOv8n for real-time edge inference using Lobotomizer.

Requires: pip install ultralytics
"""

import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("This example requires ultralytics: pip install ultralytics")
    sys.exit(1)

import lobotomizer as lob

# --- Load model ---
print("Loading YOLOv8n (~6.2 MB)...")
yolo = YOLO("yolov8n.pt")
model = yolo.model
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

# --- Compress: prune + quantize for edge ---
print("\nCompressing for edge deployment...")
result = lob.compress(model, recipe=[
    lob.Prune(method="l1_structured", sparsity=0.3),
    lob.Quantize(method="dynamic", dtype="qint8"),
])

# --- Print before/after summary ---
print("\n" + result.summary())

# --- Save ---
output_dir = "compressed_yolov8n_edge"
result.save(output_dir)
print(f"\nCompressed model saved to {output_dir}/")

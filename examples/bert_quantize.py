#!/usr/bin/env python3
"""Compress BERT-base for faster CPU inference using Lobotomizer.

Requires: pip install transformers
"""

import sys

try:
    from transformers import BertModel
except ImportError:
    print("This example requires transformers: pip install transformers")
    sys.exit(1)

import lobotomizer as lob

# --- Load model ---
print("Loading BERT-base-uncased from Hugging Face (~440 MB on first run)...")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

# --- Compress: dynamic int8 quantization ---
print("\nQuantizing...")
result = lob.compress(model, recipe=[
    lob.Quantize(method="dynamic", dtype="qint8"),
])

# --- Print before/after summary ---
print("\n" + result.summary())

# --- Save ---
output_dir = "compressed_bert_quantized"
result.save(output_dir)
print(f"\nCompressed model saved to {output_dir}/")

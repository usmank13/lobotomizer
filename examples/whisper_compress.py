#!/usr/bin/env python3
"""Compress Whisper small for on-device transcription using Lobotomizer.

Requires: pip install openai-whisper
"""

import sys

try:
    import whisper
except ImportError:
    print("This example requires whisper: pip install openai-whisper")
    sys.exit(1)

import lobotomizer as lob

# --- Load model ---
print("Loading Whisper small (~461 MB download on first run)...")
model = whisper.load_model("small")
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

# --- Compress: prune + quantize ---
print("\nCompressing...")
result = lob.compress(model, recipe=[
    lob.Prune(method="l1_unstructured", sparsity=0.25),
    lob.Quantize(method="dynamic", dtype="qint8"),
])

# --- Print before/after summary ---
print("\n" + result.summary())

# --- Save ---
output_dir = "compressed_whisper_small"
result.save(output_dir)
print(f"\nCompressed model saved to {output_dir}/")

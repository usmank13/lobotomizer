#!/usr/bin/env python3
"""BERT-base quantized for faster CPU inference.

Applies dynamic int8 quantization to BERT-base (~110M params), which speeds
up all Linear layers on CPU with minimal accuracy loss. This is the simplest
and most effective compression for transformer models serving on CPU.

If `transformers` is installed, loads the real BERT-base-uncased model.
Otherwise, creates a dummy model with the same architecture (~110M params)
so you can see the compression pipeline in action.

Expected results:
  - Size on disk: ~440 MB → ~170-180 MB (~60% reduction)
  - CPU latency:  ~1.5-2x faster (int8 matmuls)
  - Accuracy:     minimal degradation for most NLP tasks

Usage:
    python examples/bert_quantize.py
"""

import torch
import torch.nn as nn

import lobotomizer as lob


def make_dummy_bert(
    num_layers: int = 12,
    hidden_size: int = 768,
    num_heads: int = 12,
    intermediate_size: int = 3072,
    vocab_size: int = 30522,
) -> nn.Module:
    """Create a dummy model matching BERT-base architecture (~110M params).

    This mimics the structure of BERT without needing the `transformers` package.
    The real model would be loaded with:
        from transformers import BertModel
        model = BertModel.from_pretrained("bert-base-uncased")
    """
    class BertLayer(nn.Module):
        def __init__(self):
            super().__init__()
            # Self-attention
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.attn_out = nn.Linear(hidden_size, hidden_size)
            self.attn_norm = nn.LayerNorm(hidden_size)
            # Feed-forward
            self.ff_in = nn.Linear(hidden_size, intermediate_size)
            self.ff_out = nn.Linear(intermediate_size, hidden_size)
            self.ff_norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            # Simplified — no real attention, just the linear transforms
            attn = self.attn_out(self.query(x) + self.key(x) + self.value(x))
            x = self.attn_norm(x + attn)
            ff = self.ff_out(torch.relu(self.ff_in(x)))
            return self.ff_norm(x + ff)

    class DummyBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([BertLayer() for _ in range(num_layers)])
            self.pooler = nn.Linear(hidden_size, hidden_size)

        def forward(self, input_ids):
            x = self.embeddings(input_ids)
            for layer in self.layers:
                x = layer(x)
            return torch.tanh(self.pooler(x[:, 0]))

    return DummyBert()


# --- Load model ---
try:
    from transformers import BertModel
    print("Loading BERT-base-uncased from Hugging Face (~440 MB on first run)...")
    model = BertModel.from_pretrained("bert-base-uncased")
    print("Loaded real BERT model.")
except ImportError:
    print("transformers not installed — using dummy BERT-base model (~110M params).")
    print("Install transformers for the real model: pip install transformers")
    model = make_dummy_bert()

model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

# --- Compress: dynamic int8 quantization ---
# Dynamic quantization is ideal for transformer models on CPU:
# - No calibration data needed
# - Converts Linear layer weights to int8
# - Activations quantized on-the-fly during inference
# - Minimal accuracy impact on NLP tasks
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

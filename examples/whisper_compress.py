#!/usr/bin/env python3
"""Whisper small compressed for on-device transcription.

Applies pruning + quantization to OpenAI's Whisper small model (~244M params)
to make it feasible for on-device transcription on phones or edge hardware.

If the `whisper` package is installed, loads the real model.
Otherwise, creates a dummy model matching Whisper small's architecture.

Expected results:
  - Size on disk: ~960 MB → ~300-400 MB
  - Suitable for mobile/edge deployment with ONNX export

Usage:
    python examples/whisper_compress.py
"""

import torch
import torch.nn as nn

import lobotomizer as lob


def make_dummy_whisper_small(
    n_mels: int = 80,
    n_audio_ctx: int = 1500,
    n_audio_state: int = 768,
    n_audio_head: int = 12,
    n_audio_layer: int = 12,
    n_vocab: int = 51865,
    n_text_ctx: int = 448,
    n_text_state: int = 768,
    n_text_head: int = 12,
    n_text_layer: int = 12,
) -> nn.Module:
    """Create a dummy model matching Whisper small architecture (~244M params).

    The real model would be loaded with:
        import whisper
        model = whisper.load_model("small")
    """
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_head, cross_attention=False):
            super().__init__()
            self.attn_q = nn.Linear(d_model, d_model)
            self.attn_k = nn.Linear(d_model, d_model, bias=False)
            self.attn_v = nn.Linear(d_model, d_model)
            self.attn_out = nn.Linear(d_model, d_model)
            self.attn_norm = nn.LayerNorm(d_model)
            if cross_attention:
                self.cross_q = nn.Linear(d_model, d_model)
                self.cross_k = nn.Linear(d_model, d_model, bias=False)
                self.cross_v = nn.Linear(d_model, d_model)
                self.cross_out = nn.Linear(d_model, d_model)
                self.cross_norm = nn.LayerNorm(d_model)
            self.cross_attention = cross_attention
            self.ff_in = nn.Linear(d_model, d_model * 4)
            self.ff_out = nn.Linear(d_model * 4, d_model)
            self.ff_norm = nn.LayerNorm(d_model)

        def forward(self, x, xa=None):
            h = self.attn_norm(x)
            h = self.attn_out(self.attn_v(h))
            x = x + h
            if self.cross_attention and xa is not None:
                h = self.cross_norm(x)
                h = self.cross_out(self.cross_v(xa))
                x = x + h
            h = self.ff_norm(x)
            x = x + self.ff_out(torch.gelu(self.ff_in(h)))
            return x

    class DummyWhisper(nn.Module):
        def __init__(self):
            super().__init__()
            # Audio encoder
            self.encoder_conv1 = nn.Conv1d(n_mels, n_audio_state, 3, padding=1)
            self.encoder_conv2 = nn.Conv1d(n_audio_state, n_audio_state, 3, stride=2, padding=1)
            self.encoder_pos = nn.Embedding(n_audio_ctx, n_audio_state)
            self.encoder_blocks = nn.ModuleList([
                TransformerBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)
            ])
            self.encoder_norm = nn.LayerNorm(n_audio_state)
            # Text decoder
            self.decoder_embed = nn.Embedding(n_vocab, n_text_state)
            self.decoder_pos = nn.Embedding(n_text_ctx, n_text_state)
            self.decoder_blocks = nn.ModuleList([
                TransformerBlock(n_text_state, n_text_head, cross_attention=True)
                for _ in range(n_text_layer)
            ])
            self.decoder_norm = nn.LayerNorm(n_text_state)

        def forward(self, mel, tokens):
            # Encoder
            x = torch.gelu(self.encoder_conv1(mel))
            x = torch.gelu(self.encoder_conv2(x))
            x = x.permute(0, 2, 1)
            for block in self.encoder_blocks:
                x = block(x)
            x = self.encoder_norm(x)
            # Decoder (simplified)
            h = self.decoder_embed(tokens)
            for block in self.decoder_blocks:
                h = block(h, xa=x)
            return self.decoder_norm(h)

    return DummyWhisper()


# --- Load model ---
try:
    import whisper
    print("Loading Whisper small (~461 MB download on first run)...")
    model = whisper.load_model("small")
    print("Loaded real Whisper model.")
except ImportError:
    print("whisper not installed — using dummy Whisper small model (~244M params).")
    print("Install for the real model: pip install openai-whisper")
    model = make_dummy_whisper_small()

model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

# --- Compress: prune + quantize ---
# For speech models, moderate pruning + quantization gives good compression
# without significantly degrading transcription quality.
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
print("For production, consider ONNX export for mobile deployment.")

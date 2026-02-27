#!/usr/bin/env python3
"""MobileViT further compressed — squeezing already-efficient models.

Demonstrates that even models designed for mobile (MobileViT, EfficientViT)
can be further compressed with pruning + quantization. Useful when targeting
ultra-constrained devices or reducing latency below what the architecture
alone provides.

If `timm` (PyTorch Image Models) is installed, loads MobileViT-XXS.
Otherwise, creates a dummy model matching MobileViT-XXS (~1.3M params).

Expected results:
  - Size on disk: ~5.2 MB → ~2-3 MB
  - Already-efficient models still benefit from int8 quantization
  - Light pruning (15%) avoids accuracy cliff on small models

Usage:
    python examples/mobilevit_compress.py
"""

import torch
import torch.nn as nn

import lobotomizer as lob


def make_dummy_mobilevit_xxs() -> nn.Module:
    """Create a dummy model matching MobileViT-XXS architecture (~1.3M params).

    The real model would be loaded with:
        import timm
        model = timm.create_model("mobilevitv2_050", pretrained=True)
    """
    class MobileNetBlock(nn.Module):
        """Inverted residual block (MobileNetV2 style)."""
        def __init__(self, in_ch, out_ch, expand=4):
            super().__init__()
            mid = in_ch * expand
            self.block = nn.Sequential(
                # Pointwise expand
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(inplace=True),
                # Depthwise
                nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(inplace=True),
                # Pointwise project
                nn.Conv2d(mid, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.use_residual = (in_ch == out_ch)

        def forward(self, x):
            out = self.block(x)
            return (x + out) if self.use_residual else out

    class MobileViTBlock(nn.Module):
        """Simplified MobileViT block — local + global processing."""
        def __init__(self, ch, d_model=64, n_heads=1, n_layers=2):
            super().__init__()
            self.local_conv = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            )
            self.proj_in = nn.Conv2d(ch, d_model, 1)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, batch_first=True),
                num_layers=n_layers,
            )
            self.proj_out = nn.Conv2d(d_model, ch, 1)

        def forward(self, x):
            local = self.local_conv(x)
            B, C, H, W = local.shape
            t = self.proj_in(local)
            t = t.flatten(2).permute(0, 2, 1)  # B, HW, d
            t = self.transformer(t)
            t = t.permute(0, 2, 1).view(B, -1, H, W)
            return x + self.proj_out(t)

    class DummyMobileViT(nn.Module):
        """Simplified MobileViT-XXS (~1.3M params)."""
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.SiLU(inplace=True),
            )
            # MobileNet stages
            self.stage1 = MobileNetBlock(16, 16)
            self.stage2 = nn.Sequential(MobileNetBlock(16, 24), MobileNetBlock(24, 24))
            self.stage3 = nn.Sequential(MobileNetBlock(24, 48), MobileViTBlock(48, 64, 1, 2))
            self.stage4 = nn.Sequential(MobileNetBlock(48, 64), MobileViTBlock(64, 80, 1, 2))
            self.stage5 = nn.Sequential(MobileNetBlock(64, 80), MobileViTBlock(80, 96, 1, 2))
            # Head
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(80, 1000),
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            return self.head(x)

    return DummyMobileViT()


# --- Load model ---
try:
    import timm
    print("Loading MobileViT-XXS from timm...")
    model = timm.create_model("mobilevitv2_050", pretrained=True)
    print("Loaded real MobileViT model.")
except ImportError:
    print("timm not installed — using dummy MobileViT-XXS model (~1.3M params).")
    print("Install for the real model: pip install timm")
    model = make_dummy_mobilevit_xxs()

model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

# --- Compress: light pruning + quantization ---
# For already-efficient models, use conservative pruning (15%) to avoid
# degrading accuracy, but still apply full int8 quantization for size/speed.
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
print("Even already-efficient architectures benefit from quantization!")

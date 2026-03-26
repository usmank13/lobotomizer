# 🧠 Lobotomizer

**Take any `nn.Module`. ~~Lobotomize it.~~ Make it smaller, faster, cheaper.**

Composable model compression for PyTorch. Run pipelines to use compression techniques like quantization, pruning, knowledge distillation, and more with a one-liner, an explicit pipeline, or the CLI.

Over time will try to serve as an easy-to-use collection of popular techniques to help with R&D in the field.

## Installation

```bash
pip install lobotomizer
```

Optional extras:

```bash
pip install lobotomizer[all]       # everything
pip install lobotomizer[dev]       # pytest
pip install lobotomizer[pruning]   # torch-pruning
pip install lobotomizer[quantize]  # bitsandbytes
```

Or install from source:

```bash
git clone https://github.com/usmank13/lobotomizer.git
cd lobotomizer
pip install -e ".[dev]"
```

## Quick Start

### One-liner

```python
import lobotomizer as lob

result = lob.compress(model, recipe="balanced")
print(result.summary())
result.save("compressed/")
```

### Explicit pipeline

```python
import lobotomizer as lob

result = lob.Pipeline([
    lob.Prune(method="l1_unstructured", sparsity=0.4),
    lob.Quantize(method="dynamic"),
]).run(model)

print(result.summary())
```

### CLI

```bash
# Compress with a recipe
lobotomize model.pt --recipe balanced --output compressed/

# Compress with explicit options
lobotomize model.pt --prune l1_unstructured --sparsity 0.3 --quantize dynamic -o out/

# Profile only
lobotomize model.pt --profile-only --input-shape "1,3,224,224"

# List available recipes
lobotomize --list-recipes
```

### Summary output

Real results from compressing [Whisper-tiny](https://huggingface.co/openai/whisper-tiny) (39M params) with dynamic int8 quantization:

```
┌───────────────────────┬────────────┬────────────┬────────┐
│ Metric                │ Before     │ After      │ Δ      │
├───────────────────────┼────────────┼────────────┼────────┤
│ param_count           │ 37,760,640 │ 37,760,640 │ +0.0%  │
│ param_count_trainable │ 37,760,640 │ 21,245,568 │ -43.7% │
│ size_mb               │ 144.10     │ 97.02      │ -32.7% │
└───────────────────────┴────────────┴────────────┴────────┘
```

## Available Stages

### Pruning

| Method | Description |
|---|---|
| `l1_unstructured` | Remove weights with smallest L1 magnitude |
| `random_unstructured` | Remove random weights |
| `l1_structured` | Remove entire channels by L1 norm (Conv2d) — mask-based |
| `random_structured` | Remove random channels (Conv2d) — mask-based |

### Quantization

| Method | Description | Extra Deps |
|---|---|---|
| `dynamic` / `dynamic_int8` | Dynamic int8 quantization (Linear layers) | — |
| `static` / `static_int8` | Static int8 quantization (requires calibration data) | — |
| `int4_weight_only` | INT4 weight-only quantization via torchao | — |
| `gptq` | GPTQ quantization (HuggingFace models) | `auto-gptq` |
| `awq` | AWQ quantization (HuggingFace models) | `autoawq` |

Methods with optional dependencies will raise a clear `ImportError` if the dep is missing.

```python
# INT4 weight-only (no extra deps needed)
lob.Quantize(method="int4_weight_only")

# GPTQ with custom settings
lob.Quantize(method="gptq", bits=4, group_size=128)

# AWQ
lob.Quantize(method="awq", w_bit=4, q_group_size=128)
```

### Knowledge Distillation

Train a compressed student model to mimic the original teacher:

```python
import lobotomizer as lob

# Logit-based distillation (Hinton-style)
result = lob.Pipeline([
    lob.StructuredPrune(sparsity=0.3),
    lob.Distill(method="logit", temperature=4.0, epochs=5, lr=1e-4),
]).run(model, training_data=train_loader)

# Feature matching — align intermediate representations
result = lob.Pipeline([
    lob.Distill(
        method="feature",
        feature_layers={"fc1": "fc1", "fc2": "fc2"},
        epochs=10,
    ),
]).run(model, training_data=train_loader)

# Both logit + feature distillation
result = lob.Pipeline([
    lob.Distill(method="both", alpha=0.7, temperature=4.0, epochs=10),
]).run(model, training_data=train_loader)
```

| Parameter | Description |
|---|---|
| `method` | `"logit"`, `"feature"`, or `"both"` |
| `temperature` | Softmax temperature for logit KD (default: 4.0) |
| `alpha` | KD loss weight; `1-alpha` goes to task loss (default: 1.0) |
| `feature_layers` | `dict[str,str]` mapping student→teacher layer names (auto-matched if `None`) |
| `teacher` | `nn.Module`, file path, or `None` (uses original model) |
| `epochs` | Training epochs (default: 5) |

YAML recipe:

```yaml
stages:
  - type: structured_prune
    sparsity: 0.3
  - type: distill
    method: logit
    temperature: 4.0
    epochs: 5
```

## Recipes

Recipes are YAML files that define a sequence of stages:

```yaml
name: balanced
description: "Structured pruning + dynamic int8 quantization"
stages:
  - type: prune
    method: l1_unstructured
    sparsity: 0.25
  - type: quantize
    method: dynamic
    dtype: qint8
```

Built-in recipes: `balanced`, `aggressive`

Use custom recipes: `lob.compress(model, recipe="path/to/recipe.yaml")`

### Structured Pruning (Physical)

`StructuredPrune` physically removes neurons and filters, producing smaller architectures with real speedups — no sparse hardware needed.

| Layer Type | What's Removed | Propagation |
|---|---|---|
| `nn.Linear` | Output neurons (rows) | Downstream Linear input columns |
| `nn.Conv2d` | Output filters (dim 0) | Downstream Conv2d input channels + BatchNorm2d |

```python
import lobotomizer as lob

# Prune 30% of neurons/filters from all layers
result = lob.Pipeline([
    lob.StructuredPrune(sparsity=0.3, criterion="l1"),
]).run(model)

# Exclude specific layers, protect output interface
result = lob.Pipeline([
    lob.StructuredPrune(
        sparsity=0.4,
        protect_output=True,        # default: don't prune output layers
        exclude_layers={"layer4"},   # skip specific layers
    ),
]).run(model)
```

**Note:** Conv2d chains are only detected within `nn.Sequential` containers to avoid breaking skip/residual connections. Models with residual blocks (ResNet, etc.) will have their conv layers automatically protected.

### Low-Rank Decomposition

Replace weight matrices with low-rank SVD approximations, factoring a single layer into two smaller layers:

```python
import lobotomizer as lob

# Fraction-based: keep 50% of singular values
result = lob.Pipeline([
    lob.LowRank(rank_fraction=0.5),
]).run(model)

# Energy-based: keep enough singular values for 90% spectral energy
result = lob.Pipeline([
    lob.LowRank(rank_fraction=0.9, criterion="energy"),
]).run(model)
```

| Parameter | Description |
|---|---|
| `rank_fraction` | Fraction of rank to keep (fraction mode) or energy to preserve (energy mode) |
| `criterion` | `"fraction"` (default) or `"energy"` |
| `min_rank` | Minimum rank to preserve (default: 1) |
| `min_compression` | Only decompose if result is ≤ this fraction of original params (default: 0.9) |
| `exclude_layers` | Named layers to skip |

Works on both `nn.Linear` and `nn.Conv2d` layers. Conv2d layers are decomposed into a spatial conv (original kernel) followed by a 1×1 conv.

YAML recipe:

```yaml
stages:
  - type: low_rank
    rank_fraction: 0.5
    criterion: energy
```

### ONNX Export

Export compressed models to ONNX for deployment with ONNX Runtime, TensorRT, etc:

```python
import lobotomizer as lob

result = lob.compress(model, recipe="balanced")

# From Result object
result.to_onnx("model.onnx", input_shape=(1, 3, 224, 224))

# Standalone function
lob.to_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

CLI:

```bash
lobotomize model.pt --recipe balanced --export-onnx model.onnx --input-shape "1,3,224,224"
```

Optional: install `onnx` for verification, `onnxsim` for graph simplification.

## How It Works

```
Model → [Stage 1] → [Stage 2] → ... → Result
         Prune        Quantize
```

1. **Pipeline** — a list of `Stage` objects run sequentially
2. **Stages** — each stage (`Prune`, `Quantize`) transforms the model in-place on a deep copy
3. **Profiler** — measures param count, size, and FLOPs before/after each stage
4. **Result** — holds the compressed model, profiles, and stage history; can save and summarize
5. **Recipes** — YAML configs that build pipelines from named stages

The original model is never mutated.

## Examples

See [`examples/`](examples/) for complete, runnable scripts:

| Script | What it does |
|---|---|
| [`resnet50_edge.py`](examples/resnet50_edge.py) | ResNet50 pruned + quantized for edge deployment |
| [`bert_quantize.py`](examples/bert_quantize.py) | BERT-base quantized for faster CPU inference |
| [`whisper_compress.py`](examples/whisper_compress.py) | Whisper small compressed for on-device transcription |
| [`yolo_edge.py`](examples/yolo_edge.py) | YOLOv8n compressed for real-time edge inference |
| [`mobilevit_compress.py`](examples/mobilevit_compress.py) | MobileViT further compressed for ultra-constrained devices |

Each script is self-contained and falls back to a dummy model if optional dependencies aren't installed. Note: not all of these are fully tested yet.

## Roadmap

- [x] Prune, Quantize, Pipeline, profiler, recipes, CLI
- [x] Knowledge distillation (logit, feature)
- [x] Low-rank decomposition (SVD)
- [ ] Sparsity techniques
- [x] ONNX export (API, CLI, Result method)
- [ ] Hardware profiling and optimization
- [ ] Search & automation (sweeps, finding lobotomization pipelines to hit given targets)

Over time: progressively support and wrap more techniques, layer types, tools, etc. 

## Contributing

Contributions welcome! Let's grow the lobotomization movement.

1. Fork & clone
2. `pip install -e ".[dev]"`
3. `pytest`
4. PR

## License

MIT

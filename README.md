# 🧠 Lobotomizer

**Take any `nn.Module`. ~~Lobotomize it.~~ Make it smaller, faster, cheaper.**

Composable model compression for PyTorch. Prune, quantize, and profile models with a one-liner, an explicit pipeline, or the CLI.

## Installation

```bash
pip install lobotomizer

# With optional extras
pip install lobotomizer[all]       # everything
pip install lobotomizer[dev]       # pytest
pip install lobotomizer[pruning]   # torch-pruning
pip install lobotomizer[quantize]  # bitsandbytes
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
| `l1_structured` | Remove entire channels by L1 norm (Conv2d) |
| `random_structured` | Remove random channels (Conv2d) |

### Quantization

| Method | Description |
|---|---|
| `dynamic` | Dynamic int8 quantization (Linear layers) |
| `static` | Static int8 quantization (requires calibration data) |

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

Built-in recipes: `balanced`

Use custom recipes: `lob.compress(model, recipe="path/to/recipe.yaml")`

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

Each script is self-contained and falls back to a dummy model if optional dependencies aren't installed.

## Roadmap

- [x] **v0.1** — Prune, Quantize, Pipeline, profiler, recipes, CLI
- [x] **v0.2** — Knowledge distillation (logit, feature, combined)
- [ ] **v0.2.1** — ONNX export
- [ ] **v0.3** — Structured pruning with fine-tuning, NAS integration
- [ ] **v0.4** — Auto-compress (search over recipe space)
- [ ] **v0.5** — Hardware-aware compression targets

## Contributing

1. Fork & clone
2. `pip install -e ".[dev]"`
3. `pytest`
4. PR

## License

MIT

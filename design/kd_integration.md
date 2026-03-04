# Knowledge Distillation Integration — Spec

**Status:** Ready for implementation  
**Last updated:** 2026-03-01

---

## 1. Overview

Distillation is fundamentally different from Prune/Quantize — those are single-model
transforms; KD is a two-model training loop. We stretch the Stage interface to keep
pipeline composability, but we DON'T build a full training framework.

```python
Pipeline([
    StructuredPrune(sparsity=0.3),   # shrink it
    Distill(method="both"),           # recover quality from teacher
    Quantize(method="dynamic"),       # compress further
]).run(model, training_data=train_loader)
```

---

## 2. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | `alpha=1.0` default (pure KD, no labels) | Teacher IS the ground truth. Task loss is the power-user path. |
| 2 | Auto-insert feature adapters | Generic `nn.Linear` projection when dims differ, `nn.Identity` when they match. Trained during distillation, **discarded after**. |
| 3 | Custom adapters via `adapter=None\|str\|type\|callable` | Progressive disclosure — simple cases simple, complex cases possible. |
| 4 | Teacher defaults to `context.original_model` | Pipeline already deep-copies it. 90% use case is "recover from pre-prune original." |
| 5 | Feature auto-match by name | Works for prune→distill (same arch, smaller). Explicit `feature_layers` dict when architectures differ. |
| 6 | Sensible training defaults + escape hatch | AdamW, lr=1e-4, 5 epochs default. Power users pass `optimizer_cls`, `scheduler_cls`. Batch size is the DataLoader's problem. |
| 7 | Forward hooks for feature capture | Lightweight torchdistill pattern. No model modification needed. |

### What We're NOT Doing
- Full training framework (that's torchdistill/Lightning territory)
- Every KD variant (attention transfer, contrastive, relational — future stages if needed)
- Distributed training (single device for now)
- Automatic architecture search for feature pairs

---

## 3. File Structure

```
stages/
    distill.py          # Distill stage (the main Stage subclass)
    distill_losses.py   # KDLoss, FeatureMatchingLoss, CombinedKDLoss
    hooks.py            # ForwardHookManager
    adapters.py         # FeatureAdapter, adapter registry, resolve_adapter()
registry.py             # Stage + adapter registries (new)
```

---

## 4. Stage & Adapter Registries (registry.py)

Stages are currently instantiated directly. We add a registry so custom stages
and adapters work with recipes and CLI by name.

```python
_STAGE_REGISTRY: dict[str, type[Stage]] = {}
_ADAPTER_REGISTRY: dict[str, type] = {}

def register_stage(name: str):
    """Decorator to register a custom Stage class."""
    def decorator(cls):
        if not issubclass(cls, Stage):
            raise TypeError(f"{cls} must subclass Stage")
        _STAGE_REGISTRY[name] = cls
        return cls
    return decorator

def register_adapter(name: str):
    """Decorator to register a custom adapter class."""
    def decorator(cls):
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator

def get_stage(name: str) -> type[Stage]:
    if name not in _STAGE_REGISTRY:
        raise KeyError(f"Unknown stage '{name}'. Available: {sorted(_STAGE_REGISTRY)}")
    return _STAGE_REGISTRY[name]

def get_adapter(name: str) -> type:
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(f"Unknown adapter '{name}'. Available: {sorted(_ADAPTER_REGISTRY)}")
    return _ADAPTER_REGISTRY[name]

# Built-in stages auto-register on import
# register_stage("prune")(Prune)
# register_stage("structured_prune")(StructuredPrune)
# register_stage("quantize")(Quantize)
# register_stage("distill")(Distill)
```

User-facing:
```python
import lobotomizer as lob

@lob.register_stage("my_custom_stage")
class MyStage(lob.Stage): ...

@lob.register_adapter("vision_transformer")
class ViTAdapter(lob.FeatureAdapter): ...
```

---

## 5. ForwardHookManager (hooks.py)

Registers forward hooks on named modules to capture intermediate outputs.
No model modification needed.

```python
class ForwardHookManager:
    def __init__(self):
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[RemovableHandle] = []

    def register(self, model: nn.Module, layer_names: list[str]):
        for name in layer_names:
            module = dict(model.named_modules())[name]
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(mod, inp, out):
            self._features[name] = out
        return hook_fn

    def pop_features(self) -> dict[str, torch.Tensor]:
        feats = dict(self._features)
        self._features.clear()
        return feats

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
```

---

## 6. Feature Adapters (adapters.py)

### FeatureAdapter

Generic linear projection to align student→teacher feature dims.

- Auto-inserted when dims differ after structured pruning
- `nn.Identity()` when dims match (zero overhead)
- `nn.Linear(student_dim, teacher_dim, bias=False)` when dims differ
- **Trained during distillation, discarded after** — scaffolding, not part of final model

**Limitation:** Doesn't handle spatial dimension mismatches or complex structures
(multi-head attention, token-level alignment). Users with exotic architectures
pass a custom adapter.

```python
class FeatureAdapter(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        if student_dim == teacher_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x):
        return self.proj(x)

    @classmethod
    def auto_build(cls, student_module: nn.Module, teacher_module: nn.Module):
        s_dim = _infer_output_dim(student_module)
        t_dim = _infer_output_dim(teacher_module)
        return cls(s_dim, t_dim)
```

### _infer_output_dim

Best-effort dimension inference:
- `nn.Linear` → `out_features`
- `nn.Conv{1,2,3}d` → `out_channels`
- Anything with `out_features` attr → use it
- Otherwise → raise with message to pass custom adapter

### resolve_adapter

Progressive disclosure — accepts multiple input types:

| Input | Behavior |
|-------|----------|
| `None` | Auto `FeatureAdapter` (generic linear projection) |
| `str` | Look up registered adapter by name |
| `type` | Instantiate via `auto_build` |
| `nn.Module` | Use directly |
| `callable` | Wrap as `FunctionalAdapter` |

```python
class FunctionalAdapter(nn.Module):
    """Wraps a user callable for custom loss computation."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, student_feat, teacher_feat):
        return self.fn(student_feat, teacher_feat)
```

---

## 7. Loss Functions (distill_losses.py)

### KDLoss — Hinton-style logit distillation

```python
class KDLoss(nn.Module):
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits):
        s = F.log_softmax(student_logits / self.T, dim=-1)
        t = F.softmax(teacher_logits / self.T, dim=-1)
        return F.kl_div(s, t, reduction="batchmean") * (self.T ** 2)
```

### FeatureMatchingLoss — FitNets-style intermediate matching

```python
class FeatureMatchingLoss(nn.Module):
    def __init__(self, adapters: dict[str, nn.Module] | None = None):
        super().__init__()
        self.adapters = nn.ModuleDict(adapters or {})

    def forward(self, student_features: dict, teacher_features: dict):
        loss = 0.0
        for key in student_features:
            s_feat = student_features[key]
            t_feat = teacher_features[key]
            if key in self.adapters:
                s_feat = self.adapters[key](s_feat)
            loss += F.mse_loss(s_feat, t_feat)
        return loss / max(len(student_features), 1)
```

### CombinedKDLoss — Logit + Feature + optional task loss

```python
class CombinedKDLoss(nn.Module):
    def __init__(self, kd_loss, feature_loss=None, alpha=1.0, beta=0.5):
        super().__init__()
        self.kd_loss = kd_loss
        self.feature_loss = feature_loss
        self.alpha = alpha    # weight of KD logit loss
        self.beta = beta      # weight of feature loss

    def forward(self, student_logits, teacher_logits,
                student_features=None, teacher_features=None,
                task_loss=None):
        loss = self.alpha * self.kd_loss(student_logits, teacher_logits)
        if self.feature_loss and student_features and teacher_features:
            loss += self.beta * self.feature_loss(student_features, teacher_features)
        if task_loss is not None:
            loss += (1 - self.alpha) * task_loss
        return loss
```

---

## 8. Distill Stage (distill.py)

### Constructor

```python
class Distill(Stage):
    def __init__(
        self,
        teacher: nn.Module | str | None = None,
        method: str = "logit",           # "logit" | "feature" | "both"
        temperature: float = 4.0,
        alpha: float = 1.0,              # 1.0 = pure KD, <1.0 mixes task loss
        feature_layers: dict[str, str] | None = None,
        adapter: None | str | type | callable = None,
        task_loss_fn: callable | None = None,
        epochs: int = 5,
        lr: float = 1e-4,
        optimizer_cls: type | None = None,       # default: AdamW
        optimizer_kwargs: dict | None = None,
        scheduler_cls: type | None = None,
        scheduler_kwargs: dict | None = None,
        log_every: int = 50,
    ):
```

### Teacher Resolution
1. `teacher=None` → use `context.original_model` (pre-pipeline snapshot)
2. `teacher=nn.Module` → use directly
3. `teacher="path/to/model.pt"` → `torch.load()`

### Training Data Resolution
1. Check `context.training_data` first (new field)
2. Fall back to `context.calibration_data` (existing field, dual-use)
3. Raise if neither provided

### validate()
- Warn if `alpha < 1.0` but no `task_loss_fn`
- Warn if feature method with no `feature_layers` (will auto-match)

### apply() Flow

```
1. Resolve teacher → freeze, eval mode
2. Resolve training data
3. If feature/both:
   a. Resolve feature layer pairs (explicit or auto-match by name)
   b. Register ForwardHookManager on student + teacher
   c. Build adapters via resolve_adapter() for each pair
   d. Create FeatureMatchingLoss with adapters
4. Create KDLoss
5. Set up optimizer (student params + adapter params if feature KD)
6. Training loop:
   for epoch in range(epochs):
       for batch in data:
           teacher_out = teacher(x)          # no_grad
           student_out = student(x)
           loss = kd_loss(student, teacher)
           if feature: loss += feature_loss(s_feats, t_feats)
           if alpha < 1.0: loss = alpha * loss + (1-alpha) * task_loss
           backward, step
       scheduler.step() if scheduler
7. Cleanup: remove hooks, discard adapters
8. Return model in eval mode
```

### Feature Layer Auto-Matching

When `feature_layers=None`:
- Find all `nn.Linear` and `nn.Conv2d` layers in both student and teacher
- Match by name (intersection)
- Raise if no common names found

This works for the prune→distill case (same architecture, just smaller).
Cross-architecture distillation requires explicit mapping.

---

## 9. PipelineContext Extension

```python
@dataclass
class PipelineContext:
    original_model: nn.Module              # existing — default teacher
    eval_fn: Callable | None = None
    target_constraints: dict = field(default_factory=dict)
    history: list[StageResult] = field(default_factory=list)
    calibration_data: DataLoader | None = None
    device: str = "cpu"
    training_data: DataLoader | None = None    # NEW
```

Pipeline.run() gains `training_data` kwarg:
```python
def run(self, model, *, training_data=None, ...):
```

---

## 10. Public API Surface

### Top-level exports (lobotomizer/__init__.py)

```python
from lobotomizer.stages.distill import Distill
from lobotomizer.stages.adapters import FeatureAdapter, register_adapter
from lobotomizer.registry import register_stage
```

### Usage Examples

**Minimal — logit KD after pruning:**
```python
result = lob.Pipeline([
    lob.StructuredPrune(sparsity=0.3),
    lob.Distill(epochs=10),
]).run(model, training_data=train_loader)
```

**Feature KD with auto adapters:**
```python
result = lob.Pipeline([
    lob.StructuredPrune(sparsity=0.3),
    lob.Distill(method="both", temperature=4.0, epochs=10),
]).run(model, training_data=train_loader)
```

**Explicit feature pairs + custom adapter:**
```python
result = lob.Pipeline([
    lob.StructuredPrune(sparsity=0.3),
    lob.Distill(
        method="feature",
        feature_layers={"student.block3": "teacher.block3"},
        adapter=MyCustomAdapter,
        epochs=20,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.01, "momentum": 0.9},
    ),
]).run(model, training_data=train_loader)
```

**Functional adapter (one-liner):**
```python
lob.Distill(
    method="feature",
    adapter=lambda s, t: F.mse_loss(s, t[:, :s.shape[1]]),
)
```

**Cross-model distillation (external teacher):**
```python
teacher = torchvision.models.resnet101(pretrained=True)
student = torchvision.models.resnet18(pretrained=True)

result = lob.Pipeline([
    lob.Distill(
        teacher=teacher,
        method="logit",
        temperature=6.0,
        alpha=0.7,
        task_loss_fn=nn.CrossEntropyLoss(),
        epochs=20,
    ),
]).run(student, training_data=train_loader, eval_fn=accuracy_fn)
```

### Recipe (YAML)

```yaml
name: prune-distill-quantize
description: "Structured prune → recover with KD → quantize"
stages:
  - type: structured_prune
    sparsity: 0.3
    criterion: l1
  - type: distill
    method: both
    temperature: 4.0
    alpha: 1.0
    epochs: 10
    lr: 1e-4
  - type: quantize
    method: dynamic
```

Note: recipe YAML can't specify `teacher` (it's a live object). Recipes always
use implicit teacher (original model). Explicit teachers require Python API.

---

## 11. Implementation Order

1. `registry.py` — stage + adapter registries, register existing stages
2. `hooks.py` — ForwardHookManager
3. `adapters.py` — FeatureAdapter, resolve_adapter, FunctionalAdapter
4. `distill_losses.py` — KDLoss, FeatureMatchingLoss, CombinedKDLoss
5. `distill.py` — Distill stage
6. PipelineContext + Pipeline.run() extension (training_data)
7. `__init__.py` exports
8. Tests
9. README update + example scripts

---

## 12. Test Plan

- **Unit:** KDLoss gradient flow, FeatureMatchingLoss with adapters, hook manager lifecycle
- **Integration:** Prune → Distill pipeline on a small MLP, verify loss decreases
- **Adapter:** auto-build with matched/mismatched dims, custom callable, registry lookup
- **Edge cases:** no training data (should raise), alpha=0 (pure task loss), feature auto-match with no common layers (should raise)
- **Recipe:** YAML round-trip for distill stage config

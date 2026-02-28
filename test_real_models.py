"""Real model validation tests for Lobotomizer v0.1."""
import sys
import time
import traceback
import io
import torch
import torch.nn as nn

import lobotomizer as lob

RESULTS = []

def get_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def test_forward(model, dummy_input):
    try:
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        return True, None
    except Exception as e:
        return False, str(e)

def run_test(model_name, model, dummy_input, pipeline_name, stages, input_shape):
    print(f"  Testing {pipeline_name}...", flush=True)
    record = {
        "model": model_name,
        "pipeline": pipeline_name,
        "orig_params": count_params(model),
        "orig_size_mb": round(get_size_mb(model), 2),
    }
    try:
        t0 = time.time()
        result = lob.compress(model, stages, device="cpu", input_shape=input_shape)
        elapsed = time.time() - t0
        compressed = result.model
        record["comp_params"] = result.profile_after["param_count"]
        record["comp_size_mb"] = result.profile_after["size_mb"]
        record["ratio"] = round(record["orig_size_mb"] / max(record["comp_size_mb"], 0.001), 2)
        record["time_s"] = round(elapsed, 1)
        record["summary"] = result.summary()
        
        # Test forward pass on compressed model
        ok, err = test_forward(compressed, dummy_input)
        record["forward_ok"] = ok
        record["forward_err"] = err
        record["error"] = None
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["comp_params"] = None
        record["comp_size_mb"] = None
        record["ratio"] = None
        record["forward_ok"] = False
        record["forward_err"] = None
        record["time_s"] = None
        record["summary"] = None
        traceback.print_exc()
    
    RESULTS.append(record)
    status = "✅" if record.get("forward_ok") else ("❌ ERROR" if record.get("error") else "⚠️ forward fail")
    print(f"    {status}", flush=True)

def make_pipelines():
    return [
        ("Prune 30% (unstructured L1)", [lob.Prune(method="l1_unstructured", sparsity=0.3)]),
        ("Quantize (dynamic int8)", [lob.Quantize(method="dynamic", dtype="qint8")]),
        ("Prune 30% + Quantize", [lob.Prune(method="l1_unstructured", sparsity=0.3), lob.Quantize(method="dynamic", dtype="qint8")]),
        ("Structured Prune 30%", [lob.StructuredPrune(sparsity=0.3, criterion="l1")]),
    ]

# ---- Models ----

def load_resnet50():
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    dummy = torch.randn(1, 3, 224, 224)
    return "ResNet-50", model, dummy, (1, 3, 224, 224)

def load_whisper_tiny():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    # Whisper needs special input
    dummy_features = torch.randn(1, 80, 3000)  # mel spectrogram
    decoder_input_ids = torch.tensor([[50258]])  # <|startoftranscript|>
    class WhisperWrapper(nn.Module):
        def __init__(self, whisper_model):
            super().__init__()
            self.model = whisper_model
        def forward(self, x):
            return self.model(input_features=x, decoder_input_ids=torch.tensor([[50258]]))
    wrapper = WhisperWrapper(model)
    return "Whisper-tiny", wrapper, dummy_features, (1, 80, 3000)

def load_mobilevit():
    import timm
    model = timm.create_model("mobilevitv2_050", pretrained=False)
    dummy = torch.randn(1, 3, 256, 256)
    return "MobileViT-v2-050", model, dummy, (1, 3, 256, 256)

# ---- Main ----

def main():
    loaders = [load_resnet50, load_whisper_tiny, load_mobilevit]
    
    for loader in loaders:
        print(f"\n{'='*60}")
        try:
            model_name, model, dummy, input_shape = loader()
            print(f"Loaded {model_name} ({count_params(model):,} params, {get_size_mb(model):.1f} MB)")
        except Exception as e:
            print(f"FAILED to load model from {loader.__name__}: {e}")
            traceback.print_exc()
            continue
        
        # Verify original forward pass
        ok, err = test_forward(model, dummy)
        if not ok:
            print(f"  Original model forward FAILED: {err}")
        
        for pipeline_name, stages in make_pipelines():
            run_test(model_name, model, dummy, pipeline_name, stages, input_shape)
    
    # Generate report
    write_report()

def write_report():
    lines = ["# Lobotomizer v0.1 — Real Model Test Report\n"]
    lines.append(f"**Date:** 2026-02-28\n")
    lines.append(f"**Models tested:** {len(set(r['model'] for r in RESULTS))}\n")
    lines.append(f"**Total tests:** {len(RESULTS)}\n")
    
    passed = sum(1 for r in RESULTS if r.get("forward_ok"))
    errored = sum(1 for r in RESULTS if r.get("error"))
    lines.append(f"**Passed:** {passed} | **Errors:** {errored} | **Forward failures:** {len(RESULTS) - passed - errored}\n")
    
    # Summary table
    lines.append("## Summary\n")
    lines.append("| Model | Pipeline | Orig Size (MB) | Comp Size (MB) | Ratio | Params Before | Params After | Forward OK | Time (s) |")
    lines.append("|-------|----------|---------------|---------------|-------|--------------|-------------|-----------|----------|")
    
    for r in RESULTS:
        if r.get("error"):
            lines.append(f"| {r['model']} | {r['pipeline']} | {r['orig_size_mb']} | ❌ ERROR | — | {r['orig_params']:,} | — | ❌ | — |")
        else:
            fwd = "✅" if r["forward_ok"] else "❌"
            lines.append(f"| {r['model']} | {r['pipeline']} | {r['orig_size_mb']} | {r['comp_size_mb']} | {r['ratio']}x | {r['orig_params']:,} | {r['comp_params']:,} | {fwd} | {r['time_s']} |")
    
    # Detailed results
    lines.append("\n## Detailed Results\n")
    for r in RESULTS:
        lines.append(f"### {r['model']} — {r['pipeline']}\n")
        if r.get("error"):
            lines.append(f"**ERROR:** `{r['error']}`\n")
        else:
            lines.append(f"```\n{r['summary']}\n```\n")
            if not r["forward_ok"]:
                lines.append(f"**Forward pass failed:** `{r['forward_err']}`\n")
    
    report = "\n".join(lines)
    with open("/home/node/.openclaw/workspace/lobotomizer/real_model_tests.md", "w") as f:
        f.write(report)
    print(f"\n{'='*60}")
    print("Report saved to real_model_tests.md")
    print(report)

if __name__ == "__main__":
    main()

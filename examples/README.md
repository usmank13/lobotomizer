# Lobotomizer Examples

Runnable compression examples. Each script is self-contained — just `python examples/<script>.py`.

All examples gracefully fall back to dummy models when optional dependencies (transformers, whisper, ultralytics, timm) aren't installed, so you only need `torch` + `lobotomizer`.

| Script | Description |
|---|---|
| `resnet50_edge.py` | ResNet50 pruned + quantized for edge deployment |
| `bert_quantize.py` | BERT-base quantized for faster CPU inference |
| `whisper_compress.py` | Whisper small compressed for on-device transcription |
| `yolo_edge.py` | YOLOv8n compressed for real-time edge inference (Jetson Nano, etc.) |
| `mobilevit_compress.py` | MobileViT further compressed — squeezing already-efficient models |

## Running

```bash
# From the repo root:
python examples/resnet50_edge.py
python examples/bert_quantize.py
python examples/whisper_compress.py
python examples/yolo_edge.py
python examples/mobilevit_compress.py
```

Compressed models are saved to `compressed_<name>/` in the current directory.

# Lobotomizer Examples

Runnable compression examples. Each script is self-contained — install the required deps, then `python examples/<script>.py`.

| Example | Model | Extra Deps | Use Case |
|---------|-------|-----------|----------|
| `resnet50_edge.py` | ResNet50 | None (torchvision) | Edge deployment |
| `bert_quantize.py` | BERT-base | `transformers` | Faster NLP inference |
| `whisper_compress.py` | Whisper small | `openai-whisper` | On-device transcription |
| `yolo_edge.py` | YOLOv8n | `ultralytics` | Real-time edge detection |
| `mobilevit_compress.py` | MobileViT-v2 | `timm` | Ultra-constrained devices |

## Install all example deps

```bash
pip install lobotomizer[test-all]
```

## Running

```bash
python examples/resnet50_edge.py
python examples/bert_quantize.py
python examples/whisper_compress.py
python examples/yolo_edge.py
python examples/mobilevit_compress.py
```

Compressed models are saved to `compressed_<name>/` in the current directory.

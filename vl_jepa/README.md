# VL-JEPA — Vision-Language Joint Embedding Predictive Architecture

A clean, self-contained PyTorch implementation of **VL-JEPA** (arXiv:2512.10942) — the non-generative vision-language model from Meta FAIR that predicts continuous embeddings in abstract representation space instead of generating tokens autoregressively.

---

## Architecture Overview

```
Video Frames ──► X-Encoder (ViT) ──► visual tokens
                                          │
                                          ▼
Text Query  ──► Y-Encoder (CLIP-text) ─► Predictor ──► predicted embedding
                                                               │
                                               cosine loss ◄──┤
                                                               │
Target Text ──► Y-Encoder (frozen) ──► target embedding ──────┘
                                                               │
                                         (optionally) Y-Decoder ──► text output
```

**Four core components:**

| Component | Role |
|---|---|
| **X-Encoder** | Encodes video frames → compact visual tokens (ViT backbone) |
| **Y-Encoder** | Encodes text → continuous semantic embeddings (shared space) |
| **Predictor** | Cross-attention transformer: visual tokens × text query → predicted embedding |
| **Y-Decoder** | Lightweight text decoder invoked *only when needed* (selective decoding) |

---

## Key Differences from Standard VLMs

| Property | Standard VLM (e.g. InstructBLIP) | VL-JEPA |
|---|---|---|
| Training objective | Next-token prediction (CE loss) | Embedding cosine/MSE loss |
| Output space | Token logits | Continuous embeddings |
| Decoder | Always active | Selective (invoked only on semantic change) |
| Parameters | Large (7B+) | Compact (~1.6B) |
| Retrieval | Requires special heads | Native (embedding similarity) |
| Classification | Requires special heads | Native (embedding nearest-neighbor) |

---

## Installation

```bash
pip install torch torchvision transformers einops tqdm
```

---

## File Structure

```
vl_jepa/
├── README.md               ← this file
├── model/
│   ├── __init__.py
│   ├── x_encoder.py        ← Video/image ViT encoder (X-encoder)
│   ├── y_encoder.py        ← Text encoder (Y-encoder, CLIP-text based)
│   ├── predictor.py        ← Cross-attention predictor
│   ├── y_decoder.py        ← Lightweight text decoder
│   └── vl_jepa.py          ← Full model combining all 4 components
├── training/
│   ├── __init__.py
│   ├── loss.py             ← Embedding prediction loss (cosine + L2 reg)
│   ├── dataset.py          ← Video-text dataset (works with Action100M FiftyOne dataset)
│   └── train.py            ← Training loop
├── inference/
│   ├── __init__.py
│   ├── classify.py         ← Zero-shot open-vocabulary classification
│   ├── retrieve.py         ← Text-to-video retrieval
│   ├── vqa.py              ← Discriminative VQA (multiple choice)
│   └── selective_decode.py ← Selective decoding for streaming video
└── demo.py                 ← End-to-end demo (no weights needed, random init)
```

---

## Quick Demo (Random Weights — No Pretrained Checkpoint Needed)

```bash
cd fiftyone_video_workshop-main
python vl_jepa/demo.py
```

---

## Training on Action100M

```bash
python vl_jepa/training/train.py \
    --dataset_name action100m \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4
```

---

## Citation

```bibtex
@article{vljepa2025,
  title={VL-JEPA: Joint Embedding Predictive Architecture for Vision-language},
  author={Meta FAIR et al.},
  journal={arXiv preprint arXiv:2512.10942},
  year={2025}
}
```

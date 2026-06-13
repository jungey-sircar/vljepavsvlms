"""
VL-JEPA Model Package

A PyTorch implementation of VL-JEPA:
  "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
  arXiv:2512.10942 — Meta FAIR, 2025

Components:
  - XEncoder:   Video/image encoder (ViT backbone)
  - YEncoder:   Text encoder (CLIP-text backbone, shared embedding space)
  - Predictor:  Cross-attention transformer mapping (visual, text query) → target embedding
  - YDecoder:   Lightweight auto-regressive text decoder (used selectively)
  - VLJepa:     Full model integrating all four components
"""

from .vl_jepa import VLJepa
from .x_encoder import XEncoder
from .y_encoder import YEncoder
from .predictor import Predictor
from .y_decoder import YDecoder

__all__ = ["VLJepa", "XEncoder", "YEncoder", "Predictor", "YDecoder"]

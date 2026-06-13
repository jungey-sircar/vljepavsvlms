"""
VL-JEPA — Top-level package init
"""
from .model import VLJepa, XEncoder, YEncoder, Predictor, YDecoder

__all__ = ["VLJepa", "XEncoder", "YEncoder", "Predictor", "YDecoder"]

__version__ = "0.1.0"
__paper__ = "arXiv:2512.10942"
__description__ = (
    "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language. "
    "Non-generative VLM that predicts continuous text embeddings from video."
)

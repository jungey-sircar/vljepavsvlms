"""
Inference — Package Init
"""
from .classify import zero_shot_classify
from .retrieve import build_video_index, text_to_video_retrieval
from .vqa import discriminative_vqa
from .selective_decode import SelectiveVideoDescriber

__all__ = [
    "zero_shot_classify",
    "build_video_index",
    "text_to_video_retrieval",
    "discriminative_vqa",
    "SelectiveVideoDescriber",
]

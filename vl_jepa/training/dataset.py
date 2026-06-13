"""
Dataset — Video-Text Dataset for VL-JEPA Training
===================================================
Two dataset classes are provided:

  1. VideoTextDataset   — Generic dataset: loads (video_path, question, answer) triplets
  2. Action100MDataset  — Reads directly from a FiftyOne `action100m` dataset on disk,
                          turning the GPT-refined temporal annotations into
                          (video_clip, action_brief, gpt_summary_brief) training pairs

VL-JEPA training sample format:
    {
        "video":              [C, T, H, W]  float32, normalized
        "query_text":         str           (the question / prompt)
        "target_text":        str           (the expected answer / caption)
        "query_input_ids":    [Lq]          int64 tokenized query
        "target_input_ids":   [Lt]          int64 tokenized target
    }
"""

import os
import json
import random
import torch
import torchvision
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Tuple, Callable

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ---------------------------------------------------------------------------
# Default video transforms
# ---------------------------------------------------------------------------

def default_video_transform(
    img_size: int = 224,
    num_frames: int = 8,
    mean: Tuple = (0.485, 0.456, 0.406),
    std:  Tuple = (0.229, 0.224, 0.225),
) -> Callable:
    """
    Returns a transform function that:
      1. Temporally subsamples to num_frames evenly
      2. Resizes spatial dims to img_size × img_size
      3. Normalizes to ImageNet stats
    """
    normalize = T.Normalize(mean=mean, std=std) if HAS_TORCHVISION else None

    def transform(frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [T, H, W, C] uint8 tensor (from torchvision.io.read_video)
        Returns:
            video: [C, num_frames, img_size, img_size] float32
        """
        T_orig = frames.shape[0]

        # Temporal subsampling: pick num_frames evenly
        if T_orig >= num_frames:
            indices = torch.linspace(0, T_orig - 1, num_frames).long()
        else:
            # Repeat last frame if video is shorter than num_frames
            indices = torch.cat([
                torch.arange(T_orig),
                torch.full((num_frames - T_orig,), T_orig - 1),
            ])
        frames = frames[indices]   # [num_frames, H, W, C]

        # [T, H, W, C] → [T, C, H, W]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0

        # Spatial resize
        frames = torch.stack([
            TF.resize(f, [img_size, img_size], antialias=True)
            for f in frames
        ])  # [T, C, H, W]

        # Normalize
        if normalize is not None:
            frames = torch.stack([normalize(f) for f in frames])

        # [T, C, H, W] → [C, T, H, W]
        return frames.permute(1, 0, 2, 3)

    return transform


# ---------------------------------------------------------------------------
# Simple byte-level tokenizer (standalone mode)
# ---------------------------------------------------------------------------

def byte_tokenize(text: str, max_len: int = 77) -> torch.Tensor:
    """
    Encode text as UTF-8 bytes, truncate to max_len - 1, append EOS (1).
    Returns [max_len] int64 tensor.
    """
    ids = list(text.encode("utf-8")[: max_len - 1]) + [1]   # 1 = EOS
    ids = ids + [0] * (max_len - len(ids))                   # 0 = PAD
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Generic VideoTextDataset
# ---------------------------------------------------------------------------

class VideoTextDataset(Dataset):
    """
    Generic dataset for VL-JEPA training.

    Expects a JSON lines file where each line is:
        {
            "video_path": "/abs/path/to/video.mp4",
            "query":      "What action is shown?",
            "target":     "A person stirs the pot with a wooden spoon."
        }

    Args:
        jsonl_path:     Path to the .jsonl annotation file
        video_dir:      Optional base directory (prepended if paths are relative)
        transform:      Video transform callable
        num_frames:     Number of frames to sample
        img_size:       Spatial resolution
        max_text_len:   Max tokenized text length
        tokenize_fn:    Custom tokenizer function (text → [L] int64 tensor)
                        Defaults to byte_tokenize for standalone mode
    """

    def __init__(
        self,
        jsonl_path: str,
        video_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        num_frames: int = 8,
        img_size: int = 224,
        max_text_len: int = 77,
        tokenize_fn: Optional[Callable] = None,
    ):
        self.video_dir = Path(video_dir) if video_dir else None
        self.num_frames = num_frames
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.tokenize_fn = tokenize_fn or (lambda t: byte_tokenize(t, max_text_len))
        self.transform = transform or default_video_transform(img_size, num_frames)

        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"VideoTextDataset: loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, path: str) -> torch.Tensor:
        """Load video, return [C, T, H, W] float32."""
        if self.video_dir is not None and not os.path.isabs(path):
            path = str(self.video_dir / path)
        try:
            frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
            return self.transform(frames)
        except Exception as e:
            # Return black frames on error (graceful degradation)
            print(f"Warning: failed to load {path}: {e}")
            return torch.zeros(3, self.num_frames, self.img_size, self.img_size)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        video = self._load_video(item["video_path"])
        query_text = item.get("query", "Describe this video.")
        target_text = item.get("target", "")

        return {
            "video": video,
            "query_text": query_text,
            "target_text": target_text,
            "query_input_ids": self.tokenize_fn(query_text),
            "target_input_ids": self.tokenize_fn(target_text),
        }


# ---------------------------------------------------------------------------
# Action100M FiftyOne Dataset
# ---------------------------------------------------------------------------

class Action100MDataset(Dataset):
    """
    Reads the `action100m` FiftyOne dataset from disk and yields
    (video_clip, query, target) triplets for VL-JEPA training.

    Requires:
        pip install fiftyone

    Each FiftyOne sample becomes multiple training pairs — one per
    temporal segment annotation. Annotation fields used:
        - gpt_action_brief    → target (action label)
        - gpt_summary_brief   → alternative target (clip caption)
        - gpt_action_detailed → alternative target (instruction style)

    For each segment, the query is randomly drawn from a set of
    action-understanding prompts (data augmentation).

    Args:
        dataset_name:   FiftyOne dataset name (default "action100m")
        annotation_field: Which GPT annotation field to use as target
                          Options: "gpt_action_brief", "gpt_summary_brief",
                                   "gpt_action_detailed"
        max_samples:    Limit number of FiftyOne samples (videos)
        num_frames:     Frames to sample per clip
        img_size:       Spatial resolution
        max_text_len:   Max tokenized text length
        transform:      Custom video transform
    """

    QUERY_TEMPLATES = [
        "What action is happening in this video?",
        "Describe the activity shown in this clip.",
        "What is the person doing?",
        "Identify the action performed in this video segment.",
        "What task is being demonstrated here?",
    ]

    def __init__(
        self,
        dataset_name: str = "action100m",
        annotation_field: str = "gpt_action_brief",
        max_samples: Optional[int] = None,
        num_frames: int = 8,
        img_size: int = 224,
        max_text_len: int = 77,
        transform: Optional[Callable] = None,
    ):
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError("fiftyone is required. Install with: pip install fiftyone")

        self.annotation_field = annotation_field
        self.num_frames = num_frames
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.transform = transform or default_video_transform(img_size, num_frames)
        self.tokenize_fn = lambda t: byte_tokenize(t, max_text_len)

        # Load FiftyOne dataset
        print(f"Loading FiftyOne dataset '{dataset_name}'...")
        fo_dataset = fo.load_dataset(dataset_name)
        if max_samples:
            fo_dataset = fo_dataset.take(max_samples)

        # Flatten into (video_path, start_sec, end_sec, label) tuples
        self.samples = []
        for sample in fo_dataset.iter_samples():
            video_path = sample.filepath
            fps = sample.metadata.frame_rate if sample.metadata else 30.0

            field_data = getattr(sample, annotation_field, None)
            if field_data is None or not hasattr(field_data, "detections"):
                continue

            for det in field_data.detections:
                if not det.label:
                    continue
                # Convert frame support → seconds
                start_frame, end_frame = det.support
                start_sec = start_frame / fps
                end_sec = end_frame / fps
                duration = end_sec - start_sec

                # Skip very short segments
                if duration < 1.0:
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "label": det.label,
                })

        print(f"Action100MDataset: {len(self.samples)} segment-level training pairs "
              f"from {len(fo_dataset)} videos (field: {annotation_field})")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip(self, video_path: str, start_sec: float, end_sec: float) -> torch.Tensor:
        """Load a temporal clip from a video file."""
        try:
            frames, _, _ = torchvision.io.read_video(
                video_path,
                start_pts=start_sec,
                end_pts=end_sec,
                pts_unit="sec",
                output_format="THWC",
            )
            if frames.shape[0] == 0:
                raise ValueError("Empty clip")
            return self.transform(frames)
        except Exception as e:
            print(f"Warning: failed to load clip {video_path}@{start_sec:.1f}s: {e}")
            return torch.zeros(3, self.num_frames, self.img_size, self.img_size)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        video = self._load_clip(item["video_path"], item["start_sec"], item["end_sec"])

        query_text = random.choice(self.QUERY_TEMPLATES)
        target_text = item["label"]

        return {
            "video": video,
            "query_text": query_text,
            "target_text": target_text,
            "query_input_ids": self.tokenize_fn(query_text),
            "target_input_ids": self.tokenize_fn(target_text),
        }

    @staticmethod
    def export_to_jsonl(dataset_name: str, output_path: str, annotation_field: str = "gpt_action_brief"):
        """
        Export Action100M training pairs to a JSONL file for use with VideoTextDataset.
        Useful for training without FiftyOne as a runtime dependency.
        """
        import fiftyone as fo
        fo_dataset = fo.load_dataset(dataset_name)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in fo_dataset.iter_samples():
                fps = sample.metadata.frame_rate if sample.metadata else 30.0
                field_data = getattr(sample, annotation_field, None)
                if field_data is None:
                    continue
                for det in field_data.detections:
                    if not det.label:
                        continue
                    start_sec = det.support[0] / fps
                    end_sec = det.support[1] / fps
                    if (end_sec - start_sec) < 1.0:
                        continue
                    record = {
                        "video_path": sample.filepath,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "query": "What action is happening in this video?",
                        "target": det.label,
                    }
                    f.write(json.dumps(record) + "\n")
        print(f"Exported to {output_path}")

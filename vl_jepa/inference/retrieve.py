"""
Text-to-Video Retrieval
========================
VL-JEPA natively supports text-to-video retrieval by operating entirely
in the shared embedding space — no task-specific heads required.

Retrieval pipeline:
  1. Offline:   Encode all videos in the database → video embeddings [N, D]
  2. Online:    Encode query text → text embedding [D]
  3. Rank:      Cosine similarity between query and all video embeddings
  4. Return:    Top-k ranked video indices

This file provides:
  - build_video_index():         Encode all videos in a directory / dataset
  - text_to_video_retrieval():   Query the index with natural language
  - FiftyOneRetriever:           Wrapper that integrates with a FiftyOne dataset
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Union, Tuple
import json


# ---------------------------------------------------------------------------
# Helper: load and transform a single video
# ---------------------------------------------------------------------------

def _load_video_tensor(
    path: str,
    num_frames: int = 8,
    img_size: int = 224,
) -> torch.Tensor:
    """Load a video file and return [C, T, H, W] float32."""
    try:
        import torchvision
        import torchvision.transforms.functional as TF

        frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
        T_orig = frames.shape[0]

        if T_orig == 0:
            return torch.zeros(3, num_frames, img_size, img_size)

        # Temporal subsample
        if T_orig >= num_frames:
            idx = torch.linspace(0, T_orig - 1, num_frames).long()
        else:
            idx = torch.cat([torch.arange(T_orig),
                             torch.full((num_frames - T_orig,), T_orig - 1)])
        frames = frames[idx]  # [T, H, W, C]

        # Normalize + resize
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        frames = torch.stack([TF.resize(f, [img_size, img_size], antialias=True) for f in frames])

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames.permute(1, 0, 2, 3)   # [C, T, H, W]
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return torch.zeros(3, num_frames, img_size, img_size)


# ---------------------------------------------------------------------------
# Build video embedding index
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_video_index(
    model,                             # VLJepa
    video_paths: List[str],
    batch_size: int = 4,
    num_frames: int = 8,
    img_size: int = 224,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Encode all videos and build a searchable embedding index.

    Args:
        model:       Trained VLJepa model (used for X-Encoder CLS embedding)
        video_paths: List of absolute paths to video files
        batch_size:  Number of videos to process at once
        num_frames:  Frames to sample per video
        img_size:    Spatial resolution
        device:      Computation device
        save_path:   If set, save the index to disk as a .pt file

    Returns:
        (embeddings, paths)
        embeddings: [N, D] float32 tensor of L2-normalized video embeddings
        paths:      List[str] of video paths (same order as embeddings)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    embeddings = []
    valid_paths = []

    print(f"Building video index for {len(video_paths)} videos...")
    for i in range(0, len(video_paths), batch_size):
        batch_paths = video_paths[i : i + batch_size]
        videos = torch.stack([
            _load_video_tensor(p, num_frames, img_size) for p in batch_paths
        ]).to(device)  # [B, C, T, H, W]

        embs = model.encode_video(videos)    # [B, D]
        embeddings.append(embs.cpu())
        valid_paths.extend(batch_paths)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Encoded {i + len(batch_paths)}/{len(video_paths)} videos")

    embeddings = torch.cat(embeddings, dim=0)   # [N, D]
    print(f"Video index built: {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    if save_path:
        torch.save({"embeddings": embeddings, "paths": valid_paths}, save_path)
        print(f"Index saved to {save_path}")

    return embeddings, valid_paths


def load_video_index(path: str) -> Tuple[torch.Tensor, List[str]]:
    """Load a previously saved video index."""
    data = torch.load(path, map_location="cpu")
    return data["embeddings"], data["paths"]


# ---------------------------------------------------------------------------
# Text-to-Video Retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def text_to_video_retrieval(
    model,                             # VLJepa
    query_texts: Union[str, List[str]],
    video_embeddings: torch.Tensor,    # [N, D] pre-computed video embeddings
    video_paths: List[str],
    top_k: int = 10,
    device: Optional[torch.device] = None,
) -> List[dict]:
    """
    Retrieve top-k videos for each query text.

    Args:
        model:             Trained VLJepa model (used for Y-Encoder)
        query_texts:       One or more natural language queries
        video_embeddings:  [N, D] L2-normalized video embeddings (from build_video_index)
        video_paths:       Corresponding video file paths
        top_k:             Number of results to return per query
        device:            Computation device

    Returns:
        List of result dicts (one per query):
        {
            "query":      str,
            "top_videos": List[str],    top-k video paths
            "scores":     List[float],  cosine similarity scores
            "ranks":      List[int],    original indices in video_embeddings
        }
    """
    if isinstance(query_texts, str):
        query_texts = [query_texts]

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Encode all query texts
    text_embeddings = model.encode_text(query_texts, device=device)   # [Q, D]

    # Move video embeddings to device for batched similarity
    vid_embs = video_embeddings.to(device)   # [N, D]

    # Cosine similarity: [Q, N]
    sim_matrix = text_embeddings @ vid_embs.T

    results = []
    for q_idx, query in enumerate(query_texts):
        sims = sim_matrix[q_idx]                          # [N]
        top_scores, top_indices = sims.topk(min(top_k, len(video_paths)))

        results.append({
            "query": query,
            "top_videos": [video_paths[i] for i in top_indices.tolist()],
            "scores": top_scores.tolist(),
            "ranks": top_indices.tolist(),
        })

    return results


# ---------------------------------------------------------------------------
# FiftyOne-integrated Retriever
# ---------------------------------------------------------------------------

class FiftyOneRetriever:
    """
    High-level retriever that works directly with a FiftyOne `action100m` dataset.

    Builds an embedding index from all videos in the dataset and supports
    natural-language retrieval queries.

    Usage:
        retriever = FiftyOneRetriever(model, dataset_name="action100m")
        retriever.build_index()
        results = retriever.search("a person stirs soup in a pot", top_k=5)
        retriever.visualize(results)   # opens FiftyOne App with results
    """

    def __init__(
        self,
        model,
        dataset_name: str = "action100m",
        num_frames: int = 8,
        img_size: int = 224,
        device: Optional[torch.device] = None,
        index_cache_path: Optional[str] = None,
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.img_size = img_size
        self.device = device or next(model.parameters()).device
        self.index_cache_path = index_cache_path

        self.embeddings: Optional[torch.Tensor] = None
        self.video_paths: Optional[List[str]] = None
        self._fo_dataset = None

    def _get_fo_dataset(self):
        if self._fo_dataset is None:
            import fiftyone as fo
            self._fo_dataset = fo.load_dataset(self.dataset_name)
        return self._fo_dataset

    def build_index(self, force_rebuild: bool = False):
        """Build (or load cached) video embedding index."""
        if (not force_rebuild and self.index_cache_path
                and Path(self.index_cache_path).exists()):
            print(f"Loading cached index from {self.index_cache_path}")
            self.embeddings, self.video_paths = load_video_index(self.index_cache_path)
            return

        dataset = self._get_fo_dataset()
        paths = [s.filepath for s in dataset.iter_samples()]

        self.embeddings, self.video_paths = build_video_index(
            model=self.model,
            video_paths=paths,
            num_frames=self.num_frames,
            img_size=self.img_size,
            device=self.device,
            save_path=self.index_cache_path,
        )

    def search(self, query: str, top_k: int = 10) -> dict:
        """Search the index with a natural language query."""
        assert self.embeddings is not None, "Call build_index() first."
        results = text_to_video_retrieval(
            model=self.model,
            query_texts=[query],
            video_embeddings=self.embeddings,
            video_paths=self.video_paths,
            top_k=top_k,
            device=self.device,
        )
        return results[0]

    def visualize(self, result: dict):
        """Open FiftyOne App showing retrieved videos."""
        import fiftyone as fo
        dataset = self._get_fo_dataset()
        retrieved_paths = set(result["top_videos"])
        view = dataset.match(fo.ViewField("filepath").is_in(list(retrieved_paths)))
        session = fo.launch_app(view)
        return session

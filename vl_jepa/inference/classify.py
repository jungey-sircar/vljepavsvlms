"""
Zero-Shot Open-Vocabulary Classification
==========================================
VL-JEPA natively supports open-vocabulary classification without any
task-specific classification heads. The model:

  1. Predicts an embedding conditioned on the visual context + query text
  2. Encodes all candidate class names via the Y-Encoder
  3. Picks the class with highest cosine similarity to the predicted embedding

This is analogous to CLIP's zero-shot classification but operates through
the Predictor rather than directly comparing visual and text encodings.

Key advantage: the Predictor can leverage video-specific temporal context
that a CLIP visual encoder might not capture.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from vl_jepa.model import VLJepa


# Default query templates for action recognition
ACTION_QUERY_TEMPLATES = [
    "What action is happening in this video?",
    "What is the person doing in this video?",
    "Identify the activity in this clip.",
]

SCENE_QUERY_TEMPLATES = [
    "What scene is shown in this video?",
    "Describe the setting of this video.",
    "What location is shown in this clip?",
]


@torch.no_grad()
def zero_shot_classify(
    model: VLJepa,
    video: torch.Tensor,                     # [B, C, T, H, W] or [C, T, H, W]
    class_names: List[str],
    query_template: str = "What action is happening in this video?",
    prompt_ensemble: bool = True,
    prompt_templates: Optional[List[str]] = None,
    top_k: int = 5,
    device: Optional[torch.device] = None,
) -> List[dict]:
    """
    Zero-shot open-vocabulary classification using VL-JEPA.

    Args:
        model:            Trained VLJepa model
        video:            Input video tensor [B, C, T, H, W] or [C, T, H, W]
        class_names:      List of candidate class name strings
        query_template:   Query text to condition the predictor
        prompt_ensemble:  If True, average embeddings over multiple query templates
                          for more robust predictions (recommended)
        prompt_templates: Custom query templates (overrides default ensemble)
        top_k:            Return top-k predictions per video
        device:           Computation device (inferred from model if None)

    Returns:
        List of dicts (one per batch element):
        {
            "top_classes": List[str],     top-k class names
            "top_scores":  List[float],   cosine similarity scores
            "all_scores":  tensor [C],    scores for all classes
        }
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Handle single video input
    if video.dim() == 4:
        video = video.unsqueeze(0)
    video = video.to(device)
    B = video.shape[0]

    # Encode class names (do once, reused across batch)
    class_embeddings = model.encode_text(class_names, device=device)   # [C, D]

    # Build query embedding(s)
    if prompt_ensemble:
        templates = prompt_templates or ACTION_QUERY_TEMPLATES
        query_embs = [model.encode_text([t], device=device) for t in templates]
        query_embedding = torch.stack(query_embs, dim=0).mean(dim=0)   # [1, D]
        query_embedding = F.normalize(query_embedding, dim=-1)
    else:
        query_embedding = model.encode_text([query_template], device=device)  # [1, D]

    # Expand query for batch
    query_embedding = query_embedding.expand(B, -1)  # [B, D]

    # Get visual tokens from X-Encoder
    visual_tokens = model.x_encoder(video)           # [B, N, Dv]

    # Predict target embedding via Predictor
    predicted = model.predictor(visual_tokens, query_embedding)   # [B, D]

    # Compute similarity to all class embeddings
    scores = predicted @ class_embeddings.T          # [B, C]

    # Format results
    results = []
    for b in range(B):
        s = scores[b]
        top_scores, top_indices = s.topk(min(top_k, len(class_names)))
        results.append({
            "top_classes": [class_names[i] for i in top_indices.tolist()],
            "top_scores": top_scores.tolist(),
            "all_scores": s.cpu(),
        })

    return results


@torch.no_grad()
def evaluate_classification(
    model: VLJepa,
    loader,                                  # DataLoader yielding (video, label_idx)
    class_names: List[str],
    query_template: str = "What action is happening in this video?",
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate top-1 and top-5 accuracy on a classification dataset.

    Args:
        model:          Trained VLJepa model
        loader:         DataLoader yielding dicts with 'video' [B,C,T,H,W]
                        and 'label_idx' [B] (integer class indices)
        class_names:    Ordered list of class names (index matches label_idx)
        device:         Computation device

    Returns:
        {"top1_acc": float, "top5_acc": float, "n_samples": int}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    top1_correct = 0
    top5_correct = 0
    n_total = 0

    # Pre-encode class names
    class_embs = model.encode_text(class_names, device=device)  # [C, D]

    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["label_idx"].to(device)    # [B]
        B = video.shape[0]

        query_emb = model.encode_text([query_template], device=device).expand(B, -1)
        vis_tokens = model.x_encoder(video)
        pred = model.predictor(vis_tokens, query_emb)       # [B, D]

        scores = pred @ class_embs.T                        # [B, C]

        # Top-1
        top1_correct += (scores.argmax(dim=-1) == labels).sum().item()

        # Top-5
        _, top5_indices = scores.topk(min(5, len(class_names)), dim=-1)
        for i in range(B):
            if labels[i].item() in top5_indices[i].tolist():
                top5_correct += 1

        n_total += B

    return {
        "top1_acc": top1_correct / max(n_total, 1),
        "top5_acc": top5_correct / max(n_total, 1),
        "n_samples": n_total,
    }

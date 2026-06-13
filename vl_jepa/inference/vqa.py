"""
Discriminative VQA (Visual Question Answering)
================================================
VL-JEPA supports multiple-choice VQA natively without task-specific heads.

For each question, the model:
  1. Predicts an embedding conditioned on (visual tokens, question embedding)
  2. Encodes each answer candidate via the Y-Encoder
  3. Selects the answer whose embedding is closest to the prediction

This discriminative approach is used on benchmarks like:
  - GQA           (visual grounding QA)
  - POPE / POPEv2 (object hallucination benchmark)
  - TallyQA       (counting questions)

Unlike generative VQA, no beam search or autoregressive decoding is needed —
inference is a single forward pass + argmax over answer embeddings.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict


@torch.no_grad()
def discriminative_vqa(
    model,                                        # VLJepa
    video: torch.Tensor,                          # [B, C, T, H, W]
    questions: List[str],                         # length B
    answer_choices: List[List[str]],              # B × num_choices
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Discriminative (multiple-choice) VQA using VL-JEPA.

    The model predicts an embedding conditioned on the question and picks
    the answer candidate whose embedding is closest.

    Args:
        model:          Trained VLJepa model
        video:          [B, C, T, H, W] video tensor
        questions:      List of question strings (one per batch element)
        answer_choices: List of answer candidate lists (one list per batch element)
        device:         Computation device

    Returns:
        List of dicts (one per batch element):
        {
            "question":       str,
            "predicted_idx":  int,            index of predicted answer
            "predicted_ans":  str,            predicted answer string
            "scores":         List[float],    similarity scores for each candidate
            "all_answers":    List[str],      the answer choices
        }
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    B = video.shape[0]
    video = video.to(device)

    assert len(questions) == B == len(answer_choices), \
        "questions and answer_choices must have length equal to batch size"

    # Encode visual context (same for all questions in batch)
    vis_tokens = model.x_encoder(video)   # [B, N, Dv]

    results = []
    for i in range(B):
        # Encode question
        q_emb = model.encode_text([questions[i]], device=device)   # [1, D]

        # Predict embedding
        pred = model.predictor(
            vis_tokens[i:i+1],   # [1, N, Dv]
            q_emb,               # [1, D]
        )                        # [1, D]

        # Encode all answer candidates
        ans_embs = model.encode_text(answer_choices[i], device=device)  # [A, D]

        # Cosine similarity
        scores = (pred @ ans_embs.T).squeeze(0)   # [A]
        pred_idx = scores.argmax().item()

        results.append({
            "question": questions[i],
            "predicted_idx": pred_idx,
            "predicted_ans": answer_choices[i][pred_idx],
            "scores": scores.tolist(),
            "all_answers": answer_choices[i],
        })

    return results


@torch.no_grad()
def binary_vqa(
    model,
    video: torch.Tensor,              # [B, C, T, H, W]
    questions: List[str],             # length B
    yes_text: str = "Yes",
    no_text: str = "No",
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Binary (Yes/No) VQA — special case of discriminative VQA.
    Used for POPE benchmark (object hallucination).

    Returns:
        List of dicts with 'answer' (str), 'confidence' (float), 'scores' (dict)
    """
    answer_choices = [[yes_text, no_text]] * len(questions)
    results = discriminative_vqa(model, video, questions, answer_choices, device)

    for r in results:
        yes_score, no_score = r["scores"]
        confidence = abs(yes_score - no_score)
        r["answer"] = r["predicted_ans"]
        r["confidence"] = confidence
        r["scores"] = {"yes": yes_score, "no": no_score}

    return results


@torch.no_grad()
def counting_vqa(
    model,
    video: torch.Tensor,              # [B, C, T, H, W]
    questions: List[str],             # length B
    max_count: int = 10,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Counting VQA — answers are numbers 0..max_count.
    Used for TallyQA benchmark.

    Returns:
        List of dicts with 'count' (int), 'confidence' (float)
    """
    count_choices = [[str(n) for n in range(max_count + 1)]] * len(questions)
    results = discriminative_vqa(model, video, questions, count_choices, device)

    for r in results:
        r["count"] = int(r["predicted_ans"])
    return results


@torch.no_grad()
def evaluate_vqa(
    model,
    loader,                          # DataLoader yielding dicts with video, question, answer_idx, answer_choices
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Evaluate discriminative VQA accuracy on a dataset.

    Loader must yield:
        batch["video"]          [B, C, T, H, W]
        batch["question"]       List[str]
        batch["answer_idx"]     [B] int64 — index of correct answer in choices
        batch["answer_choices"] List[List[str]] — per-sample answer candidates

    Returns:
        {"accuracy": float, "n_samples": int}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    correct = 0
    n_total = 0

    for batch in loader:
        video = batch["video"].to(device)
        questions = batch["question"]
        answer_choices = batch["answer_choices"]
        answer_idx = batch["answer_idx"]

        results = discriminative_vqa(model, video, questions, answer_choices, device)

        for i, r in enumerate(results):
            if r["predicted_idx"] == answer_idx[i].item():
                correct += 1
        n_total += len(results)

    return {
        "accuracy": correct / max(n_total, 1),
        "n_samples": n_total,
    }

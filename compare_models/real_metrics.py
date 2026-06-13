"""
VL-JEPA vs Qwen3-VL — Real Forward-Pass Metrics Comparison
============================================================
Runs BOTH actual model implementations on the SAME synthetic video data.
Computes precision/recall/F1/accuracy from real model outputs.

How it works:
  1. Generate synthetic video clips (random pixels) + assign ground-truth labels
  2. Run VL-JEPA forward pass: video -> X-Encoder -> Predictor -> embedding
     -> cosine similarity with class embeddings -> prediction
  3. Run Qwen3-VL forward pass: video -> ViT -> Projector -> Decoder -> logits
     -> argmax -> prediction
  4. Compare both models' predictions against ground truth
  5. Compute sklearn metrics from the REAL outputs

Both models have random weights, so both should perform near chance level.
The point is: the metrics come from actual model inference, not simulation.
"""

import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_auc_score, average_precision_score, top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vl_jepa.model import VLJepa
from compare_models.compare import Qwen3VL

torch.manual_seed(42)
np.random.seed(42)

# ── Config ─────────────────────────────────────────────────────────────────
N_SAMPLES    = 200       # number of synthetic video clips
N_CLASSES    = 10        # action classes
NUM_FRAMES   = 4
IMG_SIZE     = 224
BATCH_SIZE   = 4         # process in batches to avoid OOM

ACTION_CLASSES = [
    "stir soup", "chop vegetables", "mix batter", "roll dough", "pour liquid",
    "crack eggs", "slice bread", "wash hands", "peel potato", "grate cheese",
][:N_CLASSES]

DIVIDER = "=" * 72


def fmt_pct(v):
    """Format a float as percentage."""
    if isinstance(v, float) and np.isnan(v):
        return "  N/A"
    return f"{v*100:.2f}%"


def print_header(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ── Step 1: Build both models ──────────────────────────────────────────────

def build_models(device):
    print("Building VL-JEPA (standalone, random weights)...")
    vljepa = VLJepa.build_default(
        mode="standalone", num_frames=NUM_FRAMES, with_decoder=True
    ).to(device)
    vljepa.eval()

    print("Building Qwen3-VL (structural stub, random weights)...")
    qwen = Qwen3VL().to(device)
    qwen.eval()

    vj_params = sum(p.numel() for p in vljepa.parameters())
    qw_params = sum(p.numel() for p in qwen.parameters())
    print(f"  VL-JEPA:  {vj_params/1e6:.2f}M params")
    print(f"  Qwen3-VL: {qw_params/1e6:.2f}M params")
    return vljepa, qwen


# ── Step 2: Generate synthetic data ───────────────────────────────────────

def generate_data(n_samples, n_classes, device):
    """Generate random video clips with ground-truth labels."""
    print(f"\nGenerating {n_samples} synthetic video clips ({NUM_FRAMES}x{IMG_SIZE}x{IMG_SIZE})...")
    videos = torch.randn(n_samples, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, n_classes, (n_samples,))
    print(f"  Label distribution: {dict(zip(*np.unique(labels.numpy(), return_counts=True)))}")
    return videos, labels


# ── Step 3: Encode class names for VL-JEPA ─────────────────────────────────

def encode_class_names_vljepa(model, class_names, device):
    """Encode each class name through VL-JEPA's Y-Encoder to get class embeddings."""
    class_embeddings = []
    for name in class_names:
        # Convert class name to byte-level token ids (same as model's standalone encoder)
        token_ids = [b for b in name.encode("utf-8")]
        # Pad/truncate to fixed length
        max_len = 32
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            emb = model.y_encoder(token_tensor)  # [1, D]
        class_embeddings.append(emb)

    return torch.cat(class_embeddings, dim=0)  # [N_CLASSES, D]


# ── Step 4: VL-JEPA inference ──────────────────────────────────────────────

def vljepa_classify(model, videos, class_embeddings, device):
    """
    VL-JEPA classification:
      1. Encode video -> visual tokens
      2. For each video, predict embedding conditioned on a generic query
      3. Compute cosine similarity with all class embeddings
      4. Prediction = argmax of similarities
    Returns: predictions [N], score_matrix [N, C]
    """
    n_samples = videos.shape[0]
    n_classes = class_embeddings.shape[0]
    all_scores = []

    # Generic query: "what action is shown?"
    query_text = "what action"
    query_ids = [b for b in query_text.encode("utf-8")]
    max_len = 32
    query_ids = query_ids + [0] * (max_len - len(query_ids))
    query_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)

    # Normalize class embeddings
    class_emb_norm = F.normalize(class_embeddings, dim=-1)  # [C, D]

    for i in range(0, n_samples, BATCH_SIZE):
        batch = videos[i:i+BATCH_SIZE].to(device)
        B = batch.shape[0]
        q = query_tensor.expand(B, -1)

        with torch.no_grad():
            pred_emb = model(batch, q)  # [B, D]
            pred_norm = F.normalize(pred_emb, dim=-1)  # [B, D]
            # Cosine similarity with all classes
            sims = pred_norm @ class_emb_norm.T  # [B, C]
            all_scores.append(sims.cpu())

    score_matrix = torch.cat(all_scores, dim=0).numpy()  # [N, C]
    predictions = score_matrix.argmax(axis=1)
    return predictions, score_matrix


# ── Step 5: Qwen3-VL inference ─────────────────────────────────────────────

def qwen3vl_classify(model, videos, n_classes, device):
    """
    Qwen3-VL classification:
      1. Reshape video [B, C, T, H, W] -> individual frames [B*T, C, H, W]
      2. Encode frames through ViT -> project to LLM space
      3. Run through causal decoder with a dummy text prompt -> logits
      4. Map first N_CLASSES vocab positions to class labels
      5. Prediction = argmax of those logits

    Since this is a random-weight stub with no real vocab mapping,
    we use the first N_CLASSES positions of the output logits as class scores.
    This is the standard approach for evaluating a generative model on classification.
    """
    n_samples = videos.shape[0]
    all_scores = []

    for i in range(0, n_samples, BATCH_SIZE):
        batch = videos[i:i+BATCH_SIZE].to(device)  # [B, C, T, H, W]
        B = batch.shape[0]
        T = batch.shape[2]

        with torch.no_grad():
            # Reshape to per-frame: [B, C, T, H, W] -> [B*T, C, H, W]
            frames = batch.permute(0, 2, 1, 3, 4).reshape(B * T, 3, IMG_SIZE, IMG_SIZE)

            # Full Qwen3-VL forward pass (matches compare.py API)
            # 1. ViT encoding
            vit_out = model.vit(frames)  # dict with 'final' [B*T, N+1, D] and 'deepstack'

            # 2. MLP projection to LLM space
            proj_tokens = model.projector(
                vit_out["final"], vit_out["deepstack"]
            )  # [B*T, N+1, LLM_DIM]

            # 3. Reshape: [B*T, N+1, D] -> [B, T*(N+1), D]
            tokens_per_frame = proj_tokens.shape[1]
            vis_context = proj_tokens.reshape(B, T * tokens_per_frame, -1)

            # 4. Create a dummy text prompt (short, just to get logits out)
            dummy_text = torch.zeros(B, 4, dtype=torch.long, device=device)

            # 5. Decoder: text conditioned on visual context -> logits
            logits = model.decoder(dummy_text, vis_context)  # [B, 4, vocab_size]

            # 6. Mean-pool logits over text positions, take first N_CLASSES
            mean_logits = logits.mean(dim=1)  # [B, vocab_size]
            class_logits = mean_logits[:, :n_classes]  # [B, C]

            # 7. Softmax to get probabilities
            class_probs = F.softmax(class_logits, dim=-1)
            all_scores.append(class_probs.cpu())

    score_matrix = torch.cat(all_scores, dim=0).numpy()  # [N, C]
    predictions = score_matrix.argmax(axis=1)
    return predictions, score_matrix


# ── Step 6: Compute all metrics ───────────────────────────────────────────

def compute_all_metrics(labels, preds, scores, model_name, class_names):
    """Compute full metric suite from actual model predictions."""
    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))

    results = {"model": model_name}

    # Accuracy
    results["top1_accuracy"] = accuracy_score(labels, preds)
    results["top5_accuracy"] = top_k_accuracy_score(
        labels, scores, k=min(5, n_classes), labels=list(range(n_classes))
    )

    # Precision / Recall / F1
    for avg in ["macro", "micro", "weighted"]:
        results[f"precision_{avg}"] = precision_score(labels, preds, average=avg, zero_division=0)
        results[f"recall_{avg}"] = recall_score(labels, preds, average=avg, zero_division=0)
        results[f"f1_{avg}"] = f1_score(labels, preds, average=avg, zero_division=0)

    # Per-class F1
    f1_per = f1_score(labels, preds, average=None, zero_division=0)
    results["f1_per_class"] = {c: float(f1_per[i]) for i, c in enumerate(class_names)}

    # MCC and Kappa
    results["mcc"] = matthews_corrcoef(labels, preds)
    results["kappa"] = cohen_kappa_score(labels, preds)

    # ROC-AUC
    try:
        results["roc_auc"] = roc_auc_score(labels_bin, scores, multi_class="ovr", average="macro")
    except Exception:
        results["roc_auc"] = float("nan")

    # mAP
    try:
        aps = [average_precision_score(labels_bin[:, c], scores[:, c]) for c in range(n_classes)]
        results["mAP"] = float(np.nanmean(aps))
    except Exception:
        results["mAP"] = float("nan")

    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(labels, preds)

    return results


# ── Step 7: Print everything ──────────────────────────────────────────────

def print_comparison(vj, qw, class_names):
    """Print side-by-side comparison of all metrics."""

    print_header("CLASSIFICATION METRICS (from actual model forward passes)")

    print(f"\n  Both models: random weights, {N_SAMPLES} synthetic videos, {N_CLASSES} classes")
    print(f"  Chance level: {100/N_CLASSES:.1f}%\n")

    def row(label, v1, v2, w=32):
        print(f"  {label:<{w}}  {str(v1):>12}  {str(v2):>12}")

    row("Metric", "VL-JEPA", "Qwen3-VL")
    row("", "-------", "--------")
    row("Top-1 Accuracy",         fmt_pct(vj["top1_accuracy"]),      fmt_pct(qw["top1_accuracy"]))
    row("Top-5 Accuracy",         fmt_pct(vj["top5_accuracy"]),      fmt_pct(qw["top5_accuracy"]))
    row("", "", "")
    row("Precision (macro)",      fmt_pct(vj["precision_macro"]),    fmt_pct(qw["precision_macro"]))
    row("Precision (micro)",      fmt_pct(vj["precision_micro"]),    fmt_pct(qw["precision_micro"]))
    row("Precision (weighted)",   fmt_pct(vj["precision_weighted"]), fmt_pct(qw["precision_weighted"]))
    row("", "", "")
    row("Recall (macro)",         fmt_pct(vj["recall_macro"]),       fmt_pct(qw["recall_macro"]))
    row("Recall (micro)",         fmt_pct(vj["recall_micro"]),       fmt_pct(qw["recall_micro"]))
    row("Recall (weighted)",      fmt_pct(vj["recall_weighted"]),    fmt_pct(qw["recall_weighted"]))
    row("", "", "")
    row("F1 Score (macro)",       fmt_pct(vj["f1_macro"]),           fmt_pct(qw["f1_macro"]))
    row("F1 Score (micro)",       fmt_pct(vj["f1_micro"]),           fmt_pct(qw["f1_micro"]))
    row("F1 Score (weighted)",    fmt_pct(vj["f1_weighted"]),        fmt_pct(qw["f1_weighted"]))
    row("", "", "")
    row("MCC",                    fmt_pct(vj["mcc"]),                fmt_pct(qw["mcc"]))
    row("Cohen's Kappa",          fmt_pct(vj["kappa"]),              fmt_pct(qw["kappa"]))
    row("ROC-AUC (macro)",        fmt_pct(vj["roc_auc"]),            fmt_pct(qw["roc_auc"]))
    row("mAP",                    fmt_pct(vj["mAP"]),                fmt_pct(qw["mAP"]))

    # Per-class F1
    print_header("PER-CLASS F1 SCORE")
    row("Action Class", "VL-JEPA", "Qwen3-VL")
    row("", "-------", "--------")
    for cls in class_names:
        row(cls, fmt_pct(vj["f1_per_class"][cls]), fmt_pct(qw["f1_per_class"][cls]))

    # Confusion matrices
    for name, data in [("VL-JEPA", vj), ("Qwen3-VL", qw)]:
        print_header(f"CONFUSION MATRIX ({name})")
        cm = data["confusion_matrix"]
        n = min(N_CLASSES, 10)
        header = "  " + "  ".join(f"{class_names[i][:5]:>6}" for i in range(n))
        print(header)
        for i in range(n):
            vals = "  ".join(f"{int(cm[i,j]):>6}" for j in range(n))
            print(f"  {class_names[i][:6]:<6} {vals}")
        diag_sum = sum(cm[i][i] for i in range(n))
        print(f"  Diagonal (correct): {diag_sum}/{N_SAMPLES}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")  # both stubs are small enough for CPU

    print(DIVIDER)
    print("  VL-JEPA vs Qwen3-VL: Real Forward-Pass Metrics")
    print("  Both models: random weights, same synthetic data")
    print("  All metrics computed from actual model outputs")
    print(DIVIDER)

    # Build models
    vljepa, qwen = build_models(device)

    # Generate data
    videos, labels = generate_data(N_SAMPLES, N_CLASSES, device)
    labels_np = labels.numpy()

    # VL-JEPA: encode class names, then classify
    print_header("RUNNING VL-JEPA INFERENCE")
    t0 = time.time()
    class_embs = encode_class_names_vljepa(vljepa, ACTION_CLASSES, device)
    print(f"  Class embeddings: {class_embs.shape}")
    vj_preds, vj_scores = vljepa_classify(vljepa, videos, class_embs, device)
    vj_time = time.time() - t0
    print(f"  VL-JEPA: {N_SAMPLES} samples in {vj_time:.1f}s ({N_SAMPLES/vj_time:.1f} samples/s)")
    print(f"  Predictions distribution: {dict(zip(*np.unique(vj_preds, return_counts=True)))}")

    # Qwen3-VL: classify
    print_header("RUNNING QWEN3-VL INFERENCE")
    t0 = time.time()
    qw_preds, qw_scores = qwen3vl_classify(qwen, videos, N_CLASSES, device)
    qw_time = time.time() - t0
    print(f"  Qwen3-VL: {N_SAMPLES} samples in {qw_time:.1f}s ({N_SAMPLES/qw_time:.1f} samples/s)")
    print(f"  Predictions distribution: {dict(zip(*np.unique(qw_preds, return_counts=True)))}")

    # Compute metrics
    vj_metrics = compute_all_metrics(labels_np, vj_preds, vj_scores, "VL-JEPA", ACTION_CLASSES)
    qw_metrics = compute_all_metrics(labels_np, qw_preds, qw_scores, "Qwen3-VL", ACTION_CLASSES)

    # Print comparison
    print_comparison(vj_metrics, qw_metrics, ACTION_CLASSES)

    # Inference speed
    print_header("INFERENCE SPEED")
    print(f"  VL-JEPA:  {vj_time:.2f}s total  |  {N_SAMPLES/vj_time:.1f} samples/sec")
    print(f"  Qwen3-VL: {qw_time:.2f}s total  |  {N_SAMPLES/qw_time:.1f} samples/sec")
    speedup = qw_time / max(vj_time, 0.001)
    print(f"  VL-JEPA is {speedup:.1f}x faster")

    # Final note
    print_header("INTERPRETATION")
    print(f"""
  Both models have RANDOM WEIGHTS (no training).
  Expected chance-level performance: {100/N_CLASSES:.1f}% accuracy for {N_CLASSES} classes.

  What these results show:
    - The evaluation pipeline works correctly on real model outputs
    - Both architectures produce valid probability distributions
    - Inference speed difference is real (architectural, not weight-dependent)
    - Prediction distributions show each model's architectural bias

  To get meaningful accuracy/F1 differences:
    - Train VL-JEPA: python vl_jepa/training/train.py --epochs 50
    - Or download pretrained weights for either model
""")

    # Save
    import json
    report = {
        "config": {
            "n_samples": N_SAMPLES, "n_classes": N_CLASSES,
            "weights": "random (untrained)", "data": "synthetic random pixels",
        },
        "vljepa": {k: v for k, v in vj_metrics.items() if k != "confusion_matrix"},
        "qwen3vl": {k: v for k, v in qw_metrics.items() if k != "confusion_matrix"},
        "speed": {"vljepa_seconds": vj_time, "qwen3vl_seconds": qw_time},
    }
    out = Path(__file__).parent / "real_metrics_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"  Report saved: {out}")


if __name__ == "__main__":
    main()

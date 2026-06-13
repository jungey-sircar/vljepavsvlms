"""
VL-JEPA Training + Evaluation — End-to-End Demo
=================================================
This script:
  1. Creates a synthetic video dataset where each action class has a DISTINCT
     visual pattern (color, motion direction, texture) that the model can learn
  2. Trains VL-JEPA using our v2 loss (InfoNCE + cosine + VICReg)
  3. Evaluates precision/recall/F1/accuracy after each epoch
  4. Shows accuracy climbing from chance (~10%) toward high accuracy

Why the previous run showed ~10% accuracy:
  - Both models had RANDOM WEIGHTS (never trained)
  - Random weights = random predictions = chance level
  - This is expected and correct

What this script does differently:
  - Creates learnable data (each class has unique visual signature)
  - Trains the model so it learns to map those signatures to class embeddings
  - Evaluates with real sklearn metrics after training
"""

import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    confusion_matrix, top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vl_jepa.model import VLJepa
from vl_jepa.training.loss import VLJepaLoss

torch.manual_seed(42)
np.random.seed(42)

DIVIDER = "=" * 72

# ── 10 action classes with distinct visual patterns ────────────────────────
ACTION_CLASSES = [
    "stir soup",        # 0: circular motion, warm colors (red/orange)
    "chop vegetables",  # 1: vertical stripes, green dominant
    "mix batter",       # 2: swirl pattern, yellow/cream
    "roll dough",       # 3: horizontal rolling motion, beige
    "pour liquid",      # 4: vertical flow, blue tones
    "crack eggs",       # 5: bright white spots on dark
    "slice bread",      # 6: horizontal lines, brown
    "wash hands",       # 7: bubbles/circles, cyan/white
    "peel potato",      # 8: spiral texture, tan/brown
    "grate cheese",     # 9: diagonal lines, yellow
]
N_CLASSES = len(ACTION_CLASSES)


# ── Synthetic video dataset with learnable visual patterns ─────────────────

class SyntheticActionDataset(Dataset):
    """
    Each action class generates videos with a UNIQUE visual signature:
      - Class-specific base color (RGB)
      - Class-specific spatial pattern (stripes, circles, gradients)
      - Class-specific temporal motion (direction and speed)
      - Small random noise to prevent trivial memorization

    This is a proper ML dataset — the model must learn the mapping from
    visual patterns to class embeddings through gradient descent.
    """

    # Distinct RGB base colors per class
    CLASS_COLORS = torch.tensor([
        [0.8, 0.2, 0.1],  # 0: red (stir soup)
        [0.1, 0.7, 0.2],  # 1: green (chop vegetables)
        [0.9, 0.8, 0.3],  # 2: yellow (mix batter)
        [0.7, 0.6, 0.4],  # 3: beige (roll dough)
        [0.1, 0.3, 0.8],  # 4: blue (pour liquid)
        [0.9, 0.9, 0.9],  # 5: white (crack eggs)
        [0.5, 0.3, 0.1],  # 6: brown (slice bread)
        [0.2, 0.8, 0.8],  # 7: cyan (wash hands)
        [0.6, 0.4, 0.2],  # 8: tan (peel potato)
        [0.9, 0.9, 0.2],  # 9: bright yellow (grate cheese)
    ], dtype=torch.float32)

    def __init__(self, n_samples=500, num_frames=4, img_size=224, noise_std=0.15):
        self.n_samples = n_samples
        self.num_frames = num_frames
        self.img_size = img_size
        self.noise_std = noise_std
        self.labels = torch.randint(0, N_CLASSES, (n_samples,))

    def __len__(self):
        return self.n_samples

    def _make_pattern(self, class_id, t, H, W):
        """Generate a class-specific spatial pattern for frame t."""
        color = self.CLASS_COLORS[class_id]  # [3]
        frame = torch.zeros(3, H, W)

        # Spatial coordinates
        yy = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
        xx = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)

        # Time-varying offset for motion
        phase = t / self.num_frames

        if class_id == 0:    # Circular motion (stir)
            cx = 0.5 + 0.2 * math.cos(2 * math.pi * phase)
            cy = 0.5 + 0.2 * math.sin(2 * math.pi * phase)
            r = ((xx - cx)**2 + (yy - cy)**2).sqrt()
            pattern = (r < 0.3).float()
        elif class_id == 1:  # Vertical stripes (chop)
            freq = 8.0
            pattern = (torch.sin(freq * xx * 2 * math.pi + phase * 4) > 0).float()
        elif class_id == 2:  # Swirl (mix)
            angle = torch.atan2(yy - 0.5, xx - 0.5)
            r = ((xx - 0.5)**2 + (yy - 0.5)**2).sqrt()
            pattern = (torch.sin(angle * 3 + r * 10 + phase * 6) > 0).float()
        elif class_id == 3:  # Horizontal rolling (roll)
            pattern = (torch.sin(6 * yy * 2 * math.pi + phase * 8) > 0).float()
        elif class_id == 4:  # Vertical flow (pour)
            flow_pos = phase
            pattern = ((yy - flow_pos).abs() < 0.15).float()
        elif class_id == 5:  # Bright spots (crack eggs)
            pattern = torch.zeros(H, W)
            n_spots = 5
            for s in range(n_spots):
                sx = 0.2 + 0.6 * (s / n_spots)
                sy = 0.3 + 0.1 * math.sin(s + phase * 4)
                pattern += (((xx - sx)**2 + (yy - sy)**2).sqrt() < 0.08).float()
            pattern = pattern.clamp(0, 1)
        elif class_id == 6:  # Horizontal lines (slice)
            pattern = (torch.sin(12 * yy * 2 * math.pi) > 0).float()
            # Shift over time
            shift = int(phase * 20) % H
            pattern = torch.roll(pattern, shift, dims=0)
        elif class_id == 7:  # Bubbles (wash)
            pattern = torch.zeros(H, W)
            for b in range(8):
                bx = (0.1 + 0.1 * b) % 1.0
                by = (0.15 * b + phase * 0.5) % 1.0
                pattern += (((xx - bx)**2 + (yy - by)**2).sqrt() < 0.06).float()
            pattern = pattern.clamp(0, 1)
        elif class_id == 8:  # Spiral (peel)
            r = ((xx - 0.5)**2 + (yy - 0.5)**2).sqrt()
            angle = torch.atan2(yy - 0.5, xx - 0.5)
            pattern = (torch.sin(angle * 2 + r * 15 + phase * 5) > 0).float()
        else:                # Diagonal lines (grate cheese)
            pattern = (torch.sin(8 * (xx + yy) * 2 * math.pi + phase * 6) > 0).float()

        # Apply color
        for c in range(3):
            frame[c] = pattern * color[c] + (1 - pattern) * (1 - color[c]) * 0.2

        return frame

    def __getitem__(self, idx):
        class_id = self.labels[idx].item()
        H, W = self.img_size, self.img_size

        # Generate video frames with class-specific pattern + noise
        frames = []
        for t in range(self.num_frames):
            frame = self._make_pattern(class_id, t, H, W)
            noise = torch.randn_like(frame) * self.noise_std
            frame = (frame + noise).clamp(0, 1)
            frames.append(frame)

        video = torch.stack(frames, dim=1)  # [C, T, H, W]

        # Text: encode the class name as byte-level tokens
        name = ACTION_CLASSES[class_id]
        token_ids = [b for b in name.encode("utf-8")]
        max_len = 32
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids += [0] * (max_len - len(token_ids))

        return {
            "video": video,
            "label": class_id,
            "query_input_ids": torch.tensor(token_ids, dtype=torch.long),
            "target_input_ids": torch.tensor(token_ids, dtype=torch.long),
        }


# ── Evaluation function ───────────────────────────────────────────────────

def evaluate(model, class_embeddings, eval_videos, eval_labels, device, query_tensor):
    """Classify eval videos using cosine similarity, return full metrics."""
    model.eval()
    class_emb_norm = F.normalize(class_embeddings, dim=-1)

    all_scores = []
    BS = 8
    for i in range(0, len(eval_videos), BS):
        batch = eval_videos[i:i+BS].to(device)
        B = batch.shape[0]
        q = query_tensor.expand(B, -1).to(device)
        with torch.no_grad():
            pred_emb = model(batch, q)
            pred_norm = F.normalize(pred_emb, dim=-1)
            sims = pred_norm @ class_emb_norm.T
            all_scores.append(sims.cpu())

    scores = torch.cat(all_scores, dim=0).numpy()
    preds = scores.argmax(axis=1)
    labels = eval_labels.numpy()

    labels_bin = label_binarize(labels, classes=list(range(N_CLASSES)))

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "kappa": cohen_kappa_score(labels, preds),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels_bin, scores, multi_class="ovr", average="macro")
    except:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["top5_accuracy"] = top_k_accuracy_score(
            labels, scores, k=min(5, N_CLASSES), labels=list(range(N_CLASSES))
        )
    except:
        metrics["top5_accuracy"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(labels, preds)
    return metrics


# ── Encode class names ────────────────────────────────────────────────────

def encode_classes(model, device):
    class_embs = []
    for name in ACTION_CLASSES:
        ids = [b for b in name.encode("utf-8")]
        ids = ids[:32] + [0] * max(0, 32 - len(ids))
        t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            emb = model.y_encoder(t)
        class_embs.append(emb)
    return torch.cat(class_embs, dim=0)


# ── Collate function ──────────────────────────────────────────────────────

def collate(batch):
    return {
        "video": torch.stack([b["video"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "query_input_ids": torch.stack([b["query_input_ids"] for b in batch]),
        "target_input_ids": torch.stack([b["target_input_ids"] for b in batch]),
    }


# ── Main training + evaluation ────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{DIVIDER}")
    print("  VL-JEPA: Train + Evaluate (Synthetic Action Dataset)")
    print(f"  Device: {device}")
    print(DIVIDER)

    # ── Dataset ───────────────────────────────────────────────────────────
    N_TRAIN = 500
    N_EVAL  = 200
    NUM_FRAMES = 4
    IMG_SIZE = 224
    EPOCHS = 15
    BATCH_SIZE = 8
    LR = 5e-4

    print(f"\n  Train samples: {N_TRAIN}")
    print(f"  Eval samples:  {N_EVAL}")
    print(f"  Classes:       {N_CLASSES}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")

    train_ds = SyntheticActionDataset(N_TRAIN, NUM_FRAMES, IMG_SIZE, noise_std=0.15)
    eval_ds  = SyntheticActionDataset(N_EVAL,  NUM_FRAMES, IMG_SIZE, noise_std=0.10)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate, drop_last=True)

    # Pre-generate eval data
    eval_videos = torch.stack([eval_ds[i]["video"] for i in range(N_EVAL)])
    eval_labels = eval_ds.labels

    print(f"  Train label dist: {dict(zip(*np.unique(train_ds.labels.numpy(), return_counts=True)))}")
    print(f"  Eval label dist:  {dict(zip(*np.unique(eval_labels.numpy(), return_counts=True)))}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = VLJepa.build_default(
        mode="standalone", num_frames=NUM_FRAMES, with_decoder=False
    ).to(device)

    # Unfreeze predictor (the part that learns)
    for p in model.predictor.parameters():
        p.requires_grad_(True)
    # Also unfreeze x_encoder so it can learn visual patterns
    for p in model.x_encoder.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M total params")
    print(f"  Trainable: {n_trainable/1e6:.2f}M params")

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_fn = VLJepaLoss(
        alpha=1.0, beta=0.5, gamma=1.0,
        use_infonce=True, use_vicreg=True,
        temperature=0.07, hard_neg_weight=2.0,
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    # Add InfoNCE temperature param
    if hasattr(loss_fn.infonce, 'log_temp'):
        optimizer.add_param_group({
            "params": [loss_fn.infonce.log_temp],
            "lr": LR * 0.1, "weight_decay": 0.0,
        })

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Generic query ─────────────────────────────────────────────────────
    query_text = "what action"
    q_ids = [b for b in query_text.encode("utf-8")]
    q_ids = q_ids[:32] + [0] * max(0, 32 - len(q_ids))
    query_tensor = torch.tensor([q_ids], dtype=torch.long, device=device)

    # ── Pre-training eval ─────────────────────────────────────────────────
    class_embs = encode_classes(model, device)
    pre_metrics = evaluate(model, class_embs, eval_videos, eval_labels, device, query_tensor)
    print(f"\n{DIVIDER}")
    print(f"  BEFORE TRAINING (random weights)")
    print(f"{DIVIDER}")
    print(f"  Accuracy:  {pre_metrics['accuracy']*100:.2f}%")
    print(f"  F1 (macro): {pre_metrics['f1_macro']*100:.2f}%")
    print(f"  Precision:  {pre_metrics['precision_macro']*100:.2f}%")
    print(f"  Recall:     {pre_metrics['recall_macro']*100:.2f}%")
    print(f"  ROC-AUC:   {pre_metrics['roc_auc']*100:.2f}%")
    print(f"  (Chance level: {100/N_CLASSES:.1f}%)")

    # ── Training loop ─────────────────────────────────────────────────────
    history = []
    print(f"\n{DIVIDER}")
    print(f"  TRAINING")
    print(DIVIDER)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_nce = 0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            video = batch["video"].to(device)
            query = batch["query_input_ids"].to(device)
            target = batch["target_input_ids"].to(device)

            # Forward: predict embedding from video + query
            pred_emb = model(video, query)

            # Target: encode the target text (class name) through Y-Encoder
            with torch.no_grad():
                target_emb = model.y_encoder(target)

            # Combined loss: cosine + MSE + InfoNCE + VICReg
            loss_dict = loss_fn(pred_emb, target_emb)
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_nce += loss_dict["infonce_loss"].item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_nce = epoch_nce / max(n_batches, 1)
        elapsed = time.time() - t0

        # Evaluate
        class_embs = encode_classes(model, device)
        metrics = evaluate(model, class_embs, eval_videos, eval_labels, device, query_tensor)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "nce": avg_nce,
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        })

        acc = metrics["accuracy"]
        f1 = metrics["f1_macro"]
        prec = metrics["precision_macro"]
        rec = metrics["recall_macro"]
        auc = metrics["roc_auc"]
        bar = "█" * int(acc * 50)

        print(
            f"  Epoch {epoch:2d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | NCE: {avg_nce:.4f} | "
            f"Acc: {acc*100:5.1f}% | F1: {f1*100:5.1f}% | "
            f"P: {prec*100:5.1f}% | R: {rec*100:5.1f}% | "
            f"AUC: {auc*100:5.1f}% | {elapsed:.1f}s "
            f"|{bar}"
        )

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"  AFTER TRAINING ({EPOCHS} epochs)")
    print(DIVIDER)

    final = history[-1]
    print(f"  Top-1 Accuracy:     {final['accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy:     {final['top5_accuracy']*100:.2f}%")
    print(f"  Precision (macro):  {final['precision_macro']*100:.2f}%")
    print(f"  Recall (macro):     {final['recall_macro']*100:.2f}%")
    print(f"  F1 Score (macro):   {final['f1_macro']*100:.2f}%")
    print(f"  F1 Score (weighted):{final['f1_weighted']*100:.2f}%")
    print(f"  MCC:                {final['mcc']*100:.2f}%")
    print(f"  Cohen's Kappa:      {final['kappa']*100:.2f}%")
    print(f"  ROC-AUC:            {final['roc_auc']*100:.2f}%")

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    header = "         " + "".join(f"{ACTION_CLASSES[i][:6]:>7}" for i in range(N_CLASSES))
    print(header)
    for i in range(N_CLASSES):
        vals = "".join(f"{int(cm[i,j]):>7}" for j in range(N_CLASSES))
        print(f"  {ACTION_CLASSES[i][:7]:<7} {vals}")
    diag = sum(cm[i][i] for i in range(N_CLASSES))
    print(f"  Diagonal (correct): {diag}/{N_EVAL}")

    # ── Improvement summary ───────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"  IMPROVEMENT SUMMARY")
    print(DIVIDER)
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Improvement':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")

    def row(name, before, after):
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<25} {before*100:>9.2f}% {after*100:>9.2f}% {sign}{delta*100:>10.2f}%")

    row("Top-1 Accuracy",     pre_metrics["accuracy"],        final["accuracy"])
    row("Precision (macro)",  pre_metrics["precision_macro"],  final["precision_macro"])
    row("Recall (macro)",     pre_metrics["recall_macro"],     final["recall_macro"])
    row("F1 Score (macro)",   pre_metrics["f1_macro"],         final["f1_macro"])
    row("ROC-AUC",            pre_metrics["roc_auc"],          final["roc_auc"])
    row("MCC",                pre_metrics["mcc"],              final["mcc"])

    # ── Training curve ────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"  TRAINING CURVE (accuracy per epoch)")
    print(DIVIDER)
    for h in history:
        bar_len = int(h["accuracy"] * 60)
        bar = "█" * bar_len + "░" * (60 - bar_len)
        print(f"  Epoch {h['epoch']:2d}: {h['accuracy']*100:5.1f}% |{bar}|")

    print(f"\n{DIVIDER}")
    print(f"  DONE. VL-JEPA trained and evaluated on synthetic action dataset.")
    print(DIVIDER)


if __name__ == "__main__":
    main()

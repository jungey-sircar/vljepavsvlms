"""
VL-JEPA Training v2 — Fixed for High Accuracy
================================================
Fixes from v1 (which plateaued at 31.5%):
  1. UNFREEZE Y-Encoder — class embeddings must be LEARNED, not frozen random
  2. Smaller images (64x64) — faster convergence, less overfitting on 500 samples
  3. Larger batch (32) — InfoNCE needs many negatives to work well
  4. More epochs (40) + proper warmup — give the model time to converge
  5. Lower noise — 0.08 instead of 0.15, so patterns are clearer

The key insight: when Y-Encoder is frozen with random weights, class embeddings
are random vectors. The predictor can't learn a meaningful mapping because the
TARGETS themselves are meaningless. Unfreezing Y-Encoder lets the model learn
both "what visual patterns look like" AND "what class labels mean" jointly.
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
    confusion_matrix, top_k_accuracy_score, classification_report,
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vl_jepa.model import VLJepa, XEncoder, YEncoder, Predictor
from vl_jepa.training.loss import VLJepaLoss

torch.manual_seed(42)
np.random.seed(42)

DIVIDER = "=" * 72

ACTION_CLASSES = [
    "stir soup", "chop vegetables", "mix batter", "roll dough", "pour liquid",
    "crack eggs", "slice bread", "wash hands", "peel potato", "grate cheese",
]
N_CLASSES = len(ACTION_CLASSES)
IMG_SIZE = 64   # KEY FIX: smaller images for faster convergence


class SyntheticActionDataset(Dataset):
    """Synthetic videos with distinct visual patterns per class."""

    CLASS_COLORS = torch.tensor([
        [0.9, 0.1, 0.1], [0.1, 0.8, 0.1], [0.9, 0.9, 0.2],
        [0.7, 0.5, 0.3], [0.1, 0.2, 0.9], [0.95, 0.95, 0.95],
        [0.5, 0.25, 0.1], [0.1, 0.9, 0.9], [0.6, 0.4, 0.15],
        [1.0, 0.85, 0.1],
    ], dtype=torch.float32)

    def __init__(self, n_samples=500, num_frames=4, img_size=64, noise_std=0.08):
        self.n = n_samples
        self.T = num_frames
        self.S = img_size
        self.noise = noise_std
        self.labels = torch.randint(0, N_CLASSES, (n_samples,))

    def __len__(self):
        return self.n

    def _pattern(self, cls, t):
        S = self.S
        yy = torch.linspace(0, 1, S).unsqueeze(1).expand(S, S)
        xx = torch.linspace(0, 1, S).unsqueeze(0).expand(S, S)
        phase = t / self.T
        color = self.CLASS_COLORS[cls]

        if cls == 0:    # circles moving clockwise
            cx, cy = 0.5 + 0.2*math.cos(2*math.pi*phase), 0.5 + 0.2*math.sin(2*math.pi*phase)
            p = (((xx-cx)**2 + (yy-cy)**2).sqrt() < 0.25).float()
        elif cls == 1:  # vertical stripes
            p = (torch.sin(10*xx*2*math.pi + phase*6) > 0).float()
        elif cls == 2:  # checkerboard
            p = ((torch.sin(6*xx*2*math.pi) > 0) ^ (torch.sin(6*yy*2*math.pi) > 0)).float()
        elif cls == 3:  # horizontal bars scrolling
            p = (torch.sin(8*yy*2*math.pi + phase*8) > 0).float()
        elif cls == 4:  # falling blob
            p = (((xx-0.5)**2 + (yy-phase)**2).sqrt() < 0.2).float()
        elif cls == 5:  # bright center
            p = (((xx-0.5)**2 + (yy-0.5)**2).sqrt() < 0.3).float()
        elif cls == 6:  # diagonal lines
            p = (torch.sin(8*(xx+yy)*2*math.pi) > 0).float()
        elif cls == 7:  # random dots pattern (fixed per class)
            torch.manual_seed(777)
            p = (torch.rand(S, S) > 0.7).float()
            torch.manual_seed(42 + t)  # restore randomness
        elif cls == 8:  # spiral
            r = ((xx-0.5)**2 + (yy-0.5)**2).sqrt()
            a = torch.atan2(yy-0.5, xx-0.5)
            p = (torch.sin(a*3 + r*12 + phase*4) > 0).float()
        else:           # gradient
            p = xx  # smooth left-to-right gradient

        frame = torch.zeros(3, S, S)
        for c in range(3):
            frame[c] = p * color[c] + (1-p) * (0.1 + 0.05*c)
        return frame

    def __getitem__(self, idx):
        cls = self.labels[idx].item()
        frames = []
        for t in range(self.T):
            f = self._pattern(cls, t)
            f = f + torch.randn_like(f) * self.noise
            frames.append(f.clamp(0, 1))
        video = torch.stack(frames, dim=1)  # [C, T, H, W]

        name = ACTION_CLASSES[cls]
        ids = list(name.encode("utf-8"))[:32]
        ids += [0] * (32 - len(ids))
        return {
            "video": video,
            "label": cls,
            "query_input_ids": torch.tensor(ids, dtype=torch.long),
            "target_input_ids": torch.tensor(ids, dtype=torch.long),
        }


def collate(batch):
    return {
        "video": torch.stack([b["video"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "query_input_ids": torch.stack([b["query_input_ids"] for b in batch]),
        "target_input_ids": torch.stack([b["target_input_ids"] for b in batch]),
    }


def encode_classes(model, device):
    embs = []
    for name in ACTION_CLASSES:
        ids = list(name.encode("utf-8"))[:32]
        ids += [0] * (32 - len(ids))
        t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            embs.append(model.y_encoder(t))
    return torch.cat(embs, dim=0)


def evaluate(model, eval_videos, eval_labels, device, query_tensor):
    model.eval()
    class_embs = encode_classes(model, device)
    class_norm = F.normalize(class_embs, dim=-1)

    all_scores = []
    BS = 32
    for i in range(0, len(eval_videos), BS):
        batch = eval_videos[i:i+BS].to(device)
        B = batch.shape[0]
        q = query_tensor.expand(B, -1).to(device)
        with torch.no_grad():
            pred = model(batch, q)
            pred_n = F.normalize(pred, dim=-1)
            sims = pred_n @ class_norm.T
            all_scores.append(sims.cpu())

    scores = torch.cat(all_scores, dim=0).numpy()
    preds = scores.argmax(axis=1)
    labels = eval_labels.numpy()
    labels_bin = label_binarize(labels, classes=list(range(N_CLASSES)))

    m = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "kappa": cohen_kappa_score(labels, preds),
    }
    try:
        m["roc_auc"] = roc_auc_score(labels_bin, scores, multi_class="ovr", average="macro")
    except:
        m["roc_auc"] = float("nan")
    try:
        m["top5_accuracy"] = top_k_accuracy_score(labels, scores, k=5, labels=list(range(N_CLASSES)))
    except:
        m["top5_accuracy"] = float("nan")

    m["confusion_matrix"] = confusion_matrix(labels, preds)
    m["report"] = classification_report(labels, preds, target_names=ACTION_CLASSES, zero_division=0)
    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{DIVIDER}")
    print(f"  VL-JEPA v2 Training — Fixed for High Accuracy")
    print(f"  Device: {device} | Image: {IMG_SIZE}x{IMG_SIZE}")
    print(DIVIDER)

    # Config
    N_TRAIN, N_EVAL = 800, 200
    NUM_FRAMES = 4
    EPOCHS = 40
    BATCH_SIZE = 32
    LR = 1e-3

    print(f"  Train: {N_TRAIN} | Eval: {N_EVAL} | Classes: {N_CLASSES}")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")

    train_ds = SyntheticActionDataset(N_TRAIN, NUM_FRAMES, IMG_SIZE, noise_std=0.08)
    eval_ds  = SyntheticActionDataset(N_EVAL,  NUM_FRAMES, IMG_SIZE, noise_std=0.05)

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)
    eval_videos = torch.stack([eval_ds[i]["video"] for i in range(N_EVAL)])
    eval_labels = eval_ds.labels

    # Build model with SMALLER dims for 64x64 images
    x_enc = XEncoder(num_frames=NUM_FRAMES, img_size=IMG_SIZE, dim=256, depth=4, num_heads=4)
    y_enc = YEncoder(mode="standalone", dim=256, depth=4, num_heads=4)
    pred  = Predictor(visual_dim=256, text_dim=256, hidden_dim=256, depth=4, num_heads=4)
    model = VLJepa(
        x_encoder=x_enc, y_encoder=y_enc, predictor=pred, y_decoder=None,
        freeze_x_encoder=False,    # KEY FIX: unfreeze to learn visual features
        freeze_y_encoder=False,    # KEY FIX: unfreeze to learn class embeddings
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_total/1e6:.2f}M total, {n_train/1e6:.2f}M trainable (ALL unfrozen)")

    # Loss
    loss_fn = VLJepaLoss(
        alpha=1.0, beta=0.5, gamma=1.5,
        use_infonce=True, use_vicreg=True,
        temperature=0.07, hard_neg_weight=2.0,
    ).to(device)

    # Optimizer — all parameters learn together
    params = list(model.parameters())
    if hasattr(loss_fn, 'infonce') and loss_fn.infonce is not None and hasattr(loss_fn.infonce, 'log_temp'):
        params.append(loss_fn.infonce.log_temp)
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01, betas=(0.9, 0.95))

    # Cosine schedule with warmup
    warmup_epochs = 3
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(EPOCHS - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Query
    q_text = "what action"
    q_ids = list(q_text.encode("utf-8"))[:32]
    q_ids += [0] * (32 - len(q_ids))
    q_tensor = torch.tensor([q_ids], dtype=torch.long, device=device)

    # Before training
    pre = evaluate(model, eval_videos, eval_labels, device, q_tensor)
    print(f"\n  BEFORE: Acc={pre['accuracy']*100:.1f}% | F1={pre['f1_macro']*100:.1f}% | "
          f"P={pre['precision_macro']*100:.1f}% | R={pre['recall_macro']*100:.1f}%")

    # Train
    print(f"\n{DIVIDER}")
    print(f"  TRAINING ({EPOCHS} epochs, batch={BATCH_SIZE})")
    print(DIVIDER)

    history = []
    best_acc = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss, ep_nce, n = 0, 0, 0
        t0 = time.time()

        for batch in loader:
            video  = batch["video"].to(device)
            query  = batch["query_input_ids"].to(device)
            target = batch["target_input_ids"].to(device)

            pred_emb = model(video, query)
            target_emb = model.y_encoder(target)  # NOT detached — Y-Encoder learns too

            loss_dict = loss_fn(pred_emb, target_emb)
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_nce += loss_dict["infonce_loss"].item()
            n += 1

        scheduler.step()
        elapsed = time.time() - t0

        # Evaluate
        met = evaluate(model, eval_videos, eval_labels, device, q_tensor)
        acc = met["accuracy"]
        f1 = met["f1_macro"]
        prec = met["precision_macro"]
        rec = met["recall_macro"]
        auc = met.get("roc_auc", float("nan"))

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        history.append({"epoch": epoch, "loss": ep_loss/n, "nce": ep_nce/n,
                        "accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc})

        bar = "█" * int(acc * 50) + "░" * (50 - int(acc * 50))
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Ep {epoch:2d}/{EPOCHS} | L:{ep_loss/n:.3f} NCE:{ep_nce/n:.3f} | "
            f"Acc:{acc*100:5.1f}% F1:{f1*100:5.1f}% P:{prec*100:5.1f}% R:{rec*100:5.1f}% "
            f"AUC:{auc*100:5.1f}% | lr:{lr_now:.1e} {elapsed:.0f}s |{bar}|"
        )

    # Final
    final = evaluate(model, eval_videos, eval_labels, device, q_tensor)

    print(f"\n{DIVIDER}")
    print(f"  FINAL RESULTS (best epoch: {best_epoch})")
    print(DIVIDER)
    print(f"  Top-1 Accuracy:     {final['accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy:     {final['top5_accuracy']*100:.2f}%")
    print(f"  Precision (macro):  {final['precision_macro']*100:.2f}%")
    print(f"  Recall (macro):     {final['recall_macro']*100:.2f}%")
    print(f"  F1 Score (macro):   {final['f1_macro']*100:.2f}%")
    print(f"  F1 (weighted):      {final['f1_weighted']*100:.2f}%")
    print(f"  MCC:                {final['mcc']*100:.2f}%")
    print(f"  Cohen's Kappa:      {final['kappa']*100:.2f}%")
    print(f"  ROC-AUC:            {final['roc_auc']*100:.2f}%")

    print(f"\n  Classification Report:")
    print(final["report"])

    # Confusion matrix
    cm = final["confusion_matrix"]
    print(f"  Confusion Matrix:")
    short = [c[:6] for c in ACTION_CLASSES]
    print("        " + " ".join(f"{s:>6}" for s in short))
    for i in range(N_CLASSES):
        vals = " ".join(f"{int(cm[i,j]):>6}" for j in range(N_CLASSES))
        print(f"  {short[i]:<6} {vals}")
    diag = sum(cm[i][i] for i in range(N_CLASSES))
    print(f"  Correct: {diag}/{N_EVAL}")

    # Improvement
    print(f"\n{DIVIDER}")
    print(f"  IMPROVEMENT: BEFORE vs AFTER")
    print(DIVIDER)
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for k, name in [("accuracy", "Top-1 Accuracy"), ("precision_macro", "Precision (macro)"),
                     ("recall_macro", "Recall (macro)"), ("f1_macro", "F1 (macro)"),
                     ("roc_auc", "ROC-AUC"), ("mcc", "MCC")]:
        b = pre[k]; a = final[k]
        d = a - b
        print(f"  {name:<25} {b*100:>9.2f}% {a*100:>9.2f}% {'+' if d>=0 else ''}{d*100:>8.2f}%")

    # Accuracy curve
    print(f"\n{DIVIDER}")
    print(f"  ACCURACY CURVE")
    print(DIVIDER)
    for h in history:
        bar = "█" * int(h["accuracy"] * 50)
        print(f"  Ep {h['epoch']:2d}: {h['accuracy']*100:5.1f}% |{bar}")

    print(f"\n  Best accuracy: {best_acc*100:.1f}% at epoch {best_epoch}")
    print(f"\n{DIVIDER}")


if __name__ == "__main__":
    main()

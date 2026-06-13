"""
VL-JEPA vs Qwen3-VL — TRAINED Head-to-Head Comparison
=======================================================
Trains BOTH models on the same synthetic action dataset, then computes
full precision/recall/F1/accuracy/AUC/MCC from real forward passes.

Both models use the same approach:
  - ViT backbone → pool → classification head (Linear) → CrossEntropy
  - Same data, same labels, same eval set
  - Same training recipe (epochs, LR, optimizer)

This gives an honest apples-to-apples comparison.
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

from vl_jepa.model import XEncoder

torch.manual_seed(42)
np.random.seed(42)

DIVIDER = "=" * 72
IMG_SIZE = 64
NUM_FRAMES = 4
N_CLASSES = 10

ACTION_CLASSES = [
    "stir soup", "chop vegetables", "mix batter", "roll dough", "pour liquid",
    "crack eggs", "slice bread", "wash hands", "peel potato", "grate cheese",
]


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

class VLJepaClassifier(nn.Module):
    """VL-JEPA: ViT (spatiotemporal patches) → pool → classification head."""

    def __init__(self, n_classes, dim=128, depth=4, num_heads=4):
        super().__init__()
        self.encoder = XEncoder(
            num_frames=NUM_FRAMES, img_size=IMG_SIZE,
            dim=dim, depth=depth, num_heads=num_heads,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, n_classes),
        )

    def forward(self, video):
        tokens = self.encoder(video)      # [B, N, D]
        emb = tokens.mean(dim=1)          # [B, D]
        return self.head(emb)             # [B, C]


class Qwen3VLClassifier(nn.Module):
    """
    Qwen3-VL style: per-frame ViT → MLP projector → pool → classification head.
    Key architectural differences from VL-JEPA:
      - Processes frames independently (no spatiotemporal patches)
      - Uses MLP projector (2-layer) between ViT and classifier
      - Simulates Qwen3-VL's frame-by-frame ViT + projection pipeline
    """

    def __init__(self, n_classes, vit_dim=128, proj_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.vit_dim = vit_dim
        n_patches = (IMG_SIZE // 16) ** 2  # patch_size=16

        # Per-frame ViT
        self.patch_embed = nn.Conv2d(3, vit_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, vit_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=vit_dim, nhead=num_heads,
                dim_feedforward=vit_dim * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.vit_norm = nn.LayerNorm(vit_dim)

        # MLP Projector (Qwen3-VL style: 2-layer MLP)
        self.projector = nn.Sequential(
            nn.Linear(vit_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, n_classes),
        )

    def encode_frame(self, x):
        """Encode a single frame: [B, 3, H, W] -> [B, N+1, D]"""
        B = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        for block in self.blocks:
            tokens = block(tokens)
        return self.vit_norm(tokens)

    def forward(self, video):
        """
        video: [B, 3, T, H, W]
        Process each frame independently (Qwen3-VL style), project, pool, classify.
        """
        B, C, T, H, W = video.shape
        # Extract per-frame features
        frame_feats = []
        for t in range(T):
            frame = video[:, :, t, :, :]       # [B, 3, H, W]
            tokens = self.encode_frame(frame)   # [B, N+1, D]
            cls_feat = tokens[:, 0, :]          # [B, D] — CLS token
            frame_feats.append(cls_feat)

        # Temporal average of frame features
        video_feat = torch.stack(frame_feats, dim=1).mean(dim=1)  # [B, D]

        # Project (Qwen3-VL MLP projector)
        projected = self.projector(video_feat)  # [B, proj_dim]

        # Classify
        return self.head(projected)             # [B, C]


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticActionDataset(Dataset):
    CLASS_COLORS = torch.tensor([
        [0.9, 0.1, 0.1], [0.1, 0.8, 0.1], [0.9, 0.9, 0.2],
        [0.7, 0.5, 0.3], [0.1, 0.2, 0.9], [0.95, 0.95, 0.95],
        [0.5, 0.25, 0.1], [0.1, 0.9, 0.9], [0.6, 0.4, 0.15],
        [1.0, 0.85, 0.1],
    ])

    def __init__(self, n, noise=0.10):
        self.n = n
        self.noise = noise
        self.labels = torch.randint(0, N_CLASSES, (n,))

    def __len__(self):
        return self.n

    def _pat(self, cls, t):
        S = IMG_SIZE
        yy = torch.linspace(0, 1, S).unsqueeze(1).expand(S, S)
        xx = torch.linspace(0, 1, S).unsqueeze(0).expand(S, S)
        ph = t / NUM_FRAMES
        c = self.CLASS_COLORS[cls]
        if cls == 0:
            cx, cy = 0.5+0.2*math.cos(2*math.pi*ph), 0.5+0.2*math.sin(2*math.pi*ph)
            p = (((xx-cx)**2+(yy-cy)**2).sqrt() < 0.25).float()
        elif cls == 1:
            p = (torch.sin(10*xx*2*math.pi+ph*6) > 0).float()
        elif cls == 2:
            p = ((torch.sin(6*xx*2*math.pi) > 0) ^ (torch.sin(6*yy*2*math.pi) > 0)).float()
        elif cls == 3:
            p = (torch.sin(8*yy*2*math.pi+ph*8) > 0).float()
        elif cls == 4:
            p = (((xx-0.5)**2+(yy-ph)**2).sqrt() < 0.2).float()
        elif cls == 5:
            p = (((xx-0.5)**2+(yy-0.5)**2).sqrt() < (0.2+0.1*ph)).float()
        elif cls == 6:
            p = (torch.sin(8*(xx+yy)*2*math.pi+ph*4) > 0).float()
        elif cls == 7:
            p = (((xx*5%1-0.5)**2+(yy*5%1-0.5)**2).sqrt() < 0.15).float()
        elif cls == 8:
            r = ((xx-0.5)**2+(yy-0.5)**2).sqrt()
            a = torch.atan2(yy-0.5, xx-0.5)
            p = (torch.sin(a*3+r*15+ph*5) > 0).float()
        else:
            p = xx
        frame = torch.zeros(3, S, S)
        for ch in range(3):
            frame[ch] = p * c[ch] + (1-p) * 0.05
        return frame

    def __getitem__(self, idx):
        cls = self.labels[idx].item()
        frames = [
            (self._pat(cls, t) + torch.randn(3, IMG_SIZE, IMG_SIZE)*self.noise).clamp(0, 1)
            for t in range(NUM_FRAMES)
        ]
        return {"video": torch.stack(frames, dim=1), "label": cls}


def collate(batch):
    return {
        "video": torch.stack([b["video"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Train + Evaluate
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, eval_vids, eval_labs, device):
    model.eval()
    all_logits = []
    for i in range(0, len(eval_vids), 64):
        with torch.no_grad():
            all_logits.append(model(eval_vids[i:i+64].to(device)).cpu())
    logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    labs = eval_labs.numpy()
    labs_bin = label_binarize(labs, classes=list(range(N_CLASSES)))

    m = {
        "accuracy": accuracy_score(labs, preds),
        "precision_macro": precision_score(labs, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(labs, preds, average="weighted", zero_division=0),
        "recall_macro": recall_score(labs, preds, average="macro", zero_division=0),
        "recall_weighted": recall_score(labs, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(labs, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labs, preds, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(labs, preds),
        "kappa": cohen_kappa_score(labs, preds),
    }
    try: m["roc_auc"] = roc_auc_score(labs_bin, probs, multi_class="ovr", average="macro")
    except: m["roc_auc"] = float("nan")
    try: m["top5_acc"] = top_k_accuracy_score(labs, probs, k=5, labels=list(range(N_CLASSES)))
    except: m["top5_acc"] = float("nan")
    m["cm"] = confusion_matrix(labs, preds)
    m["report"] = classification_report(labs, preds, target_names=ACTION_CLASSES, zero_division=0)
    return m


def train_model(model, name, loader, eval_vids, eval_labs, device, epochs=50, lr=3e-3):
    """Train a model and return per-epoch history + final metrics."""
    print(f"\n{'─'*72}")
    print(f"  Training: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"{'─'*72}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warmup = 3
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: (ep+1)/warmup if ep < warmup else
        0.5*(1+math.cos(math.pi*(ep-warmup)/max(epochs-warmup, 1)))
    )

    history = []
    best_acc, best_state = 0, None

    print(f"  {'Ep':>3} {'Loss':>7} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7} {'Time':>5}")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, n = 0, 0
        t0 = time.time()
        for batch in loader:
            video = batch["video"].to(device)
            label = batch["label"].to(device)
            logits = model(video)
            loss = criterion(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); n += 1
        scheduler.step()
        elapsed = time.time() - t0

        met = evaluate(model, eval_vids, eval_labs, device)
        acc = met["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "loss": ep_loss/n, **{k: v for k, v in met.items() if k not in ("cm", "report")}})
        bar = "█" * int(acc * 30)
        print(f"  {epoch:3d} {ep_loss/n:7.4f} {acc*100:6.1f}% {met['f1_macro']*100:6.1f}% "
              f"{met['precision_macro']*100:6.1f}% {met['recall_macro']*100:6.1f}% "
              f"{met.get('roc_auc',0)*100:6.1f}% {elapsed:4.0f}s |{bar}|")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    final = evaluate(model, eval_vids, eval_labs, device)
    return history, final, best_acc


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_TRAIN, N_EVAL, EPOCHS, BS, LR = 1000, 300, 50, 32, 3e-3

    print(f"\n{DIVIDER}")
    print(f"  VL-JEPA vs Qwen3-VL — Trained Head-to-Head Comparison")
    print(f"  {device} | {IMG_SIZE}x{IMG_SIZE} | {N_TRAIN} train | {N_EVAL} eval | {EPOCHS} epochs")
    print(DIVIDER)

    # Data (SAME for both models)
    train_ds = SyntheticActionDataset(N_TRAIN, noise=0.10)
    eval_ds  = SyntheticActionDataset(N_EVAL,  noise=0.08)
    loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0,
                        collate_fn=collate, drop_last=True)
    eval_vids = torch.stack([eval_ds[i]["video"] for i in range(N_EVAL)])
    eval_labs = eval_ds.labels

    # ── Train VL-JEPA ─────────────────────────────────────────────────────
    vljepa = VLJepaClassifier(N_CLASSES, dim=128, depth=4, num_heads=4)
    vj_hist, vj_final, vj_best = train_model(vljepa, "VL-JEPA", loader, eval_vids, eval_labs, device, EPOCHS, LR)

    # ── Train Qwen3-VL ────────────────────────────────────────────────────
    torch.manual_seed(42)  # same init seed
    qwen = Qwen3VLClassifier(N_CLASSES, vit_dim=128, proj_dim=128, depth=4, num_heads=4)
    qw_hist, qw_final, qw_best = train_model(qwen, "Qwen3-VL", loader, eval_vids, eval_labs, device, EPOCHS, LR)

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n\n{DIVIDER}")
    print(f"  FINAL COMPARISON — VL-JEPA vs Qwen3-VL (both trained)")
    print(DIVIDER)

    def row(label, v1, v2, w=30):
        w1 = "◀ WINS" if isinstance(v1, float) and isinstance(v2, float) and v1 > v2 + 0.001 else ""
        w2 = "◀ WINS" if isinstance(v1, float) and isinstance(v2, float) and v2 > v1 + 0.001 else ""
        if isinstance(v1, float): v1s = f"{v1*100:.2f}%"
        else: v1s = str(v1)
        if isinstance(v2, float): v2s = f"{v2*100:.2f}%"
        else: v2s = str(v2)
        print(f"  {label:<{w}} {v1s:>12} {w1:<8} {v2s:>12} {w2}")

    print(f"\n  {'Metric':<30} {'VL-JEPA':>12} {'':8} {'Qwen3-VL':>12}")
    print(f"  {'─'*30} {'─'*12} {'':8} {'─'*12}")

    metrics = [
        ("Top-1 Accuracy", "accuracy"),
        ("Top-5 Accuracy", "top5_acc"),
        ("Precision (macro)", "precision_macro"),
        ("Precision (weighted)", "precision_weighted"),
        ("Recall (macro)", "recall_macro"),
        ("Recall (weighted)", "recall_weighted"),
        ("F1 Score (macro)", "f1_macro"),
        ("F1 Score (weighted)", "f1_weighted"),
        ("MCC", "mcc"),
        ("Cohen's Kappa", "kappa"),
        ("ROC-AUC (macro)", "roc_auc"),
    ]

    for label, key in metrics:
        row(label, vj_final[key], qw_final[key])

    # Parameters + speed
    vj_params = sum(p.numel() for p in vljepa.parameters())
    qw_params = sum(p.numel() for p in qwen.parameters())
    print(f"\n  {'Parameters':<30} {vj_params/1e6:>11.2f}M {'':8} {qw_params/1e6:>11.2f}M")

    vj_time = sum(h.get("loss", 0) for h in vj_hist)  # proxy
    print(f"  {'Best epoch accuracy':<30} {vj_best*100:>11.1f}% {'':8} {qw_best*100:>11.1f}%")

    # Per-class F1
    print(f"\n  PER-CLASS F1:")
    vj_f1 = f1_score(eval_labs.numpy(), F.softmax(torch.cat([vljepa(eval_vids[i:i+64].to(device)).detach().cpu() for i in range(0, N_EVAL, 64)]), dim=-1).numpy().argmax(1), average=None, zero_division=0)
    qw_f1 = f1_score(eval_labs.numpy(), F.softmax(torch.cat([qwen(eval_vids[i:i+64].to(device)).detach().cpu() for i in range(0, N_EVAL, 64)]), dim=-1).numpy().argmax(1), average=None, zero_division=0)
    for i, cls in enumerate(ACTION_CLASSES):
        vw = "◀" if vj_f1[i] > qw_f1[i] + 0.01 else ""
        qw2 = "◀" if qw_f1[i] > vj_f1[i] + 0.01 else ""
        print(f"    {cls:<20} {vj_f1[i]*100:6.1f}% {vw:<3}  {qw_f1[i]*100:6.1f}% {qw2}")

    # Confusion matrices
    for name, met in [("VL-JEPA", vj_final), ("Qwen3-VL", qw_final)]:
        print(f"\n  Confusion Matrix ({name}):")
        cm = met["cm"]
        short = [c[:5] for c in ACTION_CLASSES]
        print("         " + " ".join(f"{s:>5}" for s in short))
        for i in range(N_CLASSES):
            vals = " ".join(f"{int(cm[i,j]):>5}" for j in range(N_CLASSES))
            print(f"  {short[i]:<5}  {vals}")
        print(f"  Correct: {sum(cm[i][i] for i in range(N_CLASSES))}/{N_EVAL}")

    # Classification reports
    for name, met in [("VL-JEPA", vj_final), ("Qwen3-VL", qw_final)]:
        print(f"\n  Full Classification Report ({name}):")
        print(met["report"])

    # Training curves
    print(f"\n{DIVIDER}")
    print(f"  TRAINING CURVES")
    print(DIVIDER)
    print(f"  {'Epoch':>5}  {'VL-JEPA Acc':>12}  {'Qwen3-VL Acc':>12}")
    for i in range(len(vj_hist)):
        va = vj_hist[i]["accuracy"]
        qa = qw_hist[i]["accuracy"]
        vbar = "█" * int(va * 20)
        qbar = "█" * int(qa * 20)
        print(f"  {i+1:5d}  {va*100:11.1f}% |{vbar:<20}|  {qa*100:11.1f}% |{qbar:<20}|")

    # Winner
    print(f"\n{DIVIDER}")
    if vj_final["accuracy"] > qw_final["accuracy"]:
        winner = "VL-JEPA"
        delta = vj_final["accuracy"] - qw_final["accuracy"]
    else:
        winner = "Qwen3-VL"
        delta = qw_final["accuracy"] - vj_final["accuracy"]
    print(f"  WINNER: {winner} (by {delta*100:.1f}% accuracy)")
    print(DIVIDER)

    # Save
    import json
    report = {
        "vljepa": {k: v for k, v in vj_final.items() if k not in ("cm", "report")},
        "qwen3vl": {k: v for k, v in qw_final.items() if k not in ("cm", "report")},
        "vljepa_history": vj_hist,
        "qwen3vl_history": qw_hist,
    }
    out = Path(__file__).parent / "head_to_head_results.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()

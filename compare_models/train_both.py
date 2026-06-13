"""
VL-JEPA vs Qwen3-VL — Configurable Training & Comparison
==========================================================
Run:  python compare_models/train_both.py

Configure everything in the CONFIG section below.
"""

import sys, time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    confusion_matrix, classification_report, top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from vl_jepa.model import XEncoder

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         CONFIG — EDIT HERE                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ── Data ──────────────────────────────────────────────────────────────────
N_TRAIN       = 1000     # Number of training videos (try: 500, 1000, 2000, 5000)
N_EVAL        = 300      # Number of evaluation videos
VIDEO_MINUTES = 0.1      # Length of each video in MINUTES (0.1 = 6 seconds)
                         #   0.1  min =   6 sec  →   4 frames
                         #   0.5  min =  30 sec  →  15 frames
                         #   1.0  min =  60 sec  →  30 frames
                         #   2.0  min = 120 sec  →  60 frames
                         #   5.0  min = 300 sec  → 150 frames
FPS           = 0.67     # Frames per second to sample (0.67 ≈ 1 frame every 1.5 sec)
                         # Higher FPS = more frames per video = slower but more detail
NOISE_STD     = 0.10     # Random noise added to video (higher = harder task)

# ── Model ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 64       # Image resolution (64=fast, 128=better, 224=slow but realistic)
MODEL_DIM     = 128      # Hidden dimension (128=small/fast, 256=medium, 512=large)
MODEL_DEPTH   = 4        # Transformer depth (4=fast, 8=better, 12=large)
NUM_HEADS     = 4        # Attention heads

# ── Training ──────────────────────────────────────────────────────────────
EPOCHS        = 50       # Training epochs (more = better but slower)
BATCH_SIZE    = 32       # Batch size (32-64 recommended)
LEARNING_RATE = 3e-3     # Learning rate
WEIGHT_DECAY  = 0.05     # Regularization
WARMUP_EPOCHS = 3        # LR warmup period

# ── Classes ───────────────────────────────────────────────────────────────
ACTION_CLASSES = [
    "stir soup", "chop vegetables", "mix batter", "roll dough", "pour liquid",
    "crack eggs", "slice bread", "wash hands", "peel potato", "grate cheese",
]

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    COMPUTED FROM CONFIG (don't edit)                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

NUM_FRAMES = max(2, int(VIDEO_MINUTES * 60 * FPS))  # frames per video
N_CLASSES  = len(ACTION_CLASSES)

# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

class VLJepaClassifier(nn.Module):
    """VL-JEPA: spatiotemporal ViT (all frames at once) → pool → classify."""
    def __init__(self):
        super().__init__()
        self.encoder = XEncoder(
            num_frames=NUM_FRAMES, img_size=IMG_SIZE,
            dim=MODEL_DIM, depth=MODEL_DEPTH, num_heads=NUM_HEADS,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM), nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(MODEL_DIM, N_CLASSES),
        )
    def forward(self, video):
        return self.head(self.encoder(video).mean(dim=1))


class Qwen3VLClassifier(nn.Module):
    """Qwen3-VL: per-frame ViT → MLP projector → temporal pool → classify."""
    def __init__(self):
        super().__init__()
        n_patches = (IMG_SIZE // 16) ** 2
        self.patch_embed = nn.Conv2d(3, MODEL_DIM, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, MODEL_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, MODEL_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=MODEL_DIM, nhead=NUM_HEADS,
                dim_feedforward=MODEL_DIM * 4, batch_first=True, norm_first=True,
            ) for _ in range(MODEL_DEPTH)
        ])
        self.norm = nn.LayerNorm(MODEL_DIM)
        self.projector = nn.Sequential(
            nn.Linear(MODEL_DIM, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM), nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(MODEL_DIM, N_CLASSES),
        )

    def forward(self, video):
        B, C, T, H, W = video.shape
        feats = []
        for t in range(T):
            x = self.patch_embed(video[:, :, t]).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            x = x + self.pos_embed[:, :x.shape[1], :]
            for blk in self.blocks:
                x = blk(x)
            feats.append(self.norm(x[:, 0]))
        pooled = torch.stack(feats, dim=1).mean(dim=1)
        return self.head(self.projector(pooled))


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class ActionVideoDataset(Dataset):
    COLORS = torch.tensor([
        [0.9,0.1,0.1],[0.1,0.8,0.1],[0.9,0.9,0.2],[0.7,0.5,0.3],[0.1,0.2,0.9],
        [0.95,0.95,0.95],[0.5,0.25,0.1],[0.1,0.9,0.9],[0.6,0.4,0.15],[1.0,0.85,0.1],
    ])

    def __init__(self, n, noise=NOISE_STD):
        self.n, self.noise = n, noise
        self.labels = torch.randint(0, N_CLASSES, (n,))

    def __len__(self):
        return self.n

    def _pat(self, cls, t):
        S = IMG_SIZE
        y = torch.linspace(0,1,S).unsqueeze(1).expand(S,S)
        x = torch.linspace(0,1,S).unsqueeze(0).expand(S,S)
        ph = t / max(NUM_FRAMES, 1)
        c = self.COLORS[cls % len(self.COLORS)]
        patterns = [
            lambda: (((x-0.5-0.2*math.cos(2*math.pi*ph))**2+(y-0.5-0.2*math.sin(2*math.pi*ph))**2).sqrt()<0.25).float(),
            lambda: (torch.sin(10*x*2*math.pi+ph*6)>0).float(),
            lambda: ((torch.sin(6*x*2*math.pi)>0)^(torch.sin(6*y*2*math.pi)>0)).float(),
            lambda: (torch.sin(8*y*2*math.pi+ph*8)>0).float(),
            lambda: (((x-0.5)**2+(y-ph)**2).sqrt()<0.2).float(),
            lambda: (((x-0.5)**2+(y-0.5)**2).sqrt()<(0.2+0.1*ph)).float(),
            lambda: (torch.sin(8*(x+y)*2*math.pi+ph*4)>0).float(),
            lambda: (((x*5%1-0.5)**2+(y*5%1-0.5)**2).sqrt()<0.15).float(),
            lambda: (torch.sin(torch.atan2(y-0.5,x-0.5)*3+((x-0.5)**2+(y-0.5)**2).sqrt()*15+ph*5)>0).float(),
            lambda: x,
        ]
        p = patterns[cls % len(patterns)]()
        frame = torch.zeros(3, S, S)
        for ch in range(3):
            frame[ch] = p * c[ch] + (1-p) * 0.05
        return frame

    def __getitem__(self, idx):
        cls = self.labels[idx].item()
        frames = [(self._pat(cls, t) + torch.randn(3,IMG_SIZE,IMG_SIZE)*self.noise).clamp(0,1) for t in range(NUM_FRAMES)]
        return {"video": torch.stack(frames, dim=1), "label": cls}


def collate(batch):
    return {"video": torch.stack([b["video"] for b in batch]),
            "label": torch.tensor([b["label"] for b in batch])}


# ═══════════════════════════════════════════════════════════════════════════
# Train + Evaluate
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, vids, labs, device):
    model.eval()
    logits = []
    for i in range(0, len(vids), 64):
        with torch.no_grad():
            logits.append(model(vids[i:i+64].to(device)).cpu())
    probs = F.softmax(torch.cat(logits), dim=-1).numpy()
    preds = probs.argmax(1)
    labs = labs.numpy()
    labs_bin = label_binarize(labs, classes=list(range(N_CLASSES)))
    m = {
        "accuracy": accuracy_score(labs, preds),
        "precision": precision_score(labs, preds, average="macro", zero_division=0),
        "recall": recall_score(labs, preds, average="macro", zero_division=0),
        "f1": f1_score(labs, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labs, preds, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(labs, preds),
        "kappa": cohen_kappa_score(labs, preds),
    }
    try: m["auc"] = roc_auc_score(labs_bin, probs, multi_class="ovr", average="macro")
    except: m["auc"] = 0
    try: m["top5"] = top_k_accuracy_score(labs, probs, k=min(5,N_CLASSES), labels=list(range(N_CLASSES)))
    except: m["top5"] = 0
    m["cm"] = confusion_matrix(labs, preds)
    m["report"] = classification_report(labs, preds, target_names=ACTION_CLASSES, zero_division=0)
    return m


def train(model, name, loader, eval_v, eval_l, device):
    print(f"\n{'─'*70}")
    print(f"  Training: {name} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
    print(f"{'─'*70}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep:
        (ep+1)/WARMUP_EPOCHS if ep < WARMUP_EPOCHS else
        0.5*(1+math.cos(math.pi*(ep-WARMUP_EPOCHS)/max(EPOCHS-WARMUP_EPOCHS,1))))

    hist, best_acc, best_w = [], 0, None
    print(f"  {'Ep':>3} {'Loss':>7} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7} {'Time':>5}")

    for ep in range(1, EPOCHS+1):
        model.train()
        total_loss, n = 0, 0
        t0 = time.time()
        for b in loader:
            logits = model(b["video"].to(device))
            loss = loss_fn(logits, b["label"].to(device))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total_loss += loss.item(); n += 1
        sched.step()
        m = evaluate(model, eval_v, eval_l, device)
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_w = {k: v.clone() for k, v in model.state_dict().items()}
        hist.append({"epoch": ep, "loss": total_loss/n, **{k:v for k,v in m.items() if k not in ("cm","report")}})
        bar = "█" * int(m["accuracy"] * 30)
        print(f"  {ep:3d} {total_loss/n:7.4f} {m['accuracy']*100:6.1f}% {m['f1']*100:6.1f}% "
              f"{m['precision']*100:6.1f}% {m['recall']*100:6.1f}% {m['auc']*100:6.1f}% "
              f"{time.time()-t0:4.0f}s |{bar}|")

    if best_w: model.load_state_dict(best_w)
    final = evaluate(model, eval_v, eval_l, device)
    return hist, final, best_acc


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)

    print(f"\n{'='*70}")
    print(f"  VL-JEPA vs Qwen3-VL — Training & Comparison")
    print(f"{'='*70}")
    print(f"  Device:          {device}")
    print(f"  Training videos: {N_TRAIN}")
    print(f"  Eval videos:     {N_EVAL}")
    print(f"  Video length:    {VIDEO_MINUTES} min ({VIDEO_MINUTES*60:.0f} sec)")
    print(f"  Frames/video:    {NUM_FRAMES} (at {FPS} FPS)")
    print(f"  Resolution:      {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Classes:         {N_CLASSES}")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Model dim:       {MODEL_DIM}")
    print(f"  Model depth:     {MODEL_DEPTH}")

    # Data
    train_ds = ActionVideoDataset(N_TRAIN)
    eval_ds  = ActionVideoDataset(N_EVAL, noise=0.08)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)
    eval_v = torch.stack([eval_ds[i]["video"] for i in range(N_EVAL)])
    eval_l = eval_ds.labels

    # Train both
    vj_h, vj_f, vj_best = train(VLJepaClassifier(), "VL-JEPA", loader, eval_v, eval_l, device)
    torch.manual_seed(42)
    qw_h, qw_f, qw_best = train(Qwen3VLClassifier(), "Qwen3-VL", loader, eval_v, eval_l, device)

    # ── Final Comparison ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<28} {'VL-JEPA':>10} {'Qwen3-VL':>10} {'Winner':>10}")
    print(f"  {'─'*58}")
    for label, k in [("Top-1 Accuracy","accuracy"),("Top-5 Accuracy","top5"),
                      ("Precision (macro)","precision"),("Recall (macro)","recall"),
                      ("F1 Score (macro)","f1"),("F1 Score (weighted)","f1_weighted"),
                      ("MCC","mcc"),("Cohen's Kappa","kappa"),("ROC-AUC","auc")]:
        v, q = vj_f[k], qw_f[k]
        w = "VL-JEPA" if v > q+0.001 else "Qwen3-VL" if q > v+0.001 else "TIE"
        print(f"  {label:<28} {v*100:>9.2f}% {q*100:>9.2f}% {w:>10}")

    # Classification reports
    for name, f in [("VL-JEPA", vj_f), ("Qwen3-VL", qw_f)]:
        print(f"\n  {name} Classification Report:")
        print(f["report"])

    # Training curves
    print(f"\n{'='*70}")
    print(f"  TRAINING CURVES")
    print(f"{'='*70}")
    for i in range(len(vj_h)):
        va, qa = vj_h[i]["accuracy"], qw_h[i]["accuracy"]
        print(f"  Ep {i+1:3d}  VL-JEPA: {va*100:6.1f}%  Qwen3-VL: {qa*100:6.1f}%")

    # Save
    out = Path(__file__).parent / "train_both_results.json"
    json.dump({"vljepa": {k:v for k,v in vj_f.items() if k not in ("cm","report")},
               "qwen3vl": {k:v for k,v in qw_f.items() if k not in ("cm","report")},
               "config": {"n_train":N_TRAIN,"n_eval":N_EVAL,"video_minutes":VIDEO_MINUTES,
                          "fps":FPS,"num_frames":NUM_FRAMES,"img_size":IMG_SIZE,
                          "epochs":EPOCHS,"batch_size":BATCH_SIZE}},
              open(out,"w"), indent=2, default=str)
    print(f"\n  Results saved: {out}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

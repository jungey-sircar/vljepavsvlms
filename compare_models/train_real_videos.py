"""
VL-JEPA vs Qwen3-VL — Train on REAL Videos from Action100M
=============================================================
Run:  python compare_models/train_real_videos.py

This script:
  1. Loads 100 real Action100M videos from HuggingFace via FiftyOne
  2. Extracts frames + action labels from the GPT annotations
  3. Trains both VL-JEPA and Qwen3-VL classifiers on real video data
  4. Compares precision/recall/F1/accuracy side by side
"""

import sys, time, math, json, os
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
import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from vl_jepa.model import XEncoder

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          CONFIG                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

MAX_SAMPLES   = 1144      # Load ALL available videos from FiftyOne
NUM_FRAMES    = 4         # Frames to sample per video
IMG_SIZE      = 64        # Resize frames to this (64=fast, 128=better)
MODEL_DIM     = 128       # Model hidden dimension
MODEL_DEPTH   = 4         # Transformer depth
NUM_HEADS     = 4         # Attention heads
EPOCHS        = 30        # Training epochs
BATCH_SIZE    = 8         # Batch size (small for small datasets)
LEARNING_RATE = 1e-3      # Learning rate
MIN_SAMPLES_PER_CLASS = 3 # Need at least this many videos per class
TRAIN_SPLIT   = 0.8       # 80% train, 20% eval

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Load videos from FiftyOne / HuggingFace
# ═══════════════════════════════════════════════════════════════════════════

def load_action100m_dataset():
    """Load real Action100M videos from HuggingFace via FiftyOne."""
    print("  Loading Action100M..."); sys.stdout.flush()
    import fiftyone as fo

    # Try to load existing dataset first
    try:
        dataset = fo.load_dataset("Voxel51/action100m_tiny_subset")
        print(f"  Found existing dataset: {len(dataset)} videos")
        sys.stdout.flush()
        return dataset
    except ValueError:
        pass

    # Download from HuggingFace
    print("  Downloading from HuggingFace (first time only)..."); sys.stdout.flush()
    from fiftyone.utils.huggingface import load_from_hub
    dataset = load_from_hub(
        "Voxel51/action100m_tiny_subset",
        max_samples=MAX_SAMPLES,
    )
    print(f"  Loaded {len(dataset)} videos"); sys.stdout.flush()
    return dataset


# Action word → broad category mapping
ACTION_CATEGORIES = {
    "cook": ["cook", "fry", "bake", "roast", "boil", "simmer", "sauté", "grill", "heat"],
    "cut": ["cut", "chop", "slice", "dice", "trim", "peel", "shred"],
    "mix": ["mix", "stir", "blend", "whisk", "fold", "combine", "beat"],
    "pour": ["pour", "add", "drizzle", "spray", "squeeze", "drip"],
    "prepare": ["prepare", "place", "arrange", "set", "put", "lay", "position"],
    "wash": ["wash", "rinse", "clean", "wipe", "soak", "scrub"],
    "show": ["show", "display", "present", "demonstrate", "reveal"],
    "craft": ["make", "build", "create", "assemble", "install", "attach", "sew"],
    "move": ["move", "pick", "grab", "hold", "lift", "carry", "pull", "push", "open", "close"],
    "talk": ["talk", "speak", "explain", "describe", "say", "discuss", "introduce"],
}

def categorize_action(label):
    """Map a specific action label to a broad category."""
    label_lower = label.lower()
    for category, keywords in ACTION_CATEGORIES.items():
        for kw in keywords:
            if kw in label_lower:
                return category
    return None  # uncategorizable


def extract_labels_from_dataset(dataset):
    """
    Extract action labels from FiftyOne dataset.
    Groups specific actions into broad categories for enough samples per class.
    """
    from collections import Counter
    videos = []
    raw_labels = []
    categories = []

    for sample in dataset:
        filepath = sample.filepath

        # Get action label from GPT annotations
        label = None
        for field in ['gpt_action_brief', 'gpt_summary_brief', 'gpt_action_detailed']:
            if hasattr(sample, field):
                val = getattr(sample, field)
                if val is not None and hasattr(val, 'detections') and len(val.detections) > 0:
                    label = val.detections[0].label
                    break

        if not label or not os.path.exists(filepath):
            continue

        cat = categorize_action(label)
        if cat:
            videos.append(filepath)
            raw_labels.append(label)
            categories.append(cat)

    # Keep only categories with enough samples
    cat_counts = Counter(categories)
    valid_cats = [c for c, n in cat_counts.most_common() if n >= MIN_SAMPLES_PER_CLASS]

    if len(valid_cats) < 2:
        print(f"  WARNING: Only {len(valid_cats)} categories with >= {MIN_SAMPLES_PER_CLASS} samples.")
        print(f"  Category distribution: {dict(cat_counts.most_common())}")
        # Fall back: use ALL categories with at least 1 sample
        valid_cats = [c for c, n in cat_counts.most_common() if n >= 1]

    # Filter
    filt_videos, filt_cats = [], []
    for v, c in zip(videos, categories):
        if c in valid_cats:
            filt_videos.append(v)
            filt_cats.append(c)

    # Create label mapping
    class_names = sorted(set(filt_cats))
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    label_indices = [label_to_idx[c] for c in filt_cats]

    print(f"  Found {len(filt_videos)} videos in {len(class_names)} action categories:")
    for name in class_names:
        count = sum(1 for c in filt_cats if c == name)
        print(f"    [{label_to_idx[name]}] {name}: {count} videos")

    return filt_videos, label_indices, class_names


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Video loading (frame extraction)
# ═══════════════════════════════════════════════════════════════════════════

class RealVideoDataset(Dataset):
    """Load real video files using OpenCV, sample frames, resize."""

    def __init__(self, video_paths, labels, num_frames=4, img_size=64):
        self.paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def _load_video(self, path):
        """Load video with OpenCV and sample num_frames evenly spaced frames."""
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                cap.release()
                return None

            # Pick evenly-spaced frame indices
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                # BGR→RGB, resize, normalize to [0,1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [C,H,W]
                frames.append(frame)

            cap.release()
            video = torch.stack(frames, dim=1)  # [C, T, H, W]
            return video

        except Exception as e:
            print(f"    Warning: Could not load {Path(path).name}: {e}")
            return None

    def __getitem__(self, idx):
        video = self._load_video(self.paths[idx])
        if video is None:
            # Return a blank video if loading fails
            video = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
        return {"video": video, "label": self.labels[idx]}


def collate(batch):
    return {"video": torch.stack([b["video"] for b in batch]),
            "label": torch.tensor([b["label"] for b in batch])}


# ═══════════════════════════════════════════════════════════════════════════
# Models (same as train_both.py)
# ═══════════════════════════════════════════════════════════════════════════

class VLJepaClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = XEncoder(num_frames=NUM_FRAMES, img_size=IMG_SIZE,
                                dim=MODEL_DIM, depth=MODEL_DEPTH, num_heads=NUM_HEADS)
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM), nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(MODEL_DIM, n_classes))

    def forward(self, video):
        return self.head(self.encoder(video).mean(dim=1))


class Qwen3VLClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        n_patches = (IMG_SIZE // 16) ** 2
        self.patch_embed = nn.Conv2d(3, MODEL_DIM, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, MODEL_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+n_patches, MODEL_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=MODEL_DIM, nhead=NUM_HEADS,
                dim_feedforward=MODEL_DIM*4, batch_first=True, norm_first=True)
            for _ in range(MODEL_DEPTH)])
        self.norm = nn.LayerNorm(MODEL_DIM)
        self.projector = nn.Sequential(nn.Linear(MODEL_DIM, MODEL_DIM), nn.GELU(),
                                        nn.Linear(MODEL_DIM, MODEL_DIM))
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM), nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(MODEL_DIM, n_classes))

    def forward(self, video):
        B, C, T, H, W = video.shape
        feats = []
        for t in range(T):
            x = self.patch_embed(video[:, :, t]).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            x = x + self.pos_embed[:, :x.shape[1], :]
            for blk in self.blocks: x = blk(x)
            feats.append(self.norm(x[:, 0]))
        return self.head(self.projector(torch.stack(feats, 1).mean(1)))


# ═══════════════════════════════════════════════════════════════════════════
# Train + Eval
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, eval_loader, n_classes, device):
    """Evaluate using DataLoader to avoid loading everything into RAM."""
    model.eval()
    all_logits, all_labs = [], []
    for b in eval_loader:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            all_logits.append(model(b["video"].to(device)).float().cpu())
            all_labs.append(b["label"])
    probs = F.softmax(torch.cat(all_logits), dim=-1).numpy()
    preds = probs.argmax(1)
    labs = torch.cat(all_labs).numpy()
    labs_bin = label_binarize(labs, classes=list(range(n_classes)))
    m = {
        "accuracy": accuracy_score(labs, preds),
        "precision": precision_score(labs, preds, average="macro", zero_division=0),
        "recall": recall_score(labs, preds, average="macro", zero_division=0),
        "f1": f1_score(labs, preds, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labs, preds),
        "kappa": cohen_kappa_score(labs, preds),
    }
    try: m["auc"] = roc_auc_score(labs_bin, probs, multi_class="ovr", average="macro")
    except: m["auc"] = 0
    m["report"] = classification_report(labs, preds, zero_division=0)
    return m


def train_model(model, name, train_loader, eval_loader, n_classes, device):
    print(f"\n{'─'*70}"); sys.stdout.flush()
    print(f"  Training: {name} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
    print(f"{'─'*70}"); sys.stdout.flush()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))  # Mixed precision
    use_amp = (device.type == 'cuda')

    best_acc, best_w = 0, None
    print(f"  {'Ep':>3} {'Loss':>7} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Time':>5}")
    sys.stdout.flush()

    if len(train_loader) == 0:
        print(f"  ERROR: No batches in train loader"); sys.stdout.flush()
        dummy = evaluate(model, eval_loader, n_classes, device)
        return dummy, 0.0

    for ep in range(1, EPOCHS+1):
        model.train()
        total_loss, n = 0, 0
        t0 = time.time()
        for b in train_loader:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(b["video"].to(device))
                loss = loss_fn(logits, b["label"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total_loss += loss.item(); n += 1
        sched.step()
        m = evaluate(model, eval_loader, n_classes, device)
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_w = {k: v.clone() for k, v in model.state_dict().items()}
        bar = "█" * int(m["accuracy"] * 30)
        print(f"  {ep:3d} {total_loss/n:7.4f} {m['accuracy']*100:6.1f}% {m['f1']*100:6.1f}% "
              f"{m['precision']*100:6.1f}% {m['recall']*100:6.1f}% {time.time()-t0:4.0f}s |{bar}|")
        sys.stdout.flush()

    if best_w: model.load_state_dict(best_w)
    final = evaluate(model, eval_loader, n_classes, device)
    return final, best_acc


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)

    # GPU optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        torch.cuda.manual_seed(42)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_name, gpu_mem = 'N/A', 0

    print(f"\n{'='*70}")
    print(f"  Real Video Training — Action100M from HuggingFace")
    print(f"  Device: {device} | GPU: {gpu_name} ({gpu_mem:.1f} GB)" if device.type == 'cuda'
          else f"  Device: {device} (⚠ No CUDA — install pytorch+cu124 for 10x speedup)")
    print(f"  Mixed precision: {'ON' if device.type == 'cuda' else 'OFF'} | Max samples: {MAX_SAMPLES}")
    print(f"{'='*70}"); sys.stdout.flush()

    # Load dataset
    dataset = load_action100m_dataset()
    video_paths, label_indices, class_names = extract_labels_from_dataset(dataset)
    n_classes = len(class_names)

    if len(video_paths) < 10:
        print(f"\n  ERROR: Only found {len(video_paths)} videos with labels.")
        print(f"  Need at least 10 videos. Try increasing MAX_SAMPLES.")
        return

    # Split train/eval
    n_total = len(video_paths)
    n_train = int(n_total * TRAIN_SPLIT)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    eval_idx = indices[n_train:]

    train_paths = [video_paths[i] for i in train_idx]
    train_labels = [label_indices[i] for i in train_idx]
    eval_paths = [video_paths[i] for i in eval_idx]
    eval_labels = [label_indices[i] for i in eval_idx]

    print(f"\n  Train: {len(train_paths)} videos | Eval: {len(eval_paths)} videos")
    print(f"  Classes: {n_classes} | Frames/video: {NUM_FRAMES} | Resolution: {IMG_SIZE}x{IMG_SIZE}")

    # Create datasets — videos loaded lazily via DataLoader (no pre-loading)
    print(f"\n  Creating datasets (videos loaded lazily during training)...")
    sys.stdout.flush()
    train_ds = RealVideoDataset(train_paths, train_labels, NUM_FRAMES, IMG_SIZE)
    eval_ds  = RealVideoDataset(eval_paths, eval_labels, NUM_FRAMES, IMG_SIZE)
    actual_bs = min(BATCH_SIZE, max(1, len(train_paths) // 2))
    pin = (device.type == 'cuda')  # Pin memory for faster CPU→GPU transfer
    train_loader = DataLoader(train_ds, batch_size=actual_bs, shuffle=True,
                              num_workers=0, collate_fn=collate, drop_last=False,
                              pin_memory=pin)
    eval_loader = DataLoader(eval_ds, batch_size=actual_bs, shuffle=False,
                             num_workers=0, collate_fn=collate, drop_last=False,
                             pin_memory=pin)
    print(f"  Batch size: {actual_bs} | Train batches: {len(train_loader)} | Eval batches: {len(eval_loader)}")
    print(f"  Pin memory: {pin}")
    sys.stdout.flush()

    # Train both
    vj_final, vj_best = train_model(VLJepaClassifier(n_classes), "VL-JEPA",
                                      train_loader, eval_loader, n_classes, device)

    # Free GPU memory before training second model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"\n  GPU memory freed. Used: {torch.cuda.memory_allocated()/1e6:.0f} MB")
        sys.stdout.flush()

    torch.manual_seed(42)
    qw_final, qw_best = train_model(Qwen3VLClassifier(n_classes), "Qwen3-VL",
                                      train_loader, eval_loader, n_classes, device)

    # Final comparison
    print(f"\n\n{'='*70}")
    print(f"  FINAL COMPARISON (Real Action100M Videos)")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<25} {'VL-JEPA':>10} {'Qwen3-VL':>10}")
    print(f"  {'─'*45}")
    for label, k in [("Accuracy","accuracy"),("Precision","precision"),
                      ("Recall","recall"),("F1 (macro)","f1"),
                      ("MCC","mcc"),("Kappa","kappa"),("ROC-AUC","auc")]:
        v, q = vj_final.get(k, 0), qw_final.get(k, 0)
        w = " ◀" if v > q + 0.01 else ""
        w2 = " ◀" if q > v + 0.01 else ""
        print(f"  {label:<25} {v*100:>9.2f}%{w}  {q*100:>9.2f}%{w2}")

    print(f"\n  VL-JEPA best: {vj_best*100:.1f}% | Qwen3-VL best: {qw_best*100:.1f}%")

    for name, f in [("VL-JEPA", vj_final), ("Qwen3-VL", qw_final)]:
        print(f"\n  {name} Report:\n{f['report']}")

    # Save
    out = Path(__file__).parent / "real_video_results.json"
    json.dump({"vljepa": {k:v for k,v in vj_final.items() if k != "report"},
               "qwen3vl": {k:v for k,v in qw_final.items() if k != "report"},
               "classes": class_names, "n_train": len(train_paths), "n_eval": len(eval_paths)},
              open(out, "w"), indent=2, default=str)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()

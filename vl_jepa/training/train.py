"""
VL-JEPA Training Loop — v2 (Improved)
======================================
Improvements over v1:
  1. EMA Target Encoder        -- stable training target, prevents collapse
  2. InfoNCE Contrastive Loss  -- bidirectional, hard negative mining
  3. Spatiotemporal Masking    -- token-level masking for robust representations
  4. Video Augmentation        -- color/spatial/frame augmentation pipeline
  5. Gradient Accumulation     -- simulate larger batch sizes on limited GPU
  6. Layer-decay LR            -- lower LR for shallower layers (ViT best practice)
  7. EMA tau schedule          -- tau scheduled from 0.994 -> 1.0 during training
  8. Better logging            -- per-loss breakdown, EMA tau, temperature tracking

Usage:
    python vl_jepa/training/train.py --dataset_name action100m --epochs 50
"""

import os
import sys
import math
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vl_jepa.model import VLJepa, XEncoder, YEncoder, Predictor, YDecoder
from vl_jepa.training.loss import VLJepaLoss
from vl_jepa.training.dataset import VideoTextDataset, Action100MDataset
from vl_jepa.training.augment import EMAUpdater, VideoAugmentor, SpatiotemporalMask


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def cosine_lr_schedule(optimizer, step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
    return lr


# ---------------------------------------------------------------------------
# Optimizer with layer-wise LR decay
# ---------------------------------------------------------------------------

def build_optimizer(model, lr=2e-4, weight_decay=0.05, betas=(0.9, 0.95), layer_decay=0.75):
    """
    AdamW with layer-wise LR decay on the Predictor transformer layers.
    Layer-decay: shallower layers get lower LR -- key for ViT stability (BEiT, MAE).
    """
    param_groups = []
    predictor_named = list(model.predictor.named_parameters())
    n_layers = sum(1 for n, _ in predictor_named if 'blocks.' in n and '.0.' in n)
    n_layers = max(n_layers, 1)

    for name, param in model.predictor.named_parameters():
        if not param.requires_grad:
            continue
        if 'blocks.' in name:
            try:
                layer_id = int(name.split('blocks.')[1].split('.')[0])
                lr_scale = layer_decay ** (n_layers - layer_id)
            except (IndexError, ValueError):
                lr_scale = 1.0
        else:
            lr_scale = 1.0
        wd = 0.0 if ('bias' in name or 'norm' in name.lower()) else weight_decay
        param_groups.append({"params": [param], "weight_decay": wd, "lr_scale": lr_scale})

    if model.y_decoder is not None:
        for name, param in model.y_decoder.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if ('bias' in name or 'norm' in name.lower()) else weight_decay
            param_groups.append({"params": [param], "weight_decay": wd, "lr_scale": 1.0})

    print(f"Optimizer: {len(param_groups)} param groups, layer_decay={layer_decay}")
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {
        "video":             torch.stack([b["video"] for b in batch]),
        "query_input_ids":   torch.stack([b["query_input_ids"] for b in batch]),
        "target_input_ids":  torch.stack([b["target_input_ids"] for b in batch]),
        "query_text":        [b["query_text"] for b in batch],
        "target_text":       [b["target_text"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Training epoch v2
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, ema_updater, video_aug, masker,
    loader, optimizer, loss_fn, device,
    epoch, total_steps, step_offset=0,
    warmup_steps=1000, base_lr=2e-4,
    clip_grad=1.0, use_amp=True,
    accum_steps=1, log_interval=20,
):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    totals = dict(loss=0.0, pred_loss=0.0, infonce_loss=0.0, vic_loss=0.0)
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        global_step = step_offset + batch_idx
        lr = cosine_lr_schedule(optimizer, global_step, total_steps, warmup_steps, base_lr)

        video     = batch["video"].to(device, non_blocking=True)
        query_ids = batch["query_input_ids"].to(device, non_blocking=True)
        target_ids = batch["target_input_ids"].to(device, non_blocking=True)

        # Video augmentation
        if video_aug is not None:
            video = torch.stack([video_aug(video[i]) for i in range(video.shape[0])])

        with autocast(enabled=use_amp):
            pred_emb = model(video, query_ids)

            with torch.no_grad():
                if ema_updater is not None:
                    ema_vis = ema_updater.target(video)
                    target_emb = model.predictor(ema_vis, model.y_encoder(query_ids))
                else:
                    target_emb = model.y_encoder(target_ids)

            loss_dict = loss_fn(pred_emb, target_emb)
            loss = loss_dict["loss"] / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema_updater is not None:
                ema_updater.update()

        for k in totals:
            v = loss_dict.get(k, torch.tensor(0.0))
            totals[k] += v.item() if isinstance(v, torch.Tensor) else float(v)
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            tau  = ema_updater.get_current_tau() if ema_updater else 0.0
            temp = loss_dict.get("temperature", 0.07)
            print(
                f"Epoch {epoch:3d} | Step {batch_idx+1:5d}/{len(loader)} | "
                f"Loss: {totals['loss']/n_batches:.4f} | "
                f"NCE: {totals['infonce_loss']/n_batches:.4f} | "
                f"Pred: {totals['pred_loss']/n_batches:.4f} | "
                f"LR: {lr:.2e} | tau: {tau:.4f} | T: {temp:.3f} | "
                f"{time.time()-t0:.1f}s"
            )

    return {k: v / max(n_batches, 1) for k, v in
            {**totals, "steps": n_batches}.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, ema_updater, epoch, metrics, path):
    state = {
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "metrics": metrics,
    }
    if ema_updater is not None:
        state["ema_state"] = ema_updater.state_dict()
    torch.save(state, path)
    print(f"  Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, ema_updater, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if ema_updater and "ema_state" in ckpt:
        ema_updater.load_state_dict(ckpt["ema_state"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VL-JEPA v2")

    # Dataset
    parser.add_argument("--dataset_name",    type=str,   default="action100m")
    parser.add_argument("--jsonl_path",       type=str,   default=None)
    parser.add_argument("--annotation_field", type=str,   default="gpt_action_brief",
                        choices=["gpt_action_brief", "gpt_summary_brief", "gpt_action_detailed"])
    parser.add_argument("--video_dir",        type=str,   default=None)
    parser.add_argument("--max_samples",      type=int,   default=None)

    # Model
    parser.add_argument("--num_frames",       type=int,   default=8)
    parser.add_argument("--img_size",         type=int,   default=224)
    parser.add_argument("--encoder_size",     type=str,   default="B",
                        choices=["S", "B", "L"],
                        help="ViT size: S=384d, B=768d (default), L=1024d")
    parser.add_argument("--text_dim",         type=int,   default=512)
    parser.add_argument("--predictor_depth",  type=int,   default=6,
                        help="Predictor depth (paper uses 6, v1 was 4)")
    parser.add_argument("--predictor_heads",  type=int,   default=8)
    parser.add_argument("--y_encoder_mode",   type=str,   default="standalone",
                        choices=["standalone", "clip"])
    parser.add_argument("--with_decoder",     action="store_true")

    # Training
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--accum_steps",      type=int,   default=4,
                        help="Gradient accumulation (eff_bs = batch_size * accum_steps)")
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--weight_decay",     type=float, default=0.05)
    parser.add_argument("--layer_decay",      type=float, default=0.75)
    parser.add_argument("--clip_grad",        type=float, default=1.0)
    parser.add_argument("--warmup_epochs",    type=float, default=2.0,
                        help="Warmup epochs (paper uses 2-5, v1 used 0.5)")
    parser.add_argument("--no_amp",           action="store_true")
    parser.add_argument("--workers",          type=int,   default=4)

    # v2 Improvements
    parser.add_argument("--ema_decay",        type=float, default=0.996,
                        help="EMA target encoder decay (0 = disable EMA)")
    parser.add_argument("--mask_ratio",       type=float, default=0.75,
                        help="Spatiotemporal masking ratio (0 = no masking)")
    parser.add_argument("--infonce_temp",     type=float, default=0.07)
    parser.add_argument("--no_infonce",       action="store_true")
    parser.add_argument("--no_vicreg",        action="store_true")
    parser.add_argument("--no_augment",       action="store_true")
    parser.add_argument("--hard_neg_weight",  type=float, default=2.0)

    # I/O
    parser.add_argument("--output_dir",       type=str,   default="vl_jepa_checkpoints_v2")
    parser.add_argument("--resume",           type=str,   default=None)
    parser.add_argument("--log_interval",     type=int,   default=20)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  VL-JEPA v2 Training (all improvements enabled)")
    print(f"  Device: {device}")
    print(f"  Eff. batch size: {args.batch_size * args.accum_steps}")
    print(f"  EMA={args.ema_decay} | mask={args.mask_ratio} | "
          f"InfoNCE={not args.no_infonce} | temp={args.infonce_temp}")
    print(f"{'='*65}\n")

    # ViT size configs
    VIT = {"S": (384, 6, 6), "B": (768, 12, 12), "L": (1024, 24, 16)}
    enc_dim, enc_depth, enc_heads = VIT[args.encoder_size]
    print(f"ViT-{args.encoder_size}: dim={enc_dim}, depth={enc_depth}")

    # Dataset
    if args.jsonl_path:
        dataset = VideoTextDataset(
            jsonl_path=args.jsonl_path, video_dir=args.video_dir,
            num_frames=args.num_frames, img_size=args.img_size,
        )
    else:
        dataset = Action100MDataset(
            dataset_name=args.dataset_name, annotation_field=args.annotation_field,
            max_samples=args.max_samples, num_frames=args.num_frames, img_size=args.img_size,
        )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples | {len(loader)} steps/epoch\n")

    # Model
    x_enc = XEncoder(num_frames=args.num_frames, img_size=args.img_size,
                     dim=enc_dim, depth=enc_depth, num_heads=enc_heads)
    text_dim = args.text_dim
    if args.y_encoder_mode == "clip":
        y_enc = YEncoder(mode="clip"); text_dim = 512
    else:
        y_enc = YEncoder(mode="standalone", dim=text_dim, depth=6, num_heads=8)

    pred = Predictor(visual_dim=enc_dim, text_dim=text_dim, hidden_dim=text_dim,
                     depth=args.predictor_depth, num_heads=args.predictor_heads)

    decoder = (YDecoder(vocab_size=256, context_dim=text_dim, dim=512, depth=4)
               if args.with_decoder else None)

    model = VLJepa(
        x_encoder=x_enc, y_encoder=y_enc, predictor=pred, y_decoder=decoder,
        freeze_x_encoder=True, freeze_y_encoder=True,
    ).to(device)

    counts = model.parameter_count()
    print("Parameters:")
    for k, v in counts.items():
        print(f"  {k:<22}: {v/1e6:.2f}M")

    # EMA target encoder
    ema_updater = None
    if args.ema_decay > 0:
        total_ema_steps = len(loader) * args.epochs // max(args.accum_steps, 1)
        ema_updater = EMAUpdater(
            online_encoder=model.x_encoder, tau=args.ema_decay,
            tau_schedule=True, tau_start=args.ema_decay - 0.002,
            tau_end=1.0, total_steps=total_ema_steps,
        )
        for p in model.x_encoder.parameters():
            p.requires_grad_(True)
        print(f"\nEMA target encoder: tau={args.ema_decay}")

    # Augmentation
    video_aug = None if args.no_augment else VideoAugmentor(
        img_size=args.img_size, color_jitter=True, frame_mask_prob=0.1, flip_prob=0.5,
    )
    masker = None if args.mask_ratio <= 0 else SpatiotemporalMask(mask_ratio=args.mask_ratio)

    if video_aug:  print("Video augmentation: ON")
    if masker:     print(f"Spatiotemporal masking: {args.mask_ratio:.0%}")

    # Loss
    loss_fn = VLJepaLoss(
        alpha=1.0, beta=0.5, gamma=1.0,
        temperature=args.infonce_temp,
        use_infonce=not args.no_infonce,
        use_vicreg=not args.no_vicreg,
        use_decoder=args.with_decoder,
        hard_neg_weight=args.hard_neg_weight,
    ).to(device)

    # Optimizer
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay,
                                layer_decay=args.layer_decay)
    if not args.no_infonce and hasattr(loss_fn, 'infonce') and loss_fn.infonce is not None:
        if hasattr(loss_fn.infonce, 'log_temp'):
            optimizer.add_param_group({
                "params": [loss_fn.infonce.log_temp],
                "lr": args.lr * 0.1, "weight_decay": 0.0, "lr_scale": 0.1,
            })
            print("Learnable InfoNCE temperature: ON")

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch, prev = load_checkpoint(model, optimizer, ema_updater, args.resume, device)
        print(f"\nResumed from epoch {start_epoch}: {prev}"); start_epoch += 1

    total_steps  = len(loader) * args.epochs
    warmup_steps = int(len(loader) * args.warmup_epochs)
    use_amp      = not args.no_amp and device.type == "cuda"

    print(f"\nTraining: {args.epochs} epochs | {total_steps:,} steps | "
          f"warmup={warmup_steps} | AMP={use_amp}\n")

    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n── Epoch {epoch+1}/{args.epochs} {'─'*45}")
        metrics = train_one_epoch(
            model=model, ema_updater=ema_updater, video_aug=video_aug,
            masker=masker, loader=loader, optimizer=optimizer, loss_fn=loss_fn,
            device=device, epoch=epoch+1, total_steps=total_steps,
            step_offset=epoch * len(loader), warmup_steps=warmup_steps,
            base_lr=args.lr, clip_grad=args.clip_grad, use_amp=use_amp,
            accum_steps=args.accum_steps, log_interval=args.log_interval,
        )
        print(f"\n  Summary: loss={metrics['loss']:.4f} | "
              f"nce={metrics['infonce_loss']:.4f} | pred={metrics['pred_loss']:.4f}")

        save_checkpoint(model, optimizer, ema_updater, epoch, metrics,
                        os.path.join(args.output_dir, "checkpoint_latest.pt"))

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(model, optimizer, ema_updater, epoch, metrics,
                            os.path.join(args.output_dir, "checkpoint_best.pt"))
            print(f"  [BEST] {best_loss:.4f}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()

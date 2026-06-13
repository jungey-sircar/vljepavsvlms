"""Quick verification of all VL-JEPA v2 improvements."""
import sys, torch
sys.path.insert(0, '.')
from vl_jepa.model import VLJepa
from vl_jepa.training.loss import VLJepaLoss, InfoNCELoss
from vl_jepa.training.augment import EMAUpdater, VideoAugmentor, SpatiotemporalMask
from vl_jepa.training.train import build_optimizer

torch.manual_seed(42)
print("=== Testing All v2 Improvements ===\n")

# 1. Model
model = VLJepa.build_default(mode='standalone', num_frames=4, with_decoder=True)
n = sum(p.numel() for p in model.parameters())
print(f"[1] Model built: {n//1_000_000:.1f}M params")

# 2. InfoNCE loss
loss_fn = VLJepaLoss(use_infonce=True, use_vicreg=True, temperature=0.07, hard_neg_weight=2.0)
pred   = torch.randn(8, 256)
target = torch.randn(8, 256)
out = loss_fn(pred, target)
print(f"[2] InfoNCE loss: {out['infonce_loss'].item():.4f}  temp={out['temperature']:.3f}")
print(f"    pred={out['pred_loss'].item():.4f}  vic={out['vic_loss'].item():.4f}  total={out['loss'].item():.4f}")

# 3. EMA
ema = EMAUpdater(model.x_encoder, tau=0.996, total_steps=1000)
video = torch.randn(1, 3, 4, 224, 224)
with torch.no_grad():
    tgt_tokens = ema.target(video)
ema.update()
print(f"[3] EMA: tau={ema.get_current_tau():.4f}  target_shape={list(tgt_tokens.shape)}")

# 4. VideoAugmentor
aug = VideoAugmentor(img_size=224)
clip = torch.rand(3, 4, 256, 256)
aug_clip = aug(clip)
print(f"[4] VideoAugmentor: {list(clip.shape)} -> {list(aug_clip.shape)}")

# 5. Spatiotemporal masking
masker = SpatiotemporalMask(mask_ratio=0.75)
tokens = torch.randn(1, 4*14*14, 384)
vis, mask = masker(tokens, T=4, H=14, W=14)
n_masked = int(mask.sum())
pct = n_masked / tokens.shape[1]
print(f"[5] Mask: {tokens.shape[1]} -> {vis.shape[1]} visible, {n_masked} masked ({pct:.0%})")

# 6. Layer-decay optimizer
opt = build_optimizer(model, lr=2e-4, layer_decay=0.75)
print(f"[6] Optimizer: {len(opt.param_groups)} param groups")

# 7. Full forward + loss — use B=2 for InfoNCE + VICReg
model.eval()
video2 = torch.randn(2, 3, 4, 224, 224)
with torch.no_grad():
    pred_emb = model(video2, torch.randint(0, 256, (2, 16)))
    tgt_emb  = model.y_encoder(torch.randint(0, 256, (2, 16)))
    loss_out = loss_fn(pred_emb, tgt_emb)
print(f"[7] Full forward: pred={list(pred_emb.shape)} loss={loss_out['loss'].item():.4f}  (nce={loss_out['infonce_loss'].item():.4f})")

print("\n=== All v2 improvements: OK ===")

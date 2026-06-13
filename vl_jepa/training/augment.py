"""
VL-JEPA Video Augmentation Pipeline
=====================================
Strong data augmentation is one of the highest-leverage improvements for
self-supervised video-language models. Key techniques from VL-JEPA / V-JEPA:

  1. Spatiotemporal Masking (tubelet masking)
     - Mask random contiguous tubes in (T, H, W) space
     - Forces the Predictor to learn robust semantic representations
     - Paper uses 70-90% masking ratio — much higher than image JEPA
     - Key: masking in embedding space (after X-Encoder), not pixel space

  2. Temporal Jitter
     - Sample frames at random intervals instead of uniform
     - Makes model robust to variable video speeds

  3. Color / Spatial Augmentations
     - Random crop, horizontal flip, color jitter, grayscale
     - Applied independently per clip view for contrastive training

  4. Multi-crop Strategy (from DINO/iBOT)
     - One large global crop (224x224) + multiple small local crops (96x96)
     - Model predicts global-crop embeddings from local-crop inputs
     - Enables learning at multiple scales

  5. Frame-level Masking
     - Randomly zero out entire frames (simulates occlusion)
     - Applied in pixel space before X-Encoder

All augmentations are implemented as PyTorch transforms compatible with
the existing VideoTextDataset pipeline.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# 1. Spatiotemporal Tubelet Masking (applied to token embeddings)
# ---------------------------------------------------------------------------

class SpatiotemporalMask(nn.Module):
    """
    Masks random contiguous tubes in the (T, H', W') token space.
    Applied AFTER X-Encoder to mask visual tokens before feeding to Predictor.

    This is the key difference from masked image modeling:
      - We mask LATENT representations, not pixels
      - The Predictor must PREDICT the masked embeddings from visible ones
      - High masking ratio (70-90%) forces semantic understanding

    Args:
        mask_ratio:     Fraction of tokens to mask (paper: 0.75-0.90)
        tube_size_t:    Temporal tube size (consecutive frames, default 2)
        tube_size_h:    Spatial height of tube (default 2 patches)
        tube_size_w:    Spatial width of tube (default 2 patches)
        mask_value:     Value to fill masked tokens (default 0.0)

    Usage:
        masker = SpatiotemporalMask(mask_ratio=0.75)
        tokens, mask = masker(visual_tokens, T=8, H=14, W=14)
        # tokens: [B, N_visible, D] — only unmasked tokens
        # mask:   [B, N] bool — True = masked position
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        tube_size_t: int = 2,
        tube_size_h: int = 2,
        tube_size_w: int = 2,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.tube_t      = tube_size_t
        self.tube_h      = tube_size_h
        self.tube_w      = tube_size_w
        self.mask_value  = mask_value

    def forward(
        self,
        tokens: torch.Tensor,     # [B, T*H*W, D] flattened spatiotemporal tokens
        T: int, H: int, W: int,   # temporal and spatial token grid dimensions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            visible_tokens: [B, N_visible, D]
            mask:           [B, T*H*W] bool — True = MASKED (should be predicted)
        """
        B, N, D = tokens.shape
        assert N == T * H * W, f"Token count mismatch: {N} != {T}*{H}*{W}={T*H*W}"

        # Number of tubes in each dimension
        n_tubes_t = math.ceil(T / self.tube_t)
        n_tubes_h = math.ceil(H / self.tube_h)
        n_tubes_w = math.ceil(W / self.tube_w)
        n_tubes   = n_tubes_t * n_tubes_h * n_tubes_w

        n_mask_tubes = max(1, int(self.mask_ratio * n_tubes))

        masks = []
        for _ in range(B):
            # Sample random tube indices to mask
            perm = torch.randperm(n_tubes)[:n_mask_tubes]
            mask = torch.zeros(T * H * W, dtype=torch.bool, device=tokens.device)

            for tube_idx in perm:
                # Convert flat tube index to (tt, th, tw)
                tt = (tube_idx // (n_tubes_h * n_tubes_w)).item()
                th = ((tube_idx % (n_tubes_h * n_tubes_w)) // n_tubes_w).item()
                tw = (tube_idx % n_tubes_w).item()

                # Token range covered by this tube
                t_start = tt * self.tube_t
                h_start = th * self.tube_h
                w_start = tw * self.tube_w

                for t in range(t_start, min(t_start + self.tube_t, T)):
                    for h in range(h_start, min(h_start + self.tube_h, H)):
                        for w in range(w_start, min(w_start + self.tube_w, W)):
                            mask[t * H * W + h * W + w] = True

            masks.append(mask)

        mask_batch = torch.stack(masks, dim=0)   # [B, T*H*W]

        # Keep only visible tokens
        # For batched masking, we use the same mask pattern per sample
        visible_tokens = tokens[~mask_batch.unsqueeze(-1).expand_as(tokens)].reshape(
            B, -1, D
        ) if mask_batch.shape[0] == 1 else self._apply_mask_per_sample(tokens, mask_batch)

        return visible_tokens, mask_batch

    def _apply_mask_per_sample(
        self,
        tokens: torch.Tensor,      # [B, N, D]
        mask: torch.Tensor,        # [B, N] bool
    ) -> torch.Tensor:
        """Apply potentially different masks per sample in a batch."""
        B, N, D = tokens.shape
        # Find minimum number of visible tokens across batch
        n_visible = (~mask).sum(dim=1).min().item()
        visible = []
        for b in range(B):
            vis_b = tokens[b][~mask[b]]   # [n_vis_b, D]
            visible.append(vis_b[:n_visible])
        return torch.stack(visible, dim=0)   # [B, n_visible, D]

    def reconstruct_full(
        self,
        visible_tokens: torch.Tensor,    # [B, N_visible, D]
        mask: torch.Tensor,              # [B, N] bool
        predicted_tokens: torch.Tensor,  # [B, N_masked, D] — from Predictor
    ) -> torch.Tensor:
        """
        Combine visible and predicted tokens into full sequence.
        Used during training to compute the prediction loss on masked positions.

        Returns:
            full_tokens: [B, N, D]
        """
        B, N = mask.shape
        D = visible_tokens.shape[-1]
        full = torch.zeros(B, N, D, device=visible_tokens.device)

        for b in range(B):
            full[b][~mask[b]] = visible_tokens[b]
            full[b][mask[b]]  = predicted_tokens[b]

        return full


# ---------------------------------------------------------------------------
# 2. Video Pixel-Space Augmentations
# ---------------------------------------------------------------------------

class VideoAugmentor(nn.Module):
    """
    Pixel-space video augmentations applied before X-Encoder.

    Augmentations:
      - Random horizontal flip (p=0.5)
      - Random temporal jitter (sample frames at random intervals)
      - Color jitter (brightness, contrast, saturation, hue)
      - Random frame masking (zero out entire random frames)
      - Random spatial crop + resize

    Args:
        img_size:       Output spatial size after crop
        min_crop_scale: Minimum crop scale relative to original size
        color_jitter:   Apply color augmentation
        frame_mask_prob: Probability of masking each frame (0.0 = disabled)
        flip_prob:      Horizontal flip probability
    """

    def __init__(
        self,
        img_size: int = 224,
        min_crop_scale: float = 0.5,
        color_jitter: bool = True,
        color_jitter_strength: float = 0.4,
        frame_mask_prob: float = 0.1,
        flip_prob: float = 0.5,
        grayscale_prob: float = 0.1,
    ):
        super().__init__()
        self.img_size        = img_size
        self.min_crop_scale  = min_crop_scale
        self.color_jitter    = color_jitter
        self.cj_strength     = color_jitter_strength
        self.frame_mask_prob = frame_mask_prob
        self.flip_prob       = flip_prob
        self.grayscale_prob  = grayscale_prob

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [C, T, H, W] float32 in [0, 1] (or normalized)
        Returns:
            augmented: [C, T, H', W'] with H'=W'=img_size
        """
        C, T, H, W = video.shape

        # 1. Random horizontal flip
        if random.random() < self.flip_prob:
            video = torch.flip(video, dims=[-1])

        # 2. Random spatial crop + resize
        scale = random.uniform(self.min_crop_scale, 1.0)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        if crop_h < H or crop_w < W:
            top  = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            video = video[:, :, top:top+crop_h, left:left+crop_w]

        # Resize to target
        if video.shape[-2] != self.img_size or video.shape[-1] != self.img_size:
            # [C, T, H, W] -> [C*T, 1, H, W] for interpolation -> [C, T, img_size, img_size]
            video = video.reshape(C * T, 1, video.shape[-2], video.shape[-1])
            video = F.interpolate(video, size=(self.img_size, self.img_size),
                                  mode="bilinear", align_corners=False)
            video = video.reshape(C, T, self.img_size, self.img_size)

        # 3. Color jitter (applied independently per-frame)
        if self.color_jitter and random.random() > 0.2:
            s = self.cj_strength
            # Brightness
            bright = random.uniform(max(0, 1-s), 1+s)
            video  = (video * bright).clamp(0, 1)
            # Contrast
            contrast = random.uniform(max(0, 1-s), 1+s)
            mean     = video.mean(dim=(0, 2, 3), keepdim=True)
            video    = ((video - mean) * contrast + mean).clamp(0, 1)
            # Saturation (simple: blend with grayscale)
            sat   = random.uniform(max(0, 1-s*0.5), 1+s*0.5)
            gray  = video.mean(dim=0, keepdim=True)
            video = ((video - gray) * sat + gray).clamp(0, 1)

        # 4. Grayscale conversion (rare, simulates night/monochrome video)
        if random.random() < self.grayscale_prob:
            gray  = (0.299 * video[0] + 0.587 * video[1] + 0.114 * video[2])
            video = gray.unsqueeze(0).expand(3, -1, -1, -1)

        # 5. Frame masking (zero out entire frames)
        if self.frame_mask_prob > 0:
            for t in range(T):
                if random.random() < self.frame_mask_prob:
                    video[:, t, :, :] = 0.0

        return video


# ---------------------------------------------------------------------------
# 3. Multi-view / Multi-crop Strategy
# ---------------------------------------------------------------------------

class MultiCropAugmentor(nn.Module):
    """
    Multi-crop augmentation strategy (from DINO / iBOT).

    Generates N_global large crops and N_local small crops from each video.
    The Predictor is trained to predict global-crop embeddings given local crops.
    This multi-scale training significantly improves feature quality.

    Args:
        n_global:       Number of large crops (default 2)
        n_local:        Number of small crops (default 4)
        global_size:    Spatial size of global crops
        local_size:     Spatial size of local crops
        global_scale:   Min crop scale for global crops (0.4-1.0)
        local_scale:    Min crop scale for local crops (0.05-0.4)
    """

    def __init__(
        self,
        n_global: int = 2,
        n_local: int = 4,
        global_size: int = 224,
        local_size: int = 96,
        global_scale: Tuple[float, float] = (0.4, 1.0),
        local_scale: Tuple[float, float] = (0.05, 0.4),
    ):
        super().__init__()
        self.n_global = n_global
        self.n_local  = n_local

        self.global_aug = VideoAugmentor(
            img_size=global_size,
            min_crop_scale=global_scale[0],
            color_jitter=True,
        )
        self.local_aug = VideoAugmentor(
            img_size=local_size,
            min_crop_scale=local_scale[0],
            color_jitter=True,
            color_jitter_strength=0.6,
        )

    def forward(self, video: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            global_crops: List of n_global tensors [C, T, global_size, global_size]
            local_crops:  List of n_local tensors  [C, T, local_size, local_size]
        """
        global_crops = [self.global_aug(video) for _ in range(self.n_global)]
        local_crops  = [self.local_aug(video)  for _ in range(self.n_local)]
        return global_crops, local_crops


# ---------------------------------------------------------------------------
# 4. EMA (Exponential Moving Average) Target Encoder
# ---------------------------------------------------------------------------

class EMAUpdater:
    """
    Maintains an Exponential Moving Average copy of an encoder (target network).

    This is the core stability mechanism in JEPA architectures:
      - Online encoder: updated by gradient descent (the context encoder)
      - Target encoder: updated slowly by EMA of the online encoder
      - The Predictor tries to predict TARGET encoder outputs from CONTEXT encoder inputs

    Without EMA, the model can collapse (both encoders converge to trivial solutions).

    Mathematical update:
        theta_target = tau * theta_target + (1 - tau) * theta_online
        where tau starts near 1.0 (slow) and optionally decreases during training.

    Args:
        online_encoder:  The encoder being trained by backprop
        tau:             EMA decay rate (0.996 = paper default, higher = slower update)
        tau_schedule:    If True, warm up tau from tau_start → 1.0 over training

    Usage:
        ema = EMAUpdater(model.x_encoder, tau=0.996)
        # In training loop, after optimizer.step():
        ema.update()
        # Use ema.target for computing target embeddings:
        with torch.no_grad():
            target_tokens = ema.target(video)
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        tau: float = 0.996,
        tau_schedule: bool = True,
        tau_start: float = 0.994,
        tau_end: float = 1.0,
        total_steps: int = 10000,
    ):
        self.online  = online_encoder
        self.tau     = tau
        self.tau_schedule = tau_schedule
        self.tau_start    = tau_start
        self.tau_end      = tau_end
        self.total_steps  = total_steps
        self._step = 0

        # Create target as a deep copy with no grad
        import copy
        self.target = copy.deepcopy(online_encoder)
        for param in self.target.parameters():
            param.requires_grad_(False)

        # Initialize target = online (zero lag)
        self._sync()

    def _sync(self):
        """Hard-copy online -> target (used at initialization)."""
        for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
            t_param.data.copy_(o_param.data)

    def get_current_tau(self) -> float:
        """Cosine schedule for tau: starts at tau_start, approaches tau_end."""
        if not self.tau_schedule:
            return self.tau
        progress = min(self._step / max(self.total_steps, 1), 1.0)
        return self.tau_end - (self.tau_end - self.tau_start) * (
            1.0 - math.cos(math.pi * progress)
        ) / 2.0

    @torch.no_grad()
    def update(self):
        """
        EMA update: theta_target = tau * theta_target + (1 - tau) * theta_online.
        Call after every optimizer step.
        """
        tau = self.get_current_tau()
        for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
            t_param.data.mul_(tau).add_(o_param.data, alpha=1.0 - tau)
        self._step += 1

    def state_dict(self) -> dict:
        return {
            "target_state": self.target.state_dict(),
            "step": self._step,
            "tau": self.tau,
        }

    def load_state_dict(self, state: dict):
        self.target.load_state_dict(state["target_state"])
        self._step = state.get("step", 0)
        self.tau   = state.get("tau", self.tau)

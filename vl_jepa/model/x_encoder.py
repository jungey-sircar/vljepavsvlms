"""
X-Encoder — Video/Image Visual Encoder for VL-JEPA
====================================================
The X-Encoder encodes raw video frames (or images) into a sequence of
compact, abstract visual tokens. It is the visual backbone of VL-JEPA and
corresponds to the "X" modality in the JEPA framework.

Architecture:
  - Vision Transformer (ViT) that processes T × H × W video frames
  - Temporal patch embedding: each frame is split into non-overlapping spatial
    patches, and a 3-D (space-time) positional embedding is added
  - The output is a flat sequence of visual tokens of shape [B, N, D]
    where N = (T/t) × (H/p) × (W/p) — (temporal patches × spatial patches)

In the full VL-JEPA model this encoder can be:
  - Initialized from a pretrained V-JEPA 2 ViT-g-384 checkpoint (recommended)
  - Trained from scratch (demo mode)
  - Kept frozen during VL-JEPA training (as in the original paper)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility: Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Standard multi-head self-attention with optional flash-attention path."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # 3, B, H, N, D
        q, k, v = qkv.unbind(0)                   # each: B, H, N, D

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Utility: MLP block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ViT Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 3-D Sinusoidal Positional Embedding (space-time)
# ---------------------------------------------------------------------------

class SpaceTimePositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for space-time video tokens.
    Shape: [1, T_patches * H_patches * W_patches, dim]
    """

    def __init__(self, num_time_patches: int, num_spatial_patches: int, dim: int):
        super().__init__()
        num_tokens = num_time_patches * num_spatial_patches
        self.embedding = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.shape[1], :]


# ---------------------------------------------------------------------------
# Video Patch Embedding
# ---------------------------------------------------------------------------

class VideoPatchEmbedding(nn.Module):
    """
    Splits T × H × W video into non-overlapping 3-D patches and projects each
    patch into a D-dimensional embedding via a single Conv3D.

    Args:
        in_channels:      Number of input channels (default 3 for RGB)
        patch_size:       Spatial patch size in pixels (default 16)
        temporal_patch:   Number of frames per temporal patch (default 2)
        dim:              Output embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch: int = 2,
        dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch = temporal_patch
        self.proj = nn.Conv3d(
            in_channels,
            dim,
            kernel_size=(temporal_patch, patch_size, patch_size),
            stride=(temporal_patch, patch_size, patch_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W] — batch of video tensors

        Returns:
            tokens: [B, N, D] where N = (T//t) * (H//p) * (W//p)
        """
        x = self.proj(video)          # [B, D, T', H', W']
        B, D, Tp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, Tp*Hp*Wp, D]
        return x


# ---------------------------------------------------------------------------
# X-Encoder (Main Class)
# ---------------------------------------------------------------------------

class XEncoder(nn.Module):
    """
    Video ViT encoder — the X-Encoder component of VL-JEPA.

    Encodes a batch of video clips into a sequence of abstract visual tokens.
    The output tokens are consumed by the Predictor (cross-attention) to
    predict the target text embedding given a language query.

    Default configuration matches a compact ViT-B/16 operating on
    short video clips (8 frames × 224×224).

    Args:
        num_frames:       Number of input frames T (must be divisible by temporal_patch)
        img_size:         Spatial resolution H = W (must be divisible by patch_size)
        patch_size:       Spatial patch size in pixels
        temporal_patch:   Temporal patch size (frames per temporal token)
        in_channels:      Input channels (3 for RGB)
        dim:              Token embedding dimension
        depth:            Number of transformer blocks
        num_heads:        Number of attention heads
        mlp_ratio:        MLP hidden dim expansion ratio
        dropout:          Dropout probability
    """

    def __init__(
        self,
        num_frames: int = 8,
        img_size: int = 224,
        patch_size: int = 16,
        temporal_patch: int = 2,
        in_channels: int = 3,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        assert num_frames % temporal_patch == 0, \
            f"num_frames ({num_frames}) must be divisible by temporal_patch ({temporal_patch})"
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        num_time_patches = num_frames // temporal_patch
        num_spatial_patches = (img_size // patch_size) ** 2
        self.num_tokens = num_time_patches * num_spatial_patches

        # Patch embedding
        self.patch_embed = VideoPatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            temporal_patch=temporal_patch,
            dim=dim,
        )

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding (covers CLS + patch tokens)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_tokens, dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W] — normalized video tensor

        Returns:
            tokens: [B, 1+N, D] — CLS token followed by patch tokens
        """
        B = video.shape[0]

        # Patch embed
        x = self.patch_embed(video)                    # [B, N, D]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)         # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                 # [B, 1+N, D]

        # Add positional embedding
        x = x + self.pos_embed[:, : x.shape[1], :]

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x                                       # [B, 1+N, D]

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Convenience method — returns only the CLS token as a global
        video representation (useful for retrieval).

        Returns:
            cls_embed: [B, D]
        """
        tokens = self.forward(video)
        return tokens[:, 0, :]                         # [B, D]

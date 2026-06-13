"""
Predictor — Cross-Attention Transformer for VL-JEPA
=====================================================
The Predictor is the **core learned component** of VL-JEPA. It maps:

    (visual context tokens, text query embedding) → predicted target embedding

During training:
  - The X-Encoder produces visual tokens x = [CLS, patch₁, …, patchₙ]
  - The Y-Encoder produces a query embedding q for the text question
  - The Predictor outputs a predicted embedding ŷ
  - The loss minimizes distance between ŷ and y (target text embedding)

Architecture — two-stage cross-attention transformer:
  1. Self-attention over visual tokens (refines visual context)
  2. Cross-attention: query = text embedding, keys/values = visual tokens
     → outputs predicted embedding in text embedding space

This decoupled design lets the visual encoder (X-Encoder) be frozen while
only the Predictor learns the vision-language alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block.
    Queries come from the language side; Keys/Values from the visual side.
    """

    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection (from language space)
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        # Key/Value projections (from visual space)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)

        self.out_proj = nn.Linear(query_dim, query_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_ctx = nn.LayerNorm(context_dim)

    def forward(
        self,
        query: torch.Tensor,        # [B, Lq, Dq] — from language side
        context: torch.Tensor,      # [B, Lv, Dv] — visual tokens
        context_mask: Optional[torch.Tensor] = None,  # [B, Lv] bool mask
    ) -> torch.Tensor:
        B, Lq, Dq = query.shape
        _, Lv, _ = context.shape

        q = self.q_proj(self.norm_q(query))
        k = self.k_proj(self.norm_ctx(context))
        v = self.v_proj(context)

        # Reshape for multi-head attention
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, Lq, Lv]
        if context_mask is not None:
            # Mask out padding tokens in the visual context
            attn = attn.masked_fill(
                context_mask[:, None, None, :],  # broadcast over H and Lq
                float("-inf"),
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Lq, Dq)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Self-Attention Block (for visual token refinement)
# ---------------------------------------------------------------------------

class VisualSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + res
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Combined Cross-Attention + MLP block
# ---------------------------------------------------------------------------

class PredictorBlock(nn.Module):
    """
    One Predictor layer:
      1. Self-attend visual tokens (optional — only in the first few layers)
      2. Cross-attend text query over visual tokens
      3. MLP
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(query_dim, context_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(query_dim)
        hidden = int(query_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, query_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention (with residual)
        query = query + self.dropout(self.cross_attn(query, context, context_mask))
        # MLP (with residual)
        query = query + self.mlp(self.norm(query))
        return query


# ---------------------------------------------------------------------------
# Predictor — main class
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """
    VL-JEPA Predictor: maps (visual tokens, text query) → predicted embedding.

    The predicted embedding is compared against the target text embedding
    (produced by the frozen Y-Encoder) during training.

    Args:
        visual_dim:     Dimension of X-Encoder output tokens
        text_dim:       Dimension of Y-Encoder output embeddings
        hidden_dim:     Internal predictor dimension (can differ from both)
        depth:          Number of PredictorBlock layers
        num_heads:      Number of attention heads
        num_vis_self_attn_layers:
                        How many initial layers apply self-attention over
                        visual tokens before cross-attending (default: 2)
        mlp_ratio:      MLP expansion ratio
        dropout:        Dropout probability
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 512,
        hidden_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        num_vis_self_attn_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project visual tokens into predictor's hidden dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim, bias=False)

        # Project text query into predictor's query dimension
        self.query_proj = nn.Linear(text_dim, hidden_dim, bias=False)

        # Optional visual self-attention layers (refine context before cross-attn)
        self.vis_self_attn = nn.ModuleList([
            VisualSelfAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_vis_self_attn_layers)
        ])

        # Predictor layers (cross-attention: query=text, context=visual)
        self.layers = nn.ModuleList([
            PredictorBlock(hidden_dim, hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection: hidden_dim → text_dim (matching Y-Encoder output)
        self.output_proj = nn.Linear(hidden_dim, text_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,          # [B, N, visual_dim]
        query_embedding: torch.Tensor,        # [B, text_dim]  — text query
        visual_mask: Optional[torch.Tensor] = None,  # [B, N] bool (True = padding)
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens:    [B, N, visual_dim] — X-Encoder output
            query_embedding:  [B, text_dim]      — Y-Encoder output for the query text
            visual_mask:      [B, N] bool optional — True marks padding tokens

        Returns:
            predicted_embedding: [B, text_dim] — L2-normalized predicted target embedding
        """
        B = visual_tokens.shape[0]

        # Project to hidden space
        ctx = self.visual_proj(visual_tokens)          # [B, N, H]
        q = self.query_proj(query_embedding)           # [B, H]
        q = q.unsqueeze(1)                             # [B, 1, H]  — single query token

        # Refine visual context with self-attention
        for sa in self.vis_self_attn:
            ctx = sa(ctx)

        # Cross-attend: query over visual context, layer-by-layer
        for layer in self.layers:
            q = layer(q, ctx, visual_mask)

        q = self.norm(q)                               # [B, 1, H]
        q = q.squeeze(1)                               # [B, H]

        # Project back to text embedding space
        pred = self.output_proj(q)                     # [B, text_dim]
        return F.normalize(pred, dim=-1)

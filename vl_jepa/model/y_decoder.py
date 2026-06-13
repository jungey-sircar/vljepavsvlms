"""
Y-Decoder — Lightweight Text Decoder for VL-JEPA (Selective Decoding)
======================================================================
The Y-Decoder translates a predicted embedding produced by the Predictor back
into natural language text.

Key insight from the paper:
  The decoder is NOT always active. VL-JEPA uses **selective decoding**:
  the predicted embedding is monitored, and text is generated only when the
  semantic content changes significantly (e.g., in streaming video). This
  reduces decoding operations by ~2.85× while maintaining performance.

For tasks like open-vocabulary classification and text-to-video retrieval,
the Y-Decoder is not used at all — comparisons are done directly in
embedding space.

Architecture:
  - Small autoregressive transformer decoder
  - Conditioned on the predicted embedding via cross-attention
  - Uses teacher-forcing during training; greedy/beam search at inference

This module also provides:
  - SemanticChangeDetector: detects whether a new predicted embedding
    differs enough from the previous one to warrant generating text
    (used in streaming / selective-decoding mode)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ---------------------------------------------------------------------------
# Causal Self-Attention (for decoder)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[:, :, :L, :L] == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# Decoder Cross-Attention (conditioning on predicted embedding)
# ---------------------------------------------------------------------------

class DecoderCrossAttention(nn.Module):
    """
    Cross-attends each decoder token to the predicted embedding (1 context token).
    """

    def __init__(self, dim: int, context_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, L, D]  — decoder hidden states
            context: [B, Dc]    — predicted embedding (single vector)
        """
        B, L, D = x.shape
        ctx = context.unsqueeze(1)  # [B, 1, Dc]

        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(ctx).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(ctx).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, L, 1]
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, max_seq_len: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, num_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = DecoderCrossAttention(dim, context_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Semantic Change Detector (for selective decoding)
# ---------------------------------------------------------------------------

class SemanticChangeDetector:
    """
    Monitors a stream of predicted embeddings and signals when the semantic
    content has changed enough to warrant generating new text.

    Uses cosine similarity against the last decoded embedding.
    Trigger threshold of 0.85 means "only decode if similarity < 0.85"
    (i.e., content has drifted significantly).

    This implements the selective decoding strategy from VL-JEPA Section 3.4.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._last_embedding: Optional[torch.Tensor] = None

    def reset(self):
        self._last_embedding = None

    def should_decode(self, embedding: torch.Tensor) -> bool:
        """
        Args:
            embedding: [D] normalized embedding for the current frame/segment

        Returns:
            True if the model should generate text, False to skip.
        """
        if self._last_embedding is None:
            self._last_embedding = embedding.detach()
            return True

        sim = F.cosine_similarity(
            embedding.unsqueeze(0),
            self._last_embedding.unsqueeze(0),
        ).item()

        if sim < self.threshold:
            self._last_embedding = embedding.detach()
            return True
        return False


# ---------------------------------------------------------------------------
# Y-Decoder — Main Class
# ---------------------------------------------------------------------------

class YDecoder(nn.Module):
    """
    Lightweight autoregressive text decoder conditioned on a predicted embedding.

    This decoder is used **selectively** — only when the Predictor's output
    embedding differs significantly from the previous one (streaming mode),
    or when explicit text generation is requested (VQA, captioning).

    For classification and retrieval tasks, the Y-Decoder is NOT invoked.

    Args:
        vocab_size:   Vocabulary size (use 256 for standalone byte-level demo)
        context_dim:  Dimension of the predicted embedding (from Predictor output)
        dim:          Decoder hidden dimension
        depth:        Number of decoder transformer layers
        num_heads:    Number of attention heads
        max_seq_len:  Maximum generation length
        mlp_ratio:    MLP expansion ratio
        dropout:      Dropout probability
        pad_id:       Padding token id
        eos_id:       End-of-sequence token id
    """

    def __init__(
        self,
        vocab_size: int = 256,
        context_dim: int = 512,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pad_id: int = 0,
        eos_id: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.eos_id = eos_id

        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            DecoderBlock(dim, context_dim, num_heads, mlp_ratio, dropout, max_seq_len)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights (token_embed ↔ lm_head) — common in language models
        # Only tie if dims match
        if dim == vocab_size:
            self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,    # [B, L] — decoder input tokens (teacher-forced)
        context: torch.Tensor,      # [B, context_dim] — predicted embedding
    ) -> torch.Tensor:
        """
        Training forward pass (teacher forcing).

        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_embed(input_ids) + self.pos_embed(positions)

        for block in self.blocks:
            x = block(x, context)

        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,     # [B, context_dim]
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> List[List[int]]:
        """
        Autoregressive greedy/top-k sampling.

        Returns:
            List of token id lists (one per batch element).
        """
        B = context.shape[0]
        device = context.device

        # Start with BOS token (id=2, reserved) or pad
        bos_id = 2
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, context)   # [B, L, V]
            next_logits = logits[:, -1, :] / temperature   # [B, V]

            if top_k > 0:
                top_vals, _ = next_logits.topk(top_k, dim=-1)
                threshold = top_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs = next_logits.softmax(dim=-1)
            next_tok = torch.multinomial(probs, 1)     # [B, 1]

            generated = torch.cat([generated, next_tok], dim=1)
            finished |= (next_tok.squeeze(-1) == self.eos_id)
            if finished.all():
                break

        # Convert to list of lists (strip BOS, stop at EOS)
        results = []
        for seq in generated.tolist():
            tok = seq[1:]   # strip BOS
            if self.eos_id in tok:
                tok = tok[:tok.index(self.eos_id)]
            results.append(tok)
        return results

    @staticmethod
    def decode_bytes(token_ids: List[int]) -> str:
        """Decode byte-level token ids back to string."""
        return bytes(token_ids).decode("utf-8", errors="replace")

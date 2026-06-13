"""
Y-Encoder — Text Encoder for VL-JEPA
======================================
The Y-Encoder maps text (queries or target answers) into a continuous
embedding space that the Predictor learns to predict.

Key design choices from the paper:
  1. The Y-Encoder produces **continuous vector embeddings**, NOT token logits.
  2. The same Y-Encoder is used for both:
       - The **query** text (conditioning the Predictor on the question)
       - The **target** text (the ground truth the Predictor must predict)
  3. The Y-Encoder is typically kept **frozen** during VL-JEPA training.
     Only the Predictor (and optionally the Y-Decoder) are learned.
  4. It uses a CLIP-style text transformer (WordPiece tokens → CLS embedding).

This implementation provides:
  - A CLIP-text-based encoder using HuggingFace `transformers`
  - A lightweight fallback transformer for demo/offline use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


# ---------------------------------------------------------------------------
# Lightweight standalone text transformer (for demo / no-internet use)
# ---------------------------------------------------------------------------

class _TextTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + residual
        x = x + self.mlp(self.norm2(x))
        return x


class StandaloneTextEncoder(nn.Module):
    """
    A minimal CLIP-style text encoder built from scratch.
    Uses character-level token IDs (vocab_size=256) for simplicity in demo mode.
    In production, replace with a pretrained CLIP tokenizer + text transformer.

    Output: normalized embedding of shape [B, dim]
    """

    def __init__(
        self,
        vocab_size: int = 256,
        max_seq_len: int = 77,
        dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _TextTransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Learned [EOS] → CLS projection
        self.cls_proj = nn.Linear(dim, dim)

    @staticmethod
    def tokenize(texts: List[str], max_seq_len: int = 77, device: torch.device = None) -> torch.Tensor:
        """
        Naïve UTF-8 byte-level tokenizer.
        Returns [B, max_seq_len] int64 tensor.
        """
        ids = []
        for t in texts:
            enc = list(t.encode("utf-8")[:max_seq_len - 1]) + [0]  # 0 = EOS/pad
            enc = enc + [0] * (max_seq_len - len(enc))
            ids.append(enc)
        tok = torch.tensor(ids, dtype=torch.long)
        if device is not None:
            tok = tok.to(device)
        return tok

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] int64 token ids

        Returns:
            embedding: [B, dim] L2-normalized text embedding
        """
        B, L = input_ids.shape
        x = self.token_embed(input_ids)                       # [B, L, D]
        x = x + self.pos_embed[:, :L, :]

        pad_mask = (input_ids == 0)                           # True = pad

        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        x = self.norm(x)

        # Use the last non-padding token as the CLS embedding (CLIP-style)
        seq_lens = (input_ids != 0).sum(dim=1).clamp(min=1) - 1   # [B]
        cls = x[torch.arange(B, device=x.device), seq_lens]       # [B, D]
        cls = self.cls_proj(cls)
        return F.normalize(cls, dim=-1)


# ---------------------------------------------------------------------------
# CLIP-based text encoder (uses HuggingFace transformers)
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """
    Text encoder backed by a pretrained CLIP model from HuggingFace.
    Produces L2-normalized sentence embeddings compatible with CLIP's
    visual embedding space.

    Usage:
        enc = CLIPTextEncoder("openai/clip-vit-base-patch16")
        tokens = enc.tokenize(["a cat sitting on a mat"])
        emb = enc(tokens)  # [1, 512]
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        super().__init__()
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for CLIPTextEncoder. "
                "Install it with: pip install transformers"
            )

        self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self._model = CLIPTextModel.from_pretrained(model_name)
        self.dim = self._model.config.hidden_size

        # Freeze by default (as in the VL-JEPA paper)
        for p in self._model.parameters():
            p.requires_grad = False

    def tokenize(
        self,
        texts: Union[str, List[str]],
        device: Optional[torch.device] = None,
    ) -> dict:
        """
        Tokenize a list of strings using the CLIP tokenizer.

        Returns:
            dict with keys 'input_ids', 'attention_mask' (HF format)
        """
        if isinstance(texts, str):
            texts = [texts]
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, L] int64
            attention_mask: [B, L] int64 (optional)

        Returns:
            embedding: [B, dim] L2-normalized
        """
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        # CLIP text model returns pooler_output as the sentence embedding
        emb = out.pooler_output            # [B, D]
        return F.normalize(emb, dim=-1)


# ---------------------------------------------------------------------------
# Y-Encoder: unified interface
# ---------------------------------------------------------------------------

class YEncoder(nn.Module):
    """
    Y-Encoder: text → continuous embedding.

    Wraps either a CLIPTextEncoder (production) or StandaloneTextEncoder
    (demo / offline).

    Args:
        mode:        'clip' | 'standalone'
        clip_model:  HuggingFace model name (used when mode='clip')
        **kwargs:    Forwarded to StandaloneTextEncoder when mode='standalone'
    """

    def __init__(self, mode: str = "standalone", clip_model: str = "openai/clip-vit-base-patch16", **kwargs):
        super().__init__()
        self.mode = mode

        if mode == "clip":
            self._encoder = CLIPTextEncoder(clip_model)
            self.dim = self._encoder.dim
        elif mode == "standalone":
            self._encoder = StandaloneTextEncoder(**kwargs)
            self.dim = self._encoder.dim
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'clip' or 'standalone'.")

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        High-level convenience: tokenize + encode in one call.

        Returns:
            embedding: [B, dim] L2-normalized
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.mode == "clip":
            enc = self._encoder.tokenize(texts, device=device)
            return self._encoder(**enc)
        else:
            input_ids = StandaloneTextEncoder.tokenize(
                texts,
                max_seq_len=self._encoder.max_seq_len,
                device=device,
            )
            return self._encoder(input_ids)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass — accepts tokenized inputs.

        For standalone mode: input_ids [B, L]
        For clip mode:       input_ids [B, L], attention_mask [B, L]
        """
        if self.mode == "clip":
            return self._encoder(input_ids, attention_mask)
        else:
            return self._encoder(input_ids)

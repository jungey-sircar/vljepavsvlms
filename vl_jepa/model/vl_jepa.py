"""
VLJepa — Full VL-JEPA Model
============================
Integrates all four components into a single module:

  1. XEncoder    — Video/image ViT encoder
  2. YEncoder    — Text encoder (frozen)
  3. Predictor   — Cross-attention transformer (primary learned component)
  4. YDecoder    — Lightweight text decoder (selective decoding)

Reference: "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
           arXiv:2512.10942 — Meta FAIR, 2025

Training objective:
    The Predictor learns to predict the Y-Encoder's embedding of the
    target text, given the X-Encoder's visual tokens and the Y-Encoder's
    embedding of the query text. This is a *non-generative* objective:
    no tokens are generated during training.

Inference capabilities (all native, no task-specific heads):
  - Text-to-video retrieval   → embed videos + texts, rank by cosine similarity
  - Open-vocab classification → embed class names, nearest-neighbor
  - Discriminative VQA        → embed answer candidates, pick highest similarity
  - Text generation           → invoke Y-Decoder (selective or full)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union

from .x_encoder import XEncoder
from .y_encoder import YEncoder
from .predictor import Predictor
from .y_decoder import YDecoder, SemanticChangeDetector


class VLJepa(nn.Module):
    """
    VL-JEPA: Vision-Language Joint Embedding Predictive Architecture.

    Args:
        x_encoder:  XEncoder instance (video/image ViT backbone)
        y_encoder:  YEncoder instance (text encoder, typically frozen)
        predictor:  Predictor instance (cross-attention, the main learned component)
        y_decoder:  YDecoder instance (lightweight text decoder, optional)
        freeze_x_encoder: If True, freeze X-Encoder weights (default: True,
                          as in the original paper which uses pretrained V-JEPA 2)
        freeze_y_encoder: If True, freeze Y-Encoder weights (default: True)
    """

    def __init__(
        self,
        x_encoder: XEncoder,
        y_encoder: YEncoder,
        predictor: Predictor,
        y_decoder: Optional[YDecoder] = None,
        freeze_x_encoder: bool = True,
        freeze_y_encoder: bool = True,
    ):
        super().__init__()
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.predictor = predictor
        self.y_decoder = y_decoder

        if freeze_x_encoder:
            for p in self.x_encoder.parameters():
                p.requires_grad = False

        if freeze_y_encoder:
            for p in self.y_encoder.parameters():
                p.requires_grad = False

    # ── Core forward (used during training) ─────────────────────────────────

    def forward(
        self,
        video: torch.Tensor,               # [B, C, T, H, W]
        query_input_ids: torch.Tensor,     # [B, Lq] — tokenized question
        query_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training — returns the predicted embedding.

        The loss is computed externally (see training/loss.py):
            loss = cosine_loss(predicted_embedding, target_embedding)

        Args:
            video:               [B, C, T, H, W] normalized video tensor
            query_input_ids:     [B, Lq] tokenized query text
            query_attention_mask:[B, Lq] attention mask (for CLIP mode)

        Returns:
            predicted_embedding: [B, text_dim] L2-normalized
        """
        # 1. Encode video → visual tokens
        with torch.set_grad_enabled(not self._x_frozen):
            visual_tokens = self.x_encoder(video)              # [B, N, Dv]

        # 2. Encode query text → query embedding
        with torch.set_grad_enabled(not self._y_frozen):
            query_emb = self.y_encoder(query_input_ids, query_attention_mask)  # [B, Dt]

        # 3. Predict target embedding
        pred_emb = self.predictor(visual_tokens, query_emb)    # [B, Dt]
        return pred_emb

    @property
    def _x_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.x_encoder.parameters())

    @property
    def _y_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.y_encoder.parameters())

    # ── Embedding utilities ──────────────────────────────────────────────────

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode a video clip to a global CLS embedding.
        Useful for building embedding databases for retrieval.

        Returns:
            embedding: [B, Dv] L2-normalized
        """
        tokens = self.x_encoder(video)    # [B, N, Dv]
        cls = tokens[:, 0, :]             # [B, Dv]
        return F.normalize(cls, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode text(s) to embeddings using the Y-Encoder.

        Returns:
            embedding: [B, Dt] L2-normalized
        """
        if device is None:
            device = next(self.parameters()).device
        return self.y_encoder.encode_text(texts, device=device)

    # ── Text-to-Video Retrieval ──────────────────────────────────────────────

    @torch.no_grad()
    def retrieval_scores(
        self,
        video_embeddings: torch.Tensor,   # [N_videos, D]
        text_embeddings: torch.Tensor,    # [N_texts, D]
    ) -> torch.Tensor:
        """
        Compute cosine similarity matrix between video and text embeddings.

        Returns:
            scores: [N_texts, N_videos] — higher = more similar
        """
        # Both already L2-normalized
        return text_embeddings @ video_embeddings.T

    # ── Open-Vocabulary Classification ──────────────────────────────────────

    @torch.no_grad()
    def classify(
        self,
        video: torch.Tensor,               # [B, C, T, H, W]
        class_names: List[str],
        query_text: str = "What action is happening in this video?",
    ) -> torch.Tensor:
        """
        Zero-shot open-vocabulary action classification.

        Predicts an embedding conditioned on the query, then finds the
        nearest class name embedding.

        Returns:
            scores:  [B, num_classes] cosine similarity scores
            pred_ids: [B] index of predicted class
        """
        device = video.device

        # Encode class names
        class_embs = self.encode_text(class_names, device=device)   # [C, D]

        # Encode query
        query_emb = self.encode_text([query_text], device=device)   # [1, D]
        query_emb = query_emb.expand(video.shape[0], -1)            # [B, D]

        # Get visual tokens
        vis_tokens = self.x_encoder(video)                          # [B, N, Dv]

        # Predict embedding
        pred_emb = self.predictor(vis_tokens, query_emb)            # [B, D]

        # Cosine similarity to each class
        scores = pred_emb @ class_embs.T                            # [B, C]
        pred_ids = scores.argmax(dim=-1)                            # [B]
        return scores, pred_ids

    # ── Discriminative VQA (Multiple Choice) ────────────────────────────────

    @torch.no_grad()
    def vqa_discriminative(
        self,
        video: torch.Tensor,               # [B, C, T, H, W]
        questions: List[str],              # length B
        answer_choices: List[List[str]],   # B × num_choices
    ) -> List[int]:
        """
        Discriminative (multiple-choice) VQA.
        Picks the answer whose embedding is closest to the predicted embedding.

        Returns:
            predicted_answer_indices: List[int] of length B
        """
        device = video.device
        B = video.shape[0]
        results = []

        vis_tokens = self.x_encoder(video)   # [B, N, Dv]

        for i in range(B):
            q_emb = self.encode_text([questions[i]], device=device)  # [1, D]
            pred = self.predictor(vis_tokens[i:i+1], q_emb)          # [1, D]

            ans_embs = self.encode_text(answer_choices[i], device=device)  # [A, D]
            scores = (pred @ ans_embs.T).squeeze(0)                   # [A]
            results.append(scores.argmax().item())

        return results

    # ── Text Generation (Selective Decoding) ────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        video: torch.Tensor,               # [B, C, T, H, W]
        query_text: str = "Describe what is happening in this video.",
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> List[str]:
        """
        Generate text by invoking the Y-Decoder on the predicted embedding.

        Requires y_decoder to be set.

        Returns:
            texts: List[str] of generated descriptions
        """
        if self.y_decoder is None:
            raise RuntimeError("y_decoder is None — attach a YDecoder to generate text.")

        device = video.device
        B = video.shape[0]

        vis_tokens = self.x_encoder(video)
        q_emb = self.encode_text([query_text], device=device).expand(B, -1)
        pred_emb = self.predictor(vis_tokens, q_emb)

        token_lists = self.y_decoder.generate(
            pred_emb, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
        )
        return [YDecoder.decode_bytes(t) for t in token_lists]

    @torch.no_grad()
    def selective_decode_stream(
        self,
        frame_embeddings: torch.Tensor,   # [T, Dv] — pre-encoded video frames
        query_text: str,
        threshold: float = 0.85,
        max_new_tokens: int = 64,
    ) -> List[Dict]:
        """
        Selective decoding over a stream of frame embeddings.

        Only calls the Y-Decoder when the semantic content changes significantly
        (cosine similarity < threshold), reducing decode operations by ~2.85×.

        Args:
            frame_embeddings: Pre-encoded frame-level embeddings [T, D]
            query_text:       The question/prompt to condition decoding
            threshold:        Similarity threshold (lower = decode more often)
            max_new_tokens:   Max tokens to generate per decode call

        Returns:
            List of dicts with keys 'frame_idx', 'text', 'embedding'
        """
        if self.y_decoder is None:
            raise RuntimeError("y_decoder is not set.")

        device = frame_embeddings.device
        detector = SemanticChangeDetector(threshold=threshold)
        q_emb = self.encode_text([query_text], device=device)  # [1, D]

        outputs = []
        for t, emb in enumerate(frame_embeddings):
            emb_norm = F.normalize(emb.unsqueeze(0), dim=-1)
            if detector.should_decode(emb_norm.squeeze(0)):
                # Treat the frame embedding as a single visual token
                vis_token = emb_norm.unsqueeze(1)  # [1, 1, D]
                pred = self.predictor(vis_token, q_emb)  # [1, D]
                token_ids = self.y_decoder.generate(
                    pred, max_new_tokens=max_new_tokens, temperature=1.0, top_k=50
                )[0]
                text = YDecoder.decode_bytes(token_ids)
                outputs.append({"frame_idx": t, "text": text, "embedding": pred.squeeze(0)})

        return outputs

    # ── Factory method ───────────────────────────────────────────────────────

    @classmethod
    def build_default(
        cls,
        mode: str = "standalone",
        num_frames: int = 8,
        img_size: int = 224,
        with_decoder: bool = True,
    ) -> "VLJepa":
        """
        Build a default VL-JEPA model for quick experiments.

        Args:
            mode:         'standalone' (demo, no internet) | 'clip' (pretrained)
            num_frames:   Number of video frames
            img_size:     Spatial resolution
            with_decoder: Whether to attach a Y-Decoder

        Returns:
            A VLJepa model with randomly initialized weights (untrained).
        """
        # ViT-S/16 configuration (compact, fast)
        x_enc = XEncoder(
            num_frames=num_frames,
            img_size=img_size,
            patch_size=16,
            temporal_patch=2,
            dim=384,
            depth=6,
            num_heads=6,
        )

        if mode == "clip":
            y_enc = YEncoder(mode="clip", clip_model="openai/clip-vit-base-patch16")
            text_dim = 512
        else:
            y_enc = YEncoder(mode="standalone", dim=256, depth=4, num_heads=4)
            text_dim = 256

        pred = Predictor(
            visual_dim=384,
            text_dim=text_dim,
            hidden_dim=text_dim,
            depth=4,
            num_heads=8,
        )

        decoder = None
        if with_decoder:
            decoder = YDecoder(
                vocab_size=256,
                context_dim=text_dim,
                dim=256,
                depth=3,
                num_heads=4,
            )

        return cls(
            x_encoder=x_enc,
            y_encoder=y_enc,
            predictor=pred,
            y_decoder=decoder,
            freeze_x_encoder=False,   # untrained, so don't freeze for demo
            freeze_y_encoder=False,
        )

    def trainable_parameters(self):
        """Return only the trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self) -> Dict[str, int]:
        """Return parameter counts by component."""
        def count(m):
            return sum(p.numel() for p in m.parameters())

        result = {
            "x_encoder": count(self.x_encoder),
            "y_encoder": count(self.y_encoder),
            "predictor": count(self.predictor),
        }
        if self.y_decoder is not None:
            result["y_decoder"] = count(self.y_decoder)
        result["total"] = sum(result.values())
        result["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return result

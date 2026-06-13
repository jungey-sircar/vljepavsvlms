"""
Selective Decoding for Streaming Video
========================================
One of VL-JEPA's most compelling capabilities is **selective decoding**:
the ability to monitor a video stream and generate text descriptions ONLY
when the semantic content changes significantly.

This reduces decoding operations by ~2.85× compared to uniform decoding
at every frame, while maintaining similar performance.

Algorithm:
  1. For each video segment/frame, the model produces a predicted embedding
  2. The embedding is compared to the last decoded embedding via cosine similarity
  3. If similarity < threshold (content has changed): invoke Y-Decoder → text
  4. If similarity ≥ threshold (similar content):    skip decoding, reuse last text

This is particularly useful for:
  - Real-time video surveillance / monitoring
  - Sports commentary generation
  - Wearable assistive technology (narrate surroundings only when they change)
  - Long-form video summarization

The SelectiveVideoDescriber class provides a complete pipeline.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Iterator
from dataclasses import dataclass, field


@dataclass
class DecodedSegment:
    """A segment of video with an associated text description."""
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    text: str
    embedding: torch.Tensor          # [D] predicted embedding
    similarity_to_prev: float        # cosine sim to previous decoded embedding
    was_decoded: bool                # True = new text generated; False = reused


class SelectiveVideoDescriber:
    """
    Real-time video describer using VL-JEPA selective decoding.

    Processes video frame-by-frame (or segment-by-segment) and generates
    text descriptions only when the content changes.

    Args:
        model:            Trained VLJepa model (must have y_decoder)
        query_text:       The question/prompt to condition generation
        threshold:        Semantic change threshold (cosine similarity)
                          Lower = decode more often, Higher = decode less often
                          Paper uses ~0.85 for 2.85× reduction
        segment_frames:   Number of frames per segment (temporal window)
        max_new_tokens:   Max tokens to generate per decode call
        fps:              Video frame rate (for timestamp computation)
        device:           Computation device
    """

    def __init__(
        self,
        model,
        query_text: str = "Describe what is happening.",
        threshold: float = 0.85,
        segment_frames: int = 8,
        max_new_tokens: int = 48,
        fps: float = 30.0,
        device: Optional[torch.device] = None,
    ):
        if model.y_decoder is None:
            raise ValueError("VLJepa model must have a y_decoder for selective decoding.")

        self.model = model
        self.query_text = query_text
        self.threshold = threshold
        self.segment_frames = segment_frames
        self.max_new_tokens = max_new_tokens
        self.fps = fps
        self.device = device or next(model.parameters()).device

        self._last_embedding: Optional[torch.Tensor] = None
        self._last_text: str = ""
        self._total_segments = 0
        self._decoded_count = 0

        # Pre-encode query text once
        self._query_emb = model.encode_text([query_text], device=self.device)

    def reset(self):
        """Reset state for a new video stream."""
        self._last_embedding = None
        self._last_text = ""
        self._total_segments = 0
        self._decoded_count = 0

    @property
    def decode_rate(self) -> float:
        """Fraction of segments that triggered decoding (lower = more efficient)."""
        if self._total_segments == 0:
            return 0.0
        return self._decoded_count / self._total_segments

    @property
    def speedup(self) -> float:
        """Approximate decoding speedup over uniform decoding."""
        rate = self.decode_rate
        return 1.0 / rate if rate > 0 else float("inf")

    def _should_decode(self, embedding: torch.Tensor) -> bool:
        """Check if new text should be generated based on embedding similarity."""
        if self._last_embedding is None:
            return True
        sim = F.cosine_similarity(
            embedding.unsqueeze(0),
            self._last_embedding.unsqueeze(0),
        ).item()
        return sim < self.threshold

    @torch.no_grad()
    def process_segment(
        self,
        video_segment: torch.Tensor,   # [C, T, H, W] or [1, C, T, H, W]
        frame_start: int = 0,
    ) -> DecodedSegment:
        """
        Process a single video segment and return a DecodedSegment.

        Args:
            video_segment:  Video frames [C, T, H, W]
            frame_start:    Frame index where this segment starts (for timestamps)
        """
        self.model.eval()

        if video_segment.dim() == 4:
            video_segment = video_segment.unsqueeze(0)
        video_segment = video_segment.to(self.device)

        # Predict embedding
        vis_tokens = self.model.x_encoder(video_segment)          # [1, N, Dv]
        pred_emb = self.model.predictor(vis_tokens, self._query_emb)  # [1, D]
        pred_emb_norm = F.normalize(pred_emb, dim=-1)

        self._total_segments += 1
        frame_end = frame_start + self.segment_frames - 1

        # Compute similarity to previous
        if self._last_embedding is not None:
            sim = F.cosine_similarity(pred_emb_norm, self._last_embedding.unsqueeze(0)).item()
        else:
            sim = 0.0

        # Decide whether to decode
        should_decode = self._should_decode(pred_emb_norm.squeeze(0))

        if should_decode:
            # Generate new text
            token_ids = self.model.y_decoder.generate(
                pred_emb_norm,
                max_new_tokens=self.max_new_tokens,
                temperature=1.0,
                top_k=50,
            )[0]
            from vl_jepa.model.y_decoder import YDecoder
            text = YDecoder.decode_bytes(token_ids)
            self._last_text = text
            self._last_embedding = pred_emb_norm.squeeze(0)
            self._decoded_count += 1
            was_decoded = True
        else:
            text = self._last_text
            was_decoded = False

        return DecodedSegment(
            start_frame=frame_start,
            end_frame=frame_end,
            start_sec=frame_start / self.fps,
            end_sec=frame_end / self.fps,
            text=text,
            embedding=pred_emb_norm.squeeze(0).cpu(),
            similarity_to_prev=sim,
            was_decoded=was_decoded,
        )

    @torch.no_grad()
    def process_video(
        self,
        video_tensor: torch.Tensor,   # [C, T_total, H, W]
        overlap: int = 0,
    ) -> List[DecodedSegment]:
        """
        Process an entire video and return segment-level descriptions.

        Args:
            video_tensor: Full video [C, T, H, W]
            overlap:      Frame overlap between consecutive segments

        Returns:
            List[DecodedSegment] sorted by start_frame
        """
        self.reset()
        C, T_total, H, W = video_tensor.shape
        stride = self.segment_frames - overlap
        segments = []

        for start in range(0, T_total - self.segment_frames + 1, stride):
            end = start + self.segment_frames
            clip = video_tensor[:, start:end, :, :]  # [C, T_seg, H, W]
            seg = self.process_segment(clip, frame_start=start)
            segments.append(seg)

        print(f"\nSelective decoding summary:")
        print(f"  Total segments: {self._total_segments}")
        print(f"  Decoded:        {self._decoded_count} ({self.decode_rate:.1%})")
        print(f"  Speedup:        {self.speedup:.2f}×")

        return segments

    def to_srt(self, segments: List[DecodedSegment]) -> str:
        """
        Convert decoded segments to SRT subtitle format.

        Returns:
            SRT-formatted string
        """
        def fmt_time(sec: float) -> str:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int((sec % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        lines = []
        # Merge consecutive segments with identical text
        merged = []
        for seg in segments:
            if merged and merged[-1].text == seg.text:
                # Extend the previous merged segment
                prev = merged[-1]
                merged[-1] = DecodedSegment(
                    start_frame=prev.start_frame,
                    end_frame=seg.end_frame,
                    start_sec=prev.start_sec,
                    end_sec=seg.end_sec,
                    text=prev.text,
                    embedding=prev.embedding,
                    similarity_to_prev=prev.similarity_to_prev,
                    was_decoded=prev.was_decoded,
                )
            else:
                merged.append(seg)

        for i, seg in enumerate(merged, start=1):
            lines.append(str(i))
            lines.append(f"{fmt_time(seg.start_sec)} --> {fmt_time(seg.end_sec)}")
            lines.append(seg.text)
            lines.append("")

        return "\n".join(lines)

    def to_summary(self, segments: List[DecodedSegment]) -> str:
        """
        Create a plain-text temporal summary from decoded segments.
        Only includes segments where new text was generated.
        """
        lines = []
        for seg in segments:
            if seg.was_decoded:
                lines.append(
                    f"[{seg.start_sec:6.1f}s – {seg.end_sec:6.1f}s]  {seg.text}"
                )
        return "\n".join(lines)

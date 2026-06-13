# -*- coding: utf-8 -*-
"""
VL-JEPA End-to-End Demo
========================
Demonstrates all VL-JEPA capabilities using randomly initialized weights
(no pretrained checkpoint required).

Run from the project root:
    python vl_jepa/demo.py

What this demo shows:
  1. Model construction and parameter counts
  2. Forward pass (training objective)
  3. Zero-shot classification (embedding nearest-neighbor)
  4. Text-to-video retrieval (cosine similarity in embedding space)
  5. Discriminative VQA (multiple-choice answer selection)
  6. Text generation via Y-Decoder
  7. Selective decoding (streaming video with change detection)

NOTE: With random weights, outputs are meaningless — this is purely a
      sanity check / architecture demonstration. Real results require
      training on Action100M or similar data.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Ensure vl_jepa is importable from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vl_jepa.model import VLJepa


def make_random_video(batch_size=2, num_frames=8, img_size=224) -> torch.Tensor:
    """Create a random video tensor [B, C, T, H, W]."""
    return torch.randn(batch_size, 3, num_frames, img_size, img_size)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print("\n" + "="*60)
    print("  VL-JEPA Demo — arXiv:2512.10942")
    print("  Non-generative Vision-Language Joint Embedding Predictive Architecture")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── 1. Build model ────────────────────────────────────────────────────
    section("1. Building VL-JEPA model (standalone demo mode)")

    model = VLJepa.build_default(
        mode="standalone",
        num_frames=8,
        img_size=224,
        with_decoder=True,
    ).to(device)

    counts = model.parameter_count()
    print("\nParameter counts:")
    for name, n in counts.items():
        print(f"  {name:<22s}: {n/1e6:.2f}M")

    # ── 2. Forward pass ───────────────────────────────────────────────────
    section("2. Forward pass (training mode — embedding prediction)")

    from vl_jepa.training.loss import VLJepaLoss
    from vl_jepa.training.dataset import byte_tokenize

    loss_fn = VLJepaLoss(use_vicreg=True)

    video = make_random_video(batch_size=2).to(device)
    query_text = "What action is happening in this video?"
    target_text = "A person stirs soup in a pot."

    # Tokenize (byte-level)
    query_ids = torch.stack([
        byte_tokenize(query_text) for _ in range(2)
    ]).to(device)

    target_ids = torch.stack([
        byte_tokenize(target_text) for _ in range(2)
    ]).to(device)

    # Forward: predict embedding
    model.train()
    pred_emb = model(video, query_ids)

    # Encode target (ground truth)
    with torch.no_grad():
        target_emb = model.y_encoder(target_ids)

    loss_dict = loss_fn(pred_emb, target_emb)

    print(f"\nInput video shape:      {video.shape}")
    print(f"Predicted embedding:    {pred_emb.shape}")
    print(f"Target embedding:       {target_emb.shape}")
    print(f"\nLoss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k:<20s}: {v.item():.4f}")

    # ── 3. Zero-shot classification ───────────────────────────────────────
    section("3. Zero-shot open-vocabulary classification")

    from vl_jepa.inference import zero_shot_classify

    model.eval()
    class_names = [
        "stir soup",
        "chop vegetables",
        "mix batter",
        "roll dough",
        "pour liquid",
        "crack eggs",
        "slice bread",
        "wash hands",
    ]

    video_single = make_random_video(batch_size=1).to(device)
    results = zero_shot_classify(
        model=model,
        video=video_single,
        class_names=class_names,
        prompt_ensemble=True,
    )

    print(f"\nClass names: {class_names}")
    print(f"\nTop-3 predictions (random weights — not meaningful):")
    for i, (cls, score) in enumerate(zip(results[0]["top_classes"][:3], results[0]["top_scores"][:3])):
        print(f"  {i+1}. {cls:<25s}  score: {score:.4f}")

    # ── 4. Text-to-Video Retrieval ────────────────────────────────────────
    section("4. Text-to-Video Retrieval (embedding database)")

    from vl_jepa.inference import text_to_video_retrieval

    # Simulate a small video database with random embeddings.
    # For retrieval, we project each video into TEXT embedding space via:
    #   visual_tokens (X-Encoder) + neutral query (Predictor) -> text-dim embedding
    # This ensures video and query embeddings live in the same 256-D space.
    n_videos = 10
    fake_video_paths = [f"/data/videos/video_{i:04d}.mp4" for i in range(n_videos)]

    neutral_query = "Describe this video."
    with torch.no_grad():
        fake_videos = make_random_video(batch_size=n_videos).to(device)
        vis_tokens = model.x_encoder(fake_videos)                              # [N, tokens, 384]
        neutral_q = model.encode_text([neutral_query], device=device)          # [1, 256]
        neutral_q = neutral_q.expand(n_videos, -1)                             # [N, 256]
        video_embeddings = model.predictor(vis_tokens, neutral_q).cpu()        # [N, 256]

    query_texts = [
        "a person cooking in a kitchen",
        "someone mixing ingredients in a bowl",
    ]

    retrieval_results = text_to_video_retrieval(
        model=model,
        query_texts=query_texts,
        video_embeddings=video_embeddings,
        video_paths=fake_video_paths,
        top_k=3,
        device=device,
    )

    for r in retrieval_results:
        print(f"\n  Query: '{r['query']}'")
        print(f"  Top-3 results:")
        for path, score in zip(r["top_videos"], r["scores"]):
            print(f"    {path}  (score: {score:.4f})")

    # ── 5. Discriminative VQA ─────────────────────────────────────────────
    section("5. Discriminative VQA (multiple-choice)")

    from vl_jepa.inference import discriminative_vqa

    vqa_video = make_random_video(batch_size=1).to(device)
    questions = ["What ingredient is the person adding to the bowl?"]
    answer_choices = [["flour", "sugar", "salt", "butter"]]

    vqa_results = discriminative_vqa(
        model=model,
        video=vqa_video,
        questions=questions,
        answer_choices=answer_choices,
        device=device,
    )

    print(f"\n  Question: {vqa_results[0]['question']}")
    print(f"  Choices:  {answer_choices[0]}")
    print(f"  Scores:   {[f'{s:.4f}' for s in vqa_results[0]['scores']]}")
    print(f"  Predicted: '{vqa_results[0]['predicted_ans']}' (random weights — not meaningful)")

    # ── 6. Text Generation via Y-Decoder ─────────────────────────────────
    section("6. Text Generation via Y-Decoder (selective decoding off)")

    gen_video = make_random_video(batch_size=1).to(device)
    texts = model.generate(
        video=gen_video,
        query_text="Describe what is happening in this video.",
        max_new_tokens=20,
        temperature=1.2,
    )

    print(f"\n  Generated text (random weights — gibberish expected):")
    for i, t in enumerate(texts):
        display = repr(t[:60]) if t else "(empty)"
        print(f"  [{i}] {display}")

    # ── 7. Selective Decoding ─────────────────────────────────────────────
    section("7. Selective Decoding — stream 32 frames (threshold=0.85)")

    from vl_jepa.inference import SelectiveVideoDescriber

    describer = SelectiveVideoDescriber(
        model=model,
        query_text="What is happening?",
        threshold=0.85,
        segment_frames=8,
        max_new_tokens=16,
        fps=30.0,
        device=device,
    )

    # Simulate 32-frame video stream
    full_video = torch.randn(3, 32, 224, 224).to(device)
    segments = describer.process_video(full_video)

    print(f"\n  Segments decoded:")
    for seg in segments:
        marker = "[DECODED]" if seg.was_decoded else "[skipped]"
        print(f"  [{seg.start_frame:3d}-{seg.end_frame:3d}f]  {marker}  "
              f"sim={seg.similarity_to_prev:.3f}")

    print(f"\n  Decode rate: {describer.decode_rate:.1%}  "
          f"(speedup: {describer.speedup:.2f}x)")

    # ── Summary ───────────────────────────────────────────────────────────
    section("Summary")
    print("""
  VL-JEPA key properties demonstrated:
    [OK] Embedding prediction loss (cosine + MSE) -- non-generative training
    [OK] Zero-shot classification via embedding nearest-neighbor
    [OK] Text-to-video retrieval via cosine similarity
    [OK] Discriminative VQA via answer embedding selection
    [OK] Text generation via lightweight Y-Decoder
    [OK] Selective decoding for streaming video efficiency

  Architecture highlights:
    - X-Encoder: ViT processes video frames -> visual tokens
    - Y-Encoder: Text transformer -> continuous embeddings (FROZEN in training)
    - Predictor:  Cross-attention (visual x text) -> predicted embedding (TRAINED)
    - Y-Decoder:  Small causal transformer, invoked selectively

  To train on Action100M:
    python vl_jepa/training/train.py --dataset_name action100m --epochs 10

  To train on custom JSONL data:
    python vl_jepa/training/train.py --jsonl_path data/train.jsonl

  See README.md for full usage instructions.
""")


if __name__ == "__main__":
    main()

# Model Card — ISL BiLSTM Classifier

## Model Summary

A Bidirectional LSTM (BiLSTM) classifier for isolated Indian Sign Language (ISL) word recognition from normalized body/hand keypoint sequences.

## Model Details

| Field | Value |
|-------|-------|
| Architecture | 2-layer BiLSTM with attention pooling |
| Input | `(1, 50, 225)` — batch × frames × keypoint features |
| Output | `(1, N)` logits over N vocabulary classes |
| Framework | PyTorch (training) → ONNX (inference) |
| Parameters | ~500K (approximate for 10-class model) |
| Training time | ~5–15 min on CPU for 10 classes, 150 epochs |

## Architecture Details

```
Input: (batch, 50 frames, 225 features)
  → Linear projection (225 → 128) + LayerNorm + ReLU + Dropout(0.15)
  → 2-layer BiLSTM (hidden=128, dropout=0.3)    → (batch, 50, 256)
  → Attention pooling (learned soft weights)     → (batch, 256)
  → Dropout(0.3)
  → Linear (256 → 128) + ReLU + Dropout(0.15)
  → Linear (128 → N)
  → LogSoftmax
```

**Why attention pooling?** Rather than using only the final hidden state, we learn a soft weighting over all timesteps. This lets the model focus on the most informative frames of a sign (typically the peak hand shape and movement) rather than treating all frames equally.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR, T_max=150 |
| Loss | CrossEntropyLoss |
| Early stopping | Patience=15 epochs on val accuracy |
| Augmentation | Time warping ±10%, coordinate noise σ=0.005, horizontal flip (p=0.5) |
| Train/val/test | 70/15/15 stratified split |

## Performance

> Fill in after running `python training/evaluate.py`

| Metric | Value |
|--------|-------|
| Test accuracy (top-1) | *TBD* |
| Test accuracy (top-3) | *TBD* |
| Validation accuracy (best) | *TBD* |

**Target**: 75–85% top-1 test accuracy is a strong, honest result for a 10–30 class self-collected dataset with a single signer.

See `training/plots/confusion_matrix_bilstm.png` for per-class breakdown.

## Intended Use

- Demonstration of real-time ISL gesture recognition
- Research baseline for ISL recognition on small vocabulary
- Educational tool to explore sign language technology

## Out-of-Scope Uses

> [!CAUTION]
> **This model MUST NOT be used for:**
> - Medical decision-making or patient communication in clinical settings
> - Legal proceedings or certified interpretation
> - Safety-critical communications (emergency services, etc.)
> - Any application where misrecognition could cause harm

This is a research/portfolio project trained on a small single-signer dataset. It is not a certified accessibility tool and cannot replace professional human interpreters.

## Known Limitations

- **Single signer**: trained on data from one signer — will degrade significantly on other signers
- **Regional variation**: ISL varies across India; this model captures one regional/personal signing style
- **10-word vocabulary**: far too small for practical communication
- **Controlled conditions**: performance degrades in unusual lighting, backgrounds, or camera distances
- **Gesture boundary detection**: the wrist-velocity heuristic may mis-segment fast, continuous signing
- **No temporal context**: isolated-word classifier; does not handle continuous phrase signing

## Fairness Considerations

The training data represents a single signer and signing style. The model may systematically underperform for:
- Signers with different body proportions or arm lengths
- Different regional ISL dialects
- Signers with disabilities affecting hand or arm mobility
- Unusual skin tones in conjunction with certain backgrounds (affects MediaPipe tracking)

## Model Lineage

- Inspired by action recognition literature (LSTM-based gesture classifiers)
- Keypoint extraction: MediaPipe Holistic (Google)
- No pre-trained weights used — trained from scratch on the ISL dataset

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2024 | Initial 10-class BiLSTM baseline |
| v1.1 | 2026 | Fixed duplicate "help" entry in VOCABULARY (config.py) that was silently duplicating clips in the training set; added gesture-boundary "no person in frame" detection |

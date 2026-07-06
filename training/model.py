"""
model.py — BiLSTM classifier for ISL isolated-word recognition.

Architecture:
    Input: (batch, seq_len, feature_dim)
    → Linear input projection (feature_dim → 128)
    → 2-layer Bidirectional LSTM (hidden_dim=128, dropout=0.3)
    → Attention pooling over all hidden states
    → Dropout
    → Linear classifier (hidden_dim*2 → n_classes)
    → LogSoftmax

The attention pooling layer is a key design choice: rather than just using the
final hidden state, we learn a soft weighting over all timesteps. This helps
the model focus on the most informative frames of a sign (e.g., the peak of
a hand shape) rather than treating all frames equally.

Also provided: TransformerClassifier — a drop-in replacement using a small
Transformer encoder. Switch via config or train.py --model flag once the
BiLSTM baseline is established.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_collection.config import FEATURE_DIM, LSTM_HIDDEN_DIM, LSTM_LAYERS, LSTM_DROPOUT


# ── Attention Pooling ──────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Learned soft attention over sequence timesteps.

    Given hidden states H of shape (batch, seq_len, hidden_dim), computes a
    weighted sum: output = sum_t( alpha_t * H_t ) where alpha_t = softmax(score_t)
    and score_t = tanh(W @ H_t + b).

    This is a single-head additive attention (Bahdanau-style) without a query.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) boolean mask; True = valid, False = padding.
                  If None, all positions are considered valid.

        Returns:
            context: (batch, hidden_dim) — weighted sum of hidden states.
        """
        scores = self.score(hidden_states).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            # Set padding positions to -inf before softmax
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)              # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_dim)
        return context


# ── BiLSTM Classifier ──────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM-based ISL gesture classifier.

    Args:
        n_classes: number of output classes (vocabulary size).
        feature_dim: input feature vector size (default: FEATURE_DIM=225).
        hidden_dim: LSTM hidden dimension (default: LSTM_HIDDEN_DIM=128).
        n_layers: number of LSTM layers (default: LSTM_LAYERS=2).
        dropout: dropout rate (default: LSTM_DROPOUT=0.3).
    """

    def __init__(
        self,
        n_classes: int,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        n_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection: compress raw features into a denser representation
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
        )

        # BiLSTM: bidirectional doubles the output size → hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Attention pooling over BiLSTM hidden states (hidden_dim*2)
        self.attention = AttentionPooling(hidden_dim * 2)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim) input keypoint sequences.
            mask: (batch, seq_len) boolean mask (True = valid frame).

        Returns:
            logits: (batch, n_classes) — raw (unnormalized) class scores.
                    Apply softmax for probabilities.
        """
        # Project input features
        projected = self.input_proj(x)      # (batch, seq_len, hidden_dim)

        # BiLSTM
        lstm_out, _ = self.lstm(projected)  # (batch, seq_len, hidden_dim*2)

        # Attention pooling
        context = self.attention(lstm_out, mask=mask)   # (batch, hidden_dim*2)

        # Classify
        logits = self.classifier(context)  # (batch, n_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (useful for inference)."""
        return F.softmax(self.forward(x), dim=-1)

    @property
    def n_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Transformer Classifier (stretch goal) ─────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    Transformer encoder-based ISL gesture classifier.

    A drop-in replacement for BiLSTMClassifier. Switch to this once the LSTM
    baseline is established to compare accuracy/speed tradeoffs.

    Architecture:
        Input projection → Positional encoding → N×TransformerEncoderLayer
        → [CLS] token pooling → Classifier head
    """

    def __init__(
        self,
        n_classes: int,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Global average pooling → classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim)
            mask: (batch, seq_len) boolean — True = valid, False = padding.

        Returns:
            logits: (batch, n_classes)
        """
        batch, seq_len, _ = x.shape

        # Project + positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        projected = self.input_proj(x) + self.pos_embedding(positions)    # (batch, seq_len, d_model)

        # Transformer encoder
        # src_key_padding_mask: True = ignore (opposite of our mask convention)
        padding_mask = ~mask if mask is not None else None
        encoded = self.transformer(projected, src_key_padding_mask=padding_mask)  # (batch, seq_len, d_model)

        # Global average pooling (masked)
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            pooled = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = encoded.mean(dim=1)

        return self.classifier(pooled)  # (batch, n_classes)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_model(
    n_classes: int,
    model_type: str = "bilstm",
    **kwargs,
) -> nn.Module:
    """
    Build and return a model by name.

    Args:
        n_classes: vocabulary size.
        model_type: 'bilstm' (default) or 'transformer'.
        **kwargs: passed to the model constructor.

    Returns:
        Initialized nn.Module.
    """
    if model_type == "bilstm":
        model = BiLSTMClassifier(n_classes=n_classes, **kwargs)
    elif model_type == "transformer":
        model = TransformerClassifier(n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'bilstm' or 'transformer'.")

    print(f"\n  Built {model.__class__.__name__}  |  {model.n_parameters:,} trainable parameters")
    return model


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_collection.config import CLIP_LENGTH, FEATURE_DIM, VOCABULARY

    n_classes = len(VOCABULARY)
    batch_size = 4

    for mtype in ["bilstm", "transformer"]:
        model = build_model(n_classes=n_classes, model_type=mtype)
        x = torch.randn(batch_size, CLIP_LENGTH, FEATURE_DIM)
        out = model(x)
        proba = model.predict_proba(x) if hasattr(model, "predict_proba") else torch.softmax(out, -1)
        print(f"  Output shape: {out.shape}  |  Proba sum check: {proba.sum(dim=-1)}")

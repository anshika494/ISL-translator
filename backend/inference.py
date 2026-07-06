"""
inference.py — ONNX model inference for ISL gesture classification.

Loads the exported ONNX model and provides a clean prediction interface
used by the WebSocket handler and any other consumer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_collection.config import CLIP_LENGTH, FEATURE_DIM, ONNX_OUTPUT
from data_collection.normalize import pad_or_truncate

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    print("[WARN] onnxruntime not installed — backend inference unavailable.")


class ISLInferenceEngine:
    """
    Wraps the ONNX model for efficient, stateless inference.

    Usage:
        engine = ISLInferenceEngine()
        result = engine.predict(sequence_array)
        # result = {"word": "hello", "confidence": 0.92, "top_k": [...]}
    """

    def __init__(
        self,
        onnx_path: Path | None = None,
        label_map_path: Path | None = None,
        top_k: int = 3,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime is required for backend inference.")

        self.onnx_path = onnx_path or ONNX_OUTPUT
        self.top_k = top_k

        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.onnx_path}\n"
                "Run: python training/export_onnx.py"
            )

        # Load label map
        if label_map_path is None:
            label_map_path = self.onnx_path.parent / "label_map.json"
        with open(label_map_path) as f:
            self.label_map: dict[str, int] = json.load(f)
        self.idx_to_word = {v: k for k, v in self.label_map.items()}
        self.n_classes = len(self.label_map)

        # Load ONNX session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        print(f"  [Inference] Loaded model: {self.onnx_path.name}  "
              f"({self.n_classes} classes)")

    def predict(self, sequence: np.ndarray) -> dict:
        """
        Run inference on a keypoint sequence.

        Args:
            sequence: np.ndarray of shape (T, FEATURE_DIM).
                      Will be padded/truncated to CLIP_LENGTH automatically.

        Returns:
            dict with keys:
              'word'       (str)  — top predicted word
              'confidence' (float) — probability of top prediction (0-1)
              'top_k'      (list of dict) — top-k predictions [{word, confidence}]
        """
        if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected shape (T, {FEATURE_DIM}), got {sequence.shape}"
            )

        # Pad/truncate and add batch dimension
        fixed = pad_or_truncate(sequence, CLIP_LENGTH)
        batch = fixed[np.newaxis, :, :].astype(np.float32)  # (1, CLIP_LENGTH, FEATURE_DIM)

        # ONNX inference
        logits = self.session.run(None, {self.input_name: batch})[0]  # (1, n_classes)
        logits = logits[0]  # (n_classes,)

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        top_indices = np.argsort(probs)[::-1][: self.top_k]
        top_k_results = [
            {"word": self.idx_to_word[int(i)], "confidence": float(probs[i])}
            for i in top_indices
        ]

        return {
            "word": top_k_results[0]["word"],
            "confidence": top_k_results[0]["confidence"],
            "top_k": top_k_results,
        }

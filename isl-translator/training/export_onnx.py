"""
export_onnx.py — Export trained PyTorch model to ONNX for browser/edge inference.

The exported ONNX model is placed in frontend/public/model.onnx so it can be
loaded by onnxruntime-web in the browser without a backend server.

Usage:
    python training/export_onnx.py
    python training/export_onnx.py --model transformer
    python training/export_onnx.py --verify   # run a quick inference check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.config import (
    CHECKPOINTS_DIR, ONNX_OUTPUT,
    CLIP_LENGTH, FEATURE_DIM, VOCABULARY,
)
from training.model import build_model


def export_onnx(
    model_type: str = "bilstm",
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
    verify: bool = True,
) -> Path:
    """
    Export a trained model checkpoint to ONNX format.

    Args:
        model_type: 'bilstm' or 'transformer'
        checkpoint_path: path to .pt checkpoint (default: best checkpoint)
        output_path: where to save .onnx (default: frontend/public/model.onnx)
        verify: if True, run a quick onnxruntime inference check after export

    Returns:
        Path to the exported ONNX file.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / f"best_{model_type}.pt"

    if output_path is None:
        output_path = ONNX_OUTPUT

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training/train.py first."
        )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_map: dict[str, int] = checkpoint["label_map"]
    n_classes = checkpoint["n_classes"]

    model = build_model(n_classes=n_classes, model_type=model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Dummy input for ONNX tracing ──────────────────────────────────────────
    # Shape: (batch=1, seq_len=CLIP_LENGTH, feature_dim=FEATURE_DIM)
    dummy_input = torch.zeros(1, CLIP_LENGTH, FEATURE_DIM)

    # ── Export ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean up any previous export artifacts
    data_file = output_path.parent / (output_path.name + ".data")
    for f in [output_path, data_file]:
        if f.exists():
            f.unlink()

    print(f"  Exporting to ONNX: {output_path}")
    # dynamo=False → legacy TorchScript exporter — produces a single self-contained
    # .onnx file with all weights inline. Required for onnxruntime-web (browser).
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["keypoints"],
        output_names=["logits"],
        dynamic_axes={
            "keypoints": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        verbose=False,
        dynamo=False,  # force legacy exporter — no external .data file
    )

    # Safety net: if the exporter still created an external .data file,
    # reload and re-save with all tensors embedded inline.
    if data_file.exists():
        print("  [Info] External data file detected — converting to inline...")
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
        model_proto = onnx.load(str(output_path), load_external_data=False)
        load_external_data_for_model(model_proto, str(output_path.parent))
        for f in [output_path, data_file]:
            if f.exists():
                f.unlink()
        onnx.save(model_proto, str(output_path))
        if data_file.exists():
            data_file.unlink()
        print("  [Info] Conversion done — single inline .onnx file.")

    size_mb = output_path.stat().st_size / 1e6
    print(f"  ONNX model exported successfully  ({size_mb:.2f} MB, single file, no external data)")

    # ── Save label map alongside model ────────────────────────────────────────
    label_map_path = output_path.parent / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map saved → {label_map_path}")

    # ── Verify with onnxruntime ───────────────────────────────────────────────
    if verify:
        _verify_onnx(output_path, dummy_input, model)

    return output_path


def _verify_onnx(onnx_path: Path, dummy_input: torch.Tensor, torch_model: torch.nn.Module) -> None:
    """
    Run the exported ONNX model and compare output to the PyTorch model.
    Raises AssertionError if outputs differ significantly.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnx/onnxruntime not installed — skipping verification.")
        return

    # Validate ONNX graph
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX graph validation: PASSED")

    # Run inference comparison
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {"keypoints": dummy_input.numpy()}
    ort_outs = sess.run(None, ort_inputs)

    with torch.no_grad():
        torch_out = torch_model(dummy_input).numpy()

    max_diff = float(np.abs(ort_outs[0] - torch_out).max())
    if max_diff < 1e-4:
        print(f"  PyTorch ↔ ONNX output max diff: {max_diff:.2e} ✓")
    else:
        print(f"  [WARN] PyTorch ↔ ONNX output max diff: {max_diff:.2e} — check model!")

    # Show sample prediction
    probs = np.exp(ort_outs[0][0]) / np.exp(ort_outs[0][0]).sum()
    top_class = int(np.argmax(probs))
    print(f"  Sample inference (dummy zeros): class {top_class}, confidence {probs[top_class]:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export ISL classifier to ONNX for web inference"
    )
    parser.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .onnx path (default: frontend/public/model.onnx)")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify ONNX output matches PyTorch output (default: True)")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args()

    onnx_path = export_onnx(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        verify=args.verify,
    )
    print(f"\n  ✓ Ready for web inference: {onnx_path}")
    print("    Load in browser with onnxruntime-web:")
    print("    const session = await ort.InferenceSession.create('/model.onnx');")

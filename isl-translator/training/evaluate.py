"""
evaluate.py — Model evaluation: confusion matrix, per-class accuracy, top-k accuracy.

Usage:
    python training/evaluate.py
    python training/evaluate.py --model transformer
    python training/evaluate.py --checkpoint path/to/custom.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.config import CHECKPOINTS_DIR, PLOTS_DIR, VOCABULARY
from training.dataset import get_dataloaders
from training.model import build_model


@torch.no_grad()
def run_evaluation(
    model_type: str = "bilstm",
    checkpoint_path: Path | None = None,
    device_str: str = "auto",
    top_k: int = 3,
) -> dict:
    """
    Load the best checkpoint, evaluate on the test set, and produce:
      - Overall accuracy
      - Top-k accuracy
      - Per-class accuracy table
      - Confusion matrix heatmap (saved to training/plots/)
      - Saved metrics JSON

    Returns:
        dict with evaluation metrics.
    """
    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / f"best_{model_type}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training/train.py first."
        )

    print(f"\n  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_map: dict[str, int] = checkpoint["label_map"]
    n_classes = checkpoint["n_classes"]
    idx_to_word = {v: k for k, v in label_map.items()}

    model = build_model(n_classes=n_classes, model_type=model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Data ──────────────────────────────────────────────────────────────────
    _, _, test_dl, _ = get_dataloaders()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds = []
    all_labels = []
    all_probs = []

    for batch_x, batch_y in test_dl:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = F.softmax(logits, dim=-1)

        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(batch_y.tolist())
        all_probs.extend(probs.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── Overall accuracy ──────────────────────────────────────────────────────
    overall_acc = float((all_preds == all_labels).mean())
    print(f"\n  Overall test accuracy: {overall_acc * 100:.2f}%")

    # ── Top-k accuracy ────────────────────────────────────────────────────────
    top_k_correct = 0
    for i, label in enumerate(all_labels):
        top_k_preds = np.argsort(all_probs[i])[-top_k:]
        if label in top_k_preds:
            top_k_correct += 1
    topk_acc = top_k_correct / len(all_labels)
    print(f"  Top-{top_k} test accuracy: {topk_acc * 100:.2f}%")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    print(f"\n  {'Word':<22}  {'Correct':>7}  {'Total':>6}  {'Accuracy':>9}")
    print("  " + "─" * 50)

    per_class_acc = {}
    for class_idx in range(n_classes):
        word = idx_to_word[class_idx]
        mask = all_labels == class_idx
        if mask.sum() == 0:
            continue
        correct = (all_preds[mask] == class_idx).sum()
        total = mask.sum()
        acc = correct / total
        per_class_acc[word] = round(float(acc) * 100, 1)
        bar_len = int(acc * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {word:<22}  {correct:>7}  {total:>6}  {acc*100:>8.1f}%  {bar}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    class_names = [idx_to_word[i] for i in range(n_classes)]
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        cm[label, pred] += 1

    # Normalize rows (row = true class)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_classes), max(8, n_classes - 2)))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt=".0%", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1,
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    cm_path = PLOTS_DIR / f"confusion_matrix_{model_type}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Confusion matrix saved → {cm_path}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "n_classes": n_classes,
        "label_map": label_map,
        "overall_accuracy_pct": round(overall_acc * 100, 2),
        f"top_{top_k}_accuracy_pct": round(topk_acc * 100, 2),
        "per_class_accuracy_pct": per_class_acc,
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = PLOTS_DIR / f"eval_metrics_{model_type}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Evaluation metrics saved → {metrics_path}")

    return metrics


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the ISL classifier on the test set"
    )
    parser.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"])
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a specific checkpoint file (default: best checkpoint)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    metrics = run_evaluation(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        device_str=args.device,
        top_k=args.top_k,
    )
    print(f"\n  Overall: {metrics['overall_accuracy_pct']}%  |  "
          f"Top-{args.top_k}: {metrics[f'top_{args.top_k}_accuracy_pct']}%")

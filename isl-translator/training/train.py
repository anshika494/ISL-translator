"""
train.py — Training loop for the ISL BiLSTM classifier.

Features:
  - Adam optimizer with CosineAnnealingLR scheduler
  - Early stopping on validation accuracy (patience configurable)
  - Best checkpoint saving
  - Loss/accuracy curves saved to training/plots/
  - Supports both 'bilstm' and 'transformer' model types

Usage:
    # From project root:
    python training/train.py
    python training/train.py --model transformer --epochs 200
    python training/train.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.config import (
    CHECKPOINTS_DIR, PLOTS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, EARLY_STOP_PATIENCE, RANDOM_SEED,
    VOCABULARY,
)
from training.dataset import get_dataloaders
from training.model import build_model


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training / evaluation loops ────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a DataLoader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * len(batch_y)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)

    return total_loss / total, correct / total


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_dir: Path,
) -> None:
    """Save loss and accuracy learning curves to training/plots/."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ISL Classifier Training Curves", fontsize=14, fontweight="bold")

    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, label="Train", color="#4C72B0", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val", color="#DD8452", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train", color="#4C72B0", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in val_accs], label="Val", color="#DD8452", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Training curves saved → {save_path}")


# ── Main training function ─────────────────────────────────────────────────────

def train(
    model_type: str = "bilstm",
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = RANDOM_SEED,
    device_str: str = "auto",
) -> dict:
    """
    Full training run.

    Returns:
        dict with keys: 'best_val_acc', 'test_acc', 'n_epochs_trained',
                        'checkpoint_path', 'label_map'
    """
    set_seed(seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")    # Apple Silicon
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"\n  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl, label_map = get_dataloaders(
        batch_size=batch_size, seed=seed
    )
    n_classes = len(label_map)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(n_classes=n_classes, model_type=model_type).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    criterion = nn.CrossEntropyLoss()

    # ── Checkpointing ─────────────────────────────────────────────────────────
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt = CHECKPOINTS_DIR / f"best_{model_type}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n  Training {model_type.upper()} for up to {epochs} epochs "
          f"(early stop patience={patience})")
    print("  " + "─" * 65)
    print(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>8}  {'ValAcc':>7}  {'LR':>9}")
    print("  " + "─" * 65)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        # Check if best model
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "label_map": label_map,
                "model_type": model_type,
                "n_classes": n_classes,
            }, best_ckpt)
            marker = " ★"
        else:
            patience_counter += 1

        # Log every epoch (could change to every N for speed)
        print(f"  {epoch:>6}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
              f"{val_loss:>8.4f}  {val_acc*100:>6.2f}%  {current_lr:>9.2e}{marker}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Best checkpoint → {best_ckpt}")

    # ── Plot curves ───────────────────────────────────────────────────────────
    _plot_curves(train_losses, val_losses, train_accs, val_accs, PLOTS_DIR)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n  Loading best checkpoint for test evaluation...")
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_dl, criterion, device)
    print(f"  Test accuracy: {test_acc * 100:.2f}%  |  Test loss: {test_loss:.4f}")

    # ── Save training metadata ─────────────────────────────────────────────────
    results = {
        "model_type": model_type,
        "n_classes": n_classes,
        "label_map": label_map,
        "best_val_acc": round(best_val_acc * 100, 2),
        "test_acc": round(test_acc * 100, 2),
        "n_epochs_trained": len(train_losses),
        "checkpoint_path": str(best_ckpt),
    }
    results_path = CHECKPOINTS_DIR / f"training_results_{model_type}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the ISL gesture classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python training/train.py
    python training/train.py --model transformer
    python training/train.py --epochs 100 --batch-size 16 --lr 5e-4
        """,
    )
    parser.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"],
                        help="Model architecture (default: bilstm)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default="auto",
                        help="Device: 'auto', 'cpu', 'cuda', 'mps'")
    args = parser.parse_args()

    results = train(
        model_type=args.model,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
    )

    print("\n" + "=" * 50)
    print(f"  FINAL RESULTS")
    print(f"  Best val accuracy : {results['best_val_acc']}%")
    print(f"  Test accuracy     : {results['test_acc']}%")
    print("=" * 50)

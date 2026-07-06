"""
dataset.py — PyTorch Dataset for ISL keypoint sequences.

Loads all .npy clips from data_collection/data/, builds a label map from
config.VOCABULARY, applies train/val/test splits, and provides on-the-fly
data augmentation for the training split.

Usage:
    from training.dataset import get_dataloaders
    train_dl, val_dl, test_dl, label_map = get_dataloaders()

Label map:
    { word_string: int_label }  — word → class index
    The reverse mapping (int → word) is: { v: k for k, v in label_map.items() }
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.config import (
    VOCABULARY, CLIP_LENGTH, FEATURE_DIM,
    DATA_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    BATCH_SIZE, RANDOM_SEED,
)
from data_collection.normalize import pad_or_truncate


# ── Dataset ────────────────────────────────────────────────────────────────────

class ISLDataset(Dataset):
    """
    PyTorch Dataset for ISL isolated-word keypoint sequences.

    Each item is a tuple: (tensor[CLIP_LENGTH, FEATURE_DIM], int_label)

    Args:
        filepaths: list of Path objects pointing to .npy clip files.
        labels: list of integer class labels, aligned with filepaths.
        augment: if True, apply random augmentation (training split only).
    """

    # Augmentation hyperparameters
    _COORD_NOISE_STD = 0.005        # Gaussian noise on normalized coordinates
    _TIME_WARP_MAX = 0.10           # ±10% time warping (resample frames)
    _FLIP_PROB = 0.5                # probability of horizontal flip

    def __init__(
        self,
        filepaths: list[Path],
        labels: list[int],
        augment: bool = False,
    ) -> None:
        assert len(filepaths) == len(labels)
        self.filepaths = filepaths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Load raw (T, 225) array — already normalized at collection time
        seq = np.load(self.filepaths[idx]).astype(np.float32)

        # Augmentation (training only)
        if self.augment:
            seq = self._time_warp(seq)
            seq = self._add_coord_noise(seq)
            seq = self._horizontal_flip(seq)

        # Pad or truncate to fixed CLIP_LENGTH
        seq = pad_or_truncate(seq, CLIP_LENGTH)

        return torch.from_numpy(seq), self.labels[idx]

    # ── Augmentation methods ──────────────────────────────────────────────────

    def _add_coord_noise(self, seq: np.ndarray) -> np.ndarray:
        """Add small Gaussian noise to all coordinates."""
        noise = np.random.normal(0.0, self._COORD_NOISE_STD, size=seq.shape).astype(np.float32)
        return seq + noise

    def _time_warp(self, seq: np.ndarray) -> np.ndarray:
        """
        Randomly stretch or compress the time axis by up to ±10%.
        Uses linear interpolation to resample the sequence.
        """
        T = len(seq)
        warp_factor = 1.0 + np.random.uniform(-self._TIME_WARP_MAX, self._TIME_WARP_MAX)
        new_T = max(1, int(round(T * warp_factor)))

        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, new_T)

        warped = np.zeros((new_T, seq.shape[1]), dtype=np.float32)
        for feat_idx in range(seq.shape[1]):
            warped[:, feat_idx] = np.interp(new_indices, old_indices, seq[:, feat_idx])

        return warped

    def _horizontal_flip(self, seq: np.ndarray) -> np.ndarray:
        """
        Randomly flip the sequence left-right (swap left/right hands, mirror x coords).

        This augmentation helps for ambidextrous signs and reduces left/right bias.
        Only applied with probability FLIP_PROB.

        Landmark layout (per frame, 225 values):
          [0:99]   pose (33 × 3)
          [99:162] left hand (21 × 3)
          [162:225] right hand (21 × 3)
        """
        if np.random.random() > self._FLIP_PROB:
            return seq

        seq = seq.copy()

        # Mirror x coordinates (index 0, 3, 6, ... within each landmark group)
        # Pose block
        pose = seq[:, :99].reshape(-1, 33, 3)
        pose[:, :, 0] = -pose[:, :, 0]   # flip x

        # Swap left and right hand blocks
        left_hand = seq[:, 99:162].copy()
        right_hand = seq[:, 162:225].copy()

        # Flip x within each hand block
        left_hand_r = left_hand.reshape(-1, 21, 3)
        right_hand_r = right_hand.reshape(-1, 21, 3)
        left_hand_r[:, :, 0] = -left_hand_r[:, :, 0]
        right_hand_r[:, :, 0] = -right_hand_r[:, :, 0]

        # Swap
        seq[:, 99:162] = right_hand_r.reshape(-1, 63)
        seq[:, 162:225] = left_hand_r.reshape(-1, 63)
        seq[:, :99] = pose.reshape(-1, 99)

        return seq


# ── Data loading ────────────────────────────────────────────────────────────────

def _load_all_clips() -> tuple[list[Path], list[int], dict[str, int]]:
    """
    Scan data_collection/data/ and load file paths + labels for all words
    in VOCABULARY that have at least one clip.

    Returns:
        filepaths: list of Path objects
        labels: list of int class labels
        label_map: { word: int_label }
    """
    label_map: dict[str, int] = {}
    filepaths: list[Path] = []
    labels: list[int] = []

    for word in VOCABULARY:
        word_dir = DATA_DIR / word
        if not word_dir.exists():
            print(f"  [WARN] No data found for '{word}' — skipping.")
            continue

        clips = sorted(word_dir.glob("*.npy"))
        if not clips:
            print(f"  [WARN] Directory for '{word}' is empty — skipping.")
            continue

        if word not in label_map:
            label_map[word] = len(label_map)

        for clip_path in clips:
            filepaths.append(clip_path)
            labels.append(label_map[word])

    return filepaths, labels, label_map


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """
    Build and return train/val/test DataLoaders + label map.

    Split strategy: stratified split by label to ensure each class appears
    proportionally in all splits.

    Args:
        batch_size: DataLoader batch size.
        num_workers: parallel data loading workers (0 = main process).
        seed: random seed for reproducible splits.

    Returns:
        (train_loader, val_loader, test_loader, label_map)

    Raises:
        RuntimeError if no data is found.
    """
    filepaths, labels, label_map = _load_all_clips()

    if not filepaths:
        raise RuntimeError(
            "No clip data found in data_collection/data/. "
            "Run record_clip.py first to collect training data."
        )

    n_total = len(filepaths)
    n_classes = len(label_map)
    print(f"\n  Loaded {n_total} clips across {n_classes} classes.")
    print(f"  Label map: {label_map}\n")

    # ── Stratified train/val/test split ───────────────────────────────────────
    # Step 1: split off test set
    val_test_ratio = VAL_RATIO + TEST_RATIO
    fps_train, fps_valtest, lbl_train, lbl_valtest = train_test_split(
        filepaths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )
    # Step 2: split val/test from the held-out set
    test_ratio_of_valtest = TEST_RATIO / val_test_ratio
    fps_val, fps_test, lbl_val, lbl_test = train_test_split(
        fps_valtest, lbl_valtest,
        test_size=test_ratio_of_valtest,
        stratify=lbl_valtest,
        random_state=seed,
    )

    print(f"  Split: train={len(fps_train)}, val={len(fps_val)}, test={len(fps_test)}")

    # ── Create Dataset objects ────────────────────────────────────────────────
    train_ds = ISLDataset(fps_train, lbl_train, augment=True)
    val_ds   = ISLDataset(fps_val,   lbl_val,   augment=False)
    test_ds  = ISLDataset(fps_test,  lbl_test,  augment=False)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, label_map


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    try:
        train_dl, val_dl, test_dl, lmap = get_dataloaders(batch_size=8)

        batch_x, batch_y = next(iter(train_dl))
        print(f"\n  Sample batch shape: {batch_x.shape}  (batch, seq_len, features)")
        print(f"  Labels dtype: {batch_y.dtype}")
        print(f"\n  Label map: {lmap}")
        print("\n  Dataset looks healthy ✓")
    except RuntimeError as e:
        print(f"\n  [ERROR] {e}")
        print("  Run record_clip.py first to collect data.")

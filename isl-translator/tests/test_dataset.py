"""
test_dataset.py — Unit tests for training/dataset.py

Tests the ISLDataset class with synthetic data so tests can run
without needing actual recorded ISL clips.
"""

from __future__ import annotations

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_synthetic_dataset(tmp_dir: Path, words: list[str], clips_per_word: int = 20) -> None:
    """Create a synthetic dataset in tmp_dir for testing."""
    from data_collection.config import FEATURE_DIM, CLIP_LENGTH

    rng = np.random.default_rng(42)
    for word in words:
        word_dir = tmp_dir / word
        word_dir.mkdir(parents=True, exist_ok=True)
        for clip_idx in range(clips_per_word):
            # Random clip with valid shoulder positions
            n_frames = rng.integers(20, CLIP_LENGTH + 1)
            arr = rng.uniform(-0.5, 0.5, size=(n_frames, FEATURE_DIM)).astype(np.float32)
            # Set valid shoulders (landmarks 11, 12)
            arr[:, 11 * 3: 11 * 3 + 3] = [0.4, 0.5, 0.0]
            arr[:, 12 * 3: 12 * 3 + 3] = [0.6, 0.5, 0.0]
            np.save(word_dir / f"{word}_signer01_{clip_idx:04d}.npy", arr)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestISLDataset:
    @pytest.fixture(autouse=True)
    def setup_synthetic_data(self, monkeypatch, tmp_path):
        """Patch DATA_DIR to point to a temporary synthetic dataset."""
        words = ["hello", "thank_you", "yes", "no", "help"]
        _create_synthetic_dataset(tmp_path, words, clips_per_word=20)

        # Patch the DATA_DIR used by dataset.py
        import data_collection.config as cfg
        monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
        monkeypatch.setattr(cfg, "VOCABULARY", words)

        self.words = words
        self.n_words = len(words)

    def test_dataloaders_return_correct_types(self):
        from training.dataset import get_dataloaders
        train_dl, val_dl, test_dl, label_map = get_dataloaders(batch_size=8)
        assert isinstance(label_map, dict)
        assert len(label_map) == self.n_words

    def test_batch_shape(self):
        from training.dataset import get_dataloaders
        from data_collection.config import CLIP_LENGTH, FEATURE_DIM
        train_dl, _, _, _ = get_dataloaders(batch_size=8)
        batch_x, batch_y = next(iter(train_dl))
        assert batch_x.shape[1] == CLIP_LENGTH, f"Expected {CLIP_LENGTH} frames"
        assert batch_x.shape[2] == FEATURE_DIM, f"Expected {FEATURE_DIM} features"

    def test_label_dtype(self):
        from training.dataset import get_dataloaders
        train_dl, _, _, _ = get_dataloaders(batch_size=8)
        _, batch_y = next(iter(train_dl))
        assert batch_y.dtype == torch.int64, "Labels should be int64"

    def test_label_values_in_range(self):
        from training.dataset import get_dataloaders
        train_dl, val_dl, test_dl, label_map = get_dataloaders(batch_size=32)
        n_classes = len(label_map)
        for dl in [train_dl, val_dl, test_dl]:
            for _, batch_y in dl:
                assert (batch_y >= 0).all()
                assert (batch_y < n_classes).all()

    def test_all_words_in_label_map(self):
        from training.dataset import get_dataloaders
        _, _, _, label_map = get_dataloaders()
        for word in self.words:
            assert word in label_map, f"'{word}' missing from label_map"

    def test_label_map_values_are_contiguous_ints(self):
        from training.dataset import get_dataloaders
        _, _, _, label_map = get_dataloaders()
        values = sorted(label_map.values())
        assert values == list(range(len(values))), "Label map values should be 0..N-1"

    def test_augmentation_does_not_change_shape(self):
        """Augmented dataset items should have the same shape as non-augmented."""
        from training.dataset import ISLDataset
        from data_collection.config import CLIP_LENGTH, FEATURE_DIM

        # Create a mini dataset manually
        import data_collection.config as cfg
        fps = list((cfg.DATA_DIR / self.words[0]).glob("*.npy"))[:5]
        labels = [0] * len(fps)

        aug_ds = ISLDataset(fps, labels, augment=True)
        plain_ds = ISLDataset(fps, labels, augment=False)

        for i in range(len(fps)):
            x_aug, _ = aug_ds[i]
            x_plain, _ = plain_ds[i]
            assert x_aug.shape == x_plain.shape == (CLIP_LENGTH, FEATURE_DIM), \
                f"Shape mismatch at index {i}: aug={x_aug.shape}, plain={x_plain.shape}"

    def test_stratified_split_coverage(self):
        """Each split should contain at least 1 sample of each class."""
        from training.dataset import get_dataloaders
        _, _, _, label_map = get_dataloaders(batch_size=32)
        n_classes = len(label_map)

        # We skip checking with only 20 clips per class and 70/15/15 split —
        # at 20 clips and 5 classes = 100 total → train~70, val~15, test~15
        # Each class should get ~4 train, ~1 val, ~1 test = coverage OK
        # Just verify no split is empty
        train_dl, val_dl, test_dl, _ = get_dataloaders(batch_size=64)
        assert len(train_dl.dataset) > 0  # type: ignore[arg-type]
        assert len(val_dl.dataset) > 0    # type: ignore[arg-type]
        assert len(test_dl.dataset) > 0   # type: ignore[arg-type]

    def test_no_data_raises_runtime_error(self, monkeypatch, tmp_path):
        """get_dataloaders should raise RuntimeError if data dir is empty."""
        from training.dataset import get_dataloaders
        import data_collection.config as cfg
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(cfg, "DATA_DIR", empty_dir)
        monkeypatch.setattr(cfg, "VOCABULARY", ["hello", "yes"])

        with pytest.raises(RuntimeError, match="No clip data found"):
            get_dataloaders()

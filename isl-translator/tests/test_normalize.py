"""
test_normalize.py — Unit tests for data_collection/normalize.py

These tests guard against the most common silent-failure bugs:
  - Normalization not being translation-invariant
  - Normalization not being scale-invariant
  - Padding/truncation producing wrong shapes
  - Zero-vector (absent hand) not passing through unchanged

Run:
    pytest tests/test_normalize.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow running tests from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.normalize import (
    normalize_frame,
    pad_or_truncate,
    normalize_sequence,
    wrist_velocity,
    extract_keypoints,
)
from data_collection.config import FEATURE_DIM, CLIP_LENGTH


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_frame(seed: int = 0) -> np.ndarray:
    """Create a random (225,) keypoint frame with valid shoulder positions."""
    rng = np.random.default_rng(seed)
    frame = rng.uniform(-1.0, 1.0, size=(FEATURE_DIM,)).astype(np.float32)

    # Set left shoulder (landmark 11, idx 33-35) and right shoulder (12, idx 36-38)
    # to positions that produce a shoulder_width > 0.1
    frame[11 * 3: 11 * 3 + 3] = [0.4, 0.5, 0.0]   # left shoulder
    frame[12 * 3: 12 * 3 + 3] = [0.6, 0.5, 0.0]   # right shoulder
    return frame


# ── normalize_frame tests ──────────────────────────────────────────────────────

class TestNormalizeFrame:
    def test_output_shape(self):
        frame = _make_frame()
        result = normalize_frame(frame)
        assert result.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {result.shape}"

    def test_output_dtype(self):
        frame = _make_frame()
        result = normalize_frame(frame)
        assert result.dtype == np.float32

    def test_translation_invariance(self):
        """Shifting the entire signer's position should not change the output."""
        frame = _make_frame(seed=1)

        # Build a shifted frame: shift ALL landmarks by (+0.3, -0.2) in x,y.
        # Since we shift every landmark uniformly (including the reference
        # shoulders), normalization subtracts the new midpoint and the result
        # should be identical to the original.
        shifted_frame = frame.copy()
        n_landmarks = FEATURE_DIM // 3
        for i in range(n_landmarks):
            shifted_frame[i * 3]     += 0.3
            shifted_frame[i * 3 + 1] -= 0.2

        norm_original = normalize_frame(frame)
        norm_shifted  = normalize_frame(shifted_frame)
        np.testing.assert_allclose(
            norm_original, norm_shifted, atol=1e-5,
            err_msg="normalize_frame is NOT translation-invariant"
        )

    def test_scale_invariance(self):
        """Scaling the entire frame by a positive scalar should not change the output."""
        frame = _make_frame(seed=2)
        scaled_frame = frame * 2.5

        norm_original = normalize_frame(frame)
        norm_scaled = normalize_frame(scaled_frame)
        np.testing.assert_allclose(
            norm_original, norm_scaled, atol=1e-5,
            err_msg="normalize_frame is NOT scale-invariant"
        )

    def test_zero_frame_passthrough(self):
        """A fully zero frame (no pose detected) should pass through unchanged."""
        frame = np.zeros(FEATURE_DIM, dtype=np.float32)
        result = normalize_frame(frame)
        np.testing.assert_array_equal(
            result, frame,
            err_msg="Zero frame (no pose) should pass through as zeros"
        )

    def test_shoulder_midpoint_is_origin(self):
        """After normalization, the shoulder midpoint should be near (0, 0)."""
        frame = _make_frame(seed=3)
        result = normalize_frame(frame)
        pose_block = result[:99].reshape(33, 3)
        midpoint_xy = (pose_block[11, :2] + pose_block[12, :2]) / 2.0
        np.testing.assert_allclose(
            midpoint_xy, [0.0, 0.0], atol=1e-5,
            err_msg="Shoulder midpoint should be at origin after normalization"
        )

    def test_shoulder_width_is_one(self):
        """After normalization, shoulder width (x,y distance) should be ~1.0."""
        frame = _make_frame(seed=4)
        result = normalize_frame(frame)
        pose_block = result[:99].reshape(33, 3)
        width = float(np.linalg.norm(pose_block[11, :2] - pose_block[12, :2]))
        assert abs(width - 1.0) < 1e-5, f"Shoulder width should be 1.0, got {width:.6f}"


# ── pad_or_truncate tests ─────────────────────────────────────────────────────

class TestPadOrTruncate:
    def test_exact_length(self):
        seq = np.ones((CLIP_LENGTH, FEATURE_DIM), dtype=np.float32)
        result = pad_or_truncate(seq, CLIP_LENGTH)
        assert result.shape == (CLIP_LENGTH, FEATURE_DIM)
        np.testing.assert_array_equal(result, seq)

    def test_padding_short_sequence(self):
        """Short sequences should be right-padded with zeros."""
        T = 20
        seq = np.ones((T, FEATURE_DIM), dtype=np.float32)
        result = pad_or_truncate(seq, CLIP_LENGTH)
        assert result.shape == (CLIP_LENGTH, FEATURE_DIM)
        np.testing.assert_array_equal(result[:T], seq)
        np.testing.assert_array_equal(result[T:], 0.0)

    def test_truncation_long_sequence(self):
        """Long sequences should be truncated, keeping the LAST N frames."""
        T = 80
        seq = np.arange(T, dtype=np.float32).reshape(T, 1).repeat(FEATURE_DIM, axis=1)
        result = pad_or_truncate(seq, CLIP_LENGTH)
        assert result.shape == (CLIP_LENGTH, FEATURE_DIM)
        # Last CLIP_LENGTH frames should be the end of the original
        expected_start_frame = T - CLIP_LENGTH
        np.testing.assert_array_equal(
            result[0, 0], float(expected_start_frame),
            err_msg="Truncation should keep the LAST target_length frames"
        )

    def test_output_dtype_is_float32(self):
        seq = np.ones((30, FEATURE_DIM), dtype=np.float64)
        result = pad_or_truncate(seq, CLIP_LENGTH)
        assert result.dtype == np.float32

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            pad_or_truncate(np.ones((FEATURE_DIM,)), CLIP_LENGTH)  # 1D input

    def test_single_frame(self):
        """Edge case: 1-frame sequence should pad to full length."""
        seq = np.ones((1, FEATURE_DIM), dtype=np.float32) * 5.0
        result = pad_or_truncate(seq, CLIP_LENGTH)
        assert result.shape == (CLIP_LENGTH, FEATURE_DIM)
        np.testing.assert_array_equal(result[0], 5.0)
        np.testing.assert_array_equal(result[1:], 0.0)


# ── normalize_sequence tests ──────────────────────────────────────────────────

class TestNormalizeSequence:
    def test_output_shape(self):
        frames = [_make_frame(seed=i) for i in range(30)]
        result = normalize_sequence(frames)
        assert result.shape == (30, FEATURE_DIM)

    def test_output_dtype(self):
        frames = [_make_frame(seed=i) for i in range(10)]
        result = normalize_sequence(frames)
        assert result.dtype == np.float32

    def test_each_frame_normalized(self):
        """Each frame in the output should be individually normalized."""
        frames = [_make_frame(seed=i) for i in range(5)]
        result = normalize_sequence(frames)
        for i, frame in enumerate(frames):
            expected = normalize_frame(frame)
            np.testing.assert_allclose(
                result[i], expected, atol=1e-6,
                err_msg=f"Frame {i} was not correctly normalized in sequence"
            )


# ── wrist_velocity tests ──────────────────────────────────────────────────────

class TestWristVelocity:
    def test_identical_frames_zero_velocity(self):
        frame = _make_frame(seed=5)
        norm = normalize_frame(frame)
        vel = wrist_velocity(norm, norm)
        assert vel == pytest.approx(0.0, abs=1e-6)

    def test_velocity_is_non_negative(self):
        frame_a = normalize_frame(_make_frame(seed=6))
        frame_b = normalize_frame(_make_frame(seed=7))
        vel = wrist_velocity(frame_a, frame_b)
        assert vel >= 0.0

    def test_symmetry(self):
        """wrist_velocity(a, b) should equal wrist_velocity(b, a)."""
        frame_a = normalize_frame(_make_frame(seed=8))
        frame_b = normalize_frame(_make_frame(seed=9))
        assert wrist_velocity(frame_a, frame_b) == pytest.approx(
            wrist_velocity(frame_b, frame_a), abs=1e-6
        )

"""
normalize.py — Keypoint normalization utilities for ISL Translator.

These functions are the most bug-prone part of the pipeline — small errors here
silently corrupt the entire dataset. They are unit-tested in tests/test_normalize.py.

Normalization strategy
----------------------
ISL signs must be recognized regardless of:
  - Signer distance from camera (scale)
  - Signer position in frame (translation)

Solution:
  1. Translate: subtract the shoulder midpoint so the torso is centred at origin.
  2. Scale: divide by shoulder width so coordinates are unitless and scale-invariant.
  3. Missing landmarks (hand not in frame): fill with zeros — the model learns to
     treat zero-vectors as "hand absent", which is meaningful signal.

Reference points (MediaPipe Pose indices):
  - Left shoulder  = pose landmark 11
  - Right shoulder = pose landmark 12
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

# MediaPipe pose landmark indices used as normalization anchors
_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12

# ── Extraction ─────────────────────────────────────────────────────────────────

def extract_keypoints(results: Any) -> np.ndarray:
    """
    Extract a flat (FEATURE_DIM,) keypoint vector from a MediaPipe Holistic result.

    Landmark groups included:
      - Pose (33 × 3 = 99 values)
      - Left hand (21 × 3 = 63 values)
      - Right hand (21 × 3 = 63 values)

    Missing hands are represented as zero vectors (not NaN) so downstream
    consumers don't need special-case handling.

    Args:
        results: mediapipe.solutions.holistic.Holistic result object.

    Returns:
        np.ndarray of shape (225,), dtype float32.
    """
    # Pose landmarks
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()  # (99,)
    else:
        pose = np.zeros(33 * 3, dtype=np.float32)

    # Left hand
    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()  # (63,)
    else:
        left_hand = np.zeros(21 * 3, dtype=np.float32)

    # Right hand
    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()  # (63,)
    else:
        right_hand = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, left_hand, right_hand])  # (225,)


def normalize_frame(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize a single frame's keypoint vector for scale and translation invariance.

    Steps:
      1. Extract left/right shoulder positions from pose block.
      2. Compute shoulder midpoint (reference origin).
      3. Compute shoulder width (reference scale). If shoulders are too close
         together (< 1e-6) — e.g. pose not detected — skip normalization and
         return the raw vector to avoid divide-by-zero.
      4. Translate all coordinates by subtracting the shoulder midpoint.
      5. Scale all coordinates by dividing by shoulder width.

    Note: only x,y coords are translated/scaled; z (depth) is scaled only
    (no meaningful z-origin to subtract since MediaPipe's z is relative to hips).

    Args:
        keypoints: np.ndarray of shape (225,), dtype float32.

    Returns:
        Normalized np.ndarray of shape (225,), dtype float32.
    """
    kp = keypoints.copy()

    # Pose block is the first 99 values, laid out as [x0,y0,z0, x1,y1,z1, ...]
    pose_block = kp[:99].reshape(33, 3)

    left_shoulder = pose_block[_LEFT_SHOULDER_IDX]    # (x, y, z)
    right_shoulder = pose_block[_RIGHT_SHOULDER_IDX]

    # Reference origin: midpoint between shoulders
    midpoint = (left_shoulder + right_shoulder) / 2.0  # (3,)

    # Reference scale: Euclidean distance between shoulders (x,y plane)
    shoulder_width = float(
        np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    )

    if shoulder_width < 1e-6:
        # Pose not detected reliably; return as-is (zeros remain zeros)
        return kp

    # Reshape full vector into (N, 3) for vectorized ops
    n_landmarks = len(kp) // 3
    all_landmarks = kp.reshape(n_landmarks, 3)

    # Translate x,y by shoulder midpoint; leave z unshifted
    all_landmarks[:, :2] -= midpoint[:2]

    # Scale everything by shoulder width
    all_landmarks /= shoulder_width

    return all_landmarks.flatten().astype(np.float32)


def is_pose_present(frame: np.ndarray, atol: float = 1e-9) -> bool:
    """
    Detect whether a keypoint frame actually contains a detected pose, or is
    the all-zero "nothing detected" placeholder produced by extract_keypoints()
    when MediaPipe found no person in view.

    This works on BOTH raw and normalize_frame()-processed vectors: when no
    pose is detected, extract_keypoints() emits zeros for the pose block, and
    normalize_frame() explicitly passes zero pose blocks straight through
    (shoulder_width < 1e-6 early-return), so the shoulder landmarks stay
    exactly (0, 0, 0) in either case. A genuinely normalized frame can never
    have both shoulder landmarks sitting exactly at the origin, since
    normalization forces the shoulder midpoint to (0, 0) and the shoulders
    themselves to be offset by ±half the (nonzero) shoulder width.

    Fixes: gesture-boundary logic previously treated "no person in frame" as
    just another idle frame (zero-to-zero velocity), which could trigger a
    gesture boundary on a truncated, mostly-empty sequence and emit a
    confident-looking but meaningless prediction if a user stepped out of
    frame mid-sign.

    Args:
        frame: (225,) keypoint vector, raw or normalized.
        atol: absolute tolerance for the "is it exactly zero" check.

    Returns:
        True if a pose appears to be present, False if this looks like an
        empty/no-detection frame.
    """
    left_shoulder = frame[_LEFT_SHOULDER_IDX * 3: _LEFT_SHOULDER_IDX * 3 + 3]
    right_shoulder = frame[_RIGHT_SHOULDER_IDX * 3: _RIGHT_SHOULDER_IDX * 3 + 3]
    return not (
        np.allclose(left_shoulder, 0.0, atol=atol)
        and np.allclose(right_shoulder, 0.0, atol=atol)
    )


# ── Sequence Processing ────────────────────────────────────────────────────────

def pad_or_truncate(
    sequence: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad or truncate a keypoint sequence to exactly `target_length` frames.

    Padding strategy: pad at the END with `pad_value` rows (post-padding).
    Truncation strategy: keep the LAST `target_length` frames (captures the
    gesture end which is often the most informative part of a sign).

    Args:
        sequence: np.ndarray of shape (T, feature_dim) where T may vary.
        target_length: desired number of frames.
        pad_value: value used for padding (default 0.0).

    Returns:
        np.ndarray of shape (target_length, feature_dim), dtype float32.
    """
    if sequence.ndim != 2:
        raise ValueError(
            f"Expected 2D array (frames, features), got shape {sequence.shape}"
        )

    T, feature_dim = sequence.shape
    result = np.full((target_length, feature_dim), pad_value, dtype=np.float32)

    if T >= target_length:
        # Truncate: take the last target_length frames
        result = sequence[-target_length:].astype(np.float32)
    else:
        # Pad: place sequence at the start, zeros at the end
        result[:T] = sequence.astype(np.float32)

    return result


def normalize_sequence(frames: list[np.ndarray]) -> np.ndarray:
    """
    Apply per-frame normalization to a list of raw keypoint frames and stack
    into a 2D array.

    Args:
        frames: list of (225,) arrays, one per captured frame.

    Returns:
        np.ndarray of shape (len(frames), 225), dtype float32.
    """
    return np.stack([normalize_frame(f) for f in frames], axis=0)


# ── Hand Motion Detection (used by inference sliding window) ──────────────────

def wrist_velocity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Compute the average wrist velocity between two consecutive normalized frames.

    Uses MediaPipe pose landmark 15 (left wrist) and 16 (right wrist).

    Args:
        frame_a: normalized keypoint vector (225,) at time t
        frame_b: normalized keypoint vector (225,) at time t+1

    Returns:
        float: mean L2 distance of the two wrists between frames.
    """
    pose_a = frame_a[:99].reshape(33, 3)
    pose_b = frame_b[:99].reshape(33, 3)

    left_vel = float(np.linalg.norm(pose_a[15, :2] - pose_b[15, :2]))
    right_vel = float(np.linalg.norm(pose_a[16, :2] - pose_b[16, :2]))
    return (left_vel + right_vel) / 2.0

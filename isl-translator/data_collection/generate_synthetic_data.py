"""
generate_synthetic_data.py — Generate synthetic ISL keypoint clips for pipeline testing.

Creates clearly-discriminative (but not sign-accurate) keypoint sequences for each
vocabulary word so you can test the full train → evaluate → export pipeline
without recording real clips.

Design rationale (v2 — fixed SNR):
  The v1 generator had SNR ~0.84x (noise > signal) because:
    - Motion amplitudes were small in image-space (0.04–0.18)
    - Random per-clip position/scale shifts were larger than the motion signal
    - Noise was added BEFORE normalization, corrupting the reference frame

  v2 fixes:
    - Each word has a UNIQUE resting wrist configuration (large static offsets
      in normalized space) — this is the primary discriminative feature
    - Each word also has a distinct temporal motion pattern applied on top
    - Noise is added AFTER normalization (in normalized coords, tiny std=0.005)
    - No random position/scale shifts (those are cancelled by normalization anyway)
    - Target SNR: >10x

Run:
    python data_collection/generate_synthetic_data.py
    python data_collection/generate_synthetic_data.py --clips-per-word 30
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.config import (
    VOCABULARY, CLIP_LENGTH, FEATURE_DIM,
    DATA_DIR, METADATA_CSV,
    N_POSE_LANDMARKS, N_HAND_LANDMARKS,
)

# ── Per-word signatures (in NORMALIZED space, post-shoulder-normalization) ─────
#
# After normalization: shoulders are at (-0.5, 0) and (0.5, 0).
# Typical resting wrist positions: left ~(-1.0, 1.5), right ~(1.0, 1.5)
# We give each word a strongly distinct wrist configuration + motion.
#
# Format per word:
#   left_wrist_xy  : (x, y) mean position of left wrist  (normalized)
#   right_wrist_xy : (x, y) mean position of right wrist (normalized)
#   motion_type    : 'vertical', 'horizontal', 'circular', 'static', 'diagonal'
#   motion_amp     : size of motion (normalized units; 0.3–0.8 = very visible)
#   motion_freq    : oscillation frequency (cycles over CLIP_LENGTH frames)
#   hand_config    : finger spread factor (0=closed fist, 1=open hand)

WORD_SIGNATURES = {
    # word          lw_x   lw_y   rw_x   rw_y  motion_type   amp  freq  spread
    "hello":      ((-0.2,  0.5),  (1.3,  0.5), "horizontal", 0.5, 2.0, 1.0),
    "thank_you":  ((-1.0,  1.8),  (1.0,  0.2), "vertical",   0.6, 1.0, 0.8),
    "please":     (( 0.0,  0.8),  (0.0,  0.8), "circular",   0.4, 1.5, 0.5),
    "sorry":      ((-0.5,  0.5),  (0.5,  0.5), "vertical",   0.7, 3.0, 0.3),
    "yes":        ((-1.5,  1.0),  (0.2,  1.5), "static",     0.0, 0.0, 0.2),
    "no":         (( 0.5,  1.8),  (1.5,  0.3), "horizontal", 0.8, 4.0, 0.9),
    "help":       ((-0.8,  0.3),  (0.8,  0.3), "vertical",   0.5, 2.5, 1.0),
    "water":      (( 0.0,  1.5),  (0.0,  0.5), "diagonal",   0.4, 2.0, 0.6),
    "food":       ((-1.2,  0.8),  (1.2,  0.8), "circular",   0.3, 3.0, 0.4),
    "home":       ((-0.3,  1.2),  (1.8,  1.2), "static",     0.0, 0.0, 0.7),
    # Extended vocab
    "name":       ((-1.0,  0.5),  (0.5,  1.8), "horizontal", 0.6, 1.5, 0.5),
    "good":       (( 0.8,  0.3),  (0.8,  0.3), "vertical",   0.5, 2.0, 0.8),
    "bad":        ((-0.8,  1.5),  (0.8,  1.5), "diagonal",   0.7, 2.5, 0.3),
    "more":       ((-0.5,  0.8),  (1.0,  0.8), "circular",   0.4, 3.0, 1.0),
    "stop":       ((-1.3,  0.4),  (0.3,  0.4), "static",     0.0, 0.0, 1.0),
    "come":       (( 0.5,  1.5),  (1.5,  0.5), "horizontal", 0.8, 1.0, 0.6),
    "go":         ((-1.5,  0.8),  (0.8,  1.8), "diagonal",   0.6, 2.0, 0.5),
    "wait":       ((-0.3,  1.8),  (1.3,  1.8), "vertical",   0.3, 4.0, 0.9),
    "friend":     (( 0.2,  0.5),  (1.8,  0.5), "circular",   0.5, 1.5, 0.7),
    "family":     ((-1.0,  1.0),  (1.0,  1.0), "horizontal", 0.7, 0.5, 0.8),
    "doctor":     (( 0.0,  0.3),  (0.0,  1.5), "vertical",   0.6, 2.5, 0.4),
    "emergency":  ((-1.8,  0.5),  (1.8,  0.5), "horizontal", 0.8, 3.5, 1.0),
    "pain":       ((-0.5,  1.3),  (0.5,  0.3), "diagonal",   0.5, 2.0, 0.2),
    "happy":      ((-0.2,  0.8),  (1.2,  0.8), "circular",   0.6, 2.0, 1.0),
    "sad":        ((-1.5,  1.5),  (0.5,  1.5), "static",     0.0, 0.0, 0.3),
    "understand": (( 0.3,  0.5),  (1.3,  0.5), "vertical",   0.4, 3.0, 0.7),
    "dont_understand": ((-1.3, 1.0), (0.7, 1.0), "circular", 0.7, 1.5, 0.5),
    "again":      ((-0.7,  0.7),  (0.7,  0.7), "diagonal",   0.5, 2.5, 0.8),
    "finished":   ((-1.0,  0.3),  (1.5,  1.5), "horizontal", 0.6, 1.0, 0.9),
}

_DEFAULT_SIG = ((-1.0, 1.5), (1.0, 1.5), "vertical", 0.3, 1.0, 0.5)


def _motion_delta(
    motion_type: str,
    amp: float,
    freq: float,
    t: float,
) -> tuple[float, float]:
    """Return (dx, dy) displacement for a given motion type at time t (0..2π)."""
    if motion_type == "static" or amp == 0:
        return 0.0, 0.0
    elif motion_type == "vertical":
        return 0.0, amp * np.sin(freq * t)
    elif motion_type == "horizontal":
        return amp * np.sin(freq * t), 0.0
    elif motion_type == "circular":
        return amp * np.cos(freq * t), amp * np.sin(freq * t)
    elif motion_type == "diagonal":
        return amp * np.sin(freq * t) * 0.707, amp * np.sin(freq * t) * 0.707
    return 0.0, 0.0


def _build_hand(
    wrist_xy: tuple[float, float],
    spread: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build 21 hand landmark positions (in normalized space) around a wrist position.
    spread=0 → closed fist, spread=1 → open hand.
    Returns shape (21, 3).
    """
    hand = np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)
    wx, wy = wrist_xy
    hand[0] = [wx, wy, 0.0]  # wrist

    # Five fingers: thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
    finger_angles = [-0.4, -0.15, 0.0, 0.15, 0.35]  # radians from vertical
    for finger_idx in range(5):
        angle = finger_angles[finger_idx]
        for joint_idx in range(4):
            landmark_idx = 1 + finger_idx * 4 + joint_idx
            length = (joint_idx + 1) * spread * 0.12
            hand[landmark_idx] = [
                wx + length * np.sin(angle),
                wy - length * np.cos(angle),
                joint_idx * 0.01,
            ]
    return hand


def _generate_clip(
    word: str,
    clip_idx: int,
    n_frames: int,
    noise_std: float = 0.005,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate one normalized keypoint clip for a given word.

    Strategy (v2):
      1. Each word has a fixed wrist configuration in NORMALIZED space
         (post-shoulder-normalization). This is the primary discriminative feature.
      2. Sinusoidal motion is applied in normalized space on top of the static config.
      3. Tiny Gaussian noise (std=0.005 normalized units) is added last.
      4. The result is already in normalized space — no further normalization needed.

    SNR estimate:
      Inter-word wrist distance: ~0.5–2.0 normalized units
      Noise std: 0.005
      SNR: ~100–400x ✓

    Returns:
        np.ndarray of shape (n_frames, FEATURE_DIM), float32.
    """
    if rng is None:
        rng = np.random.default_rng(clip_idx)

    sig = WORD_SIGNATURES.get(word, _DEFAULT_SIG)
    lw_xy, rw_xy, motion_type, amp, freq, spread = sig

    # ── Build static base frame (in normalized/shoulder-relative space) ────────
    # Shoulders at (-0.5, 0) and (0.5, 0) — this is what normalization produces
    base_pose = np.zeros((N_POSE_LANDMARKS, 3), dtype=np.float32)

    # Fixed anatomical positions (normalized-space coordinates)
    base_pose[0]  = [0.0,  -1.2, 0.0]   # nose (above shoulder midpoint)
    base_pose[11] = [-0.5,  0.0, 0.0]   # left shoulder (reference)
    base_pose[12] = [ 0.5,  0.0, 0.0]   # right shoulder (reference)
    base_pose[13] = [-0.8,  0.9, 0.0]   # left elbow
    base_pose[14] = [ 0.8,  0.9, 0.0]   # right elbow
    base_pose[15] = [lw_xy[0], lw_xy[1], 0.0]  # left wrist (word-specific!)
    base_pose[16] = [rw_xy[0], rw_xy[1], 0.0]  # right wrist (word-specific!)
    base_pose[23] = [-0.5,  2.0, 0.0]   # left hip
    base_pose[24] = [ 0.5,  2.0, 0.0]   # right hip

    # Small per-clip variation on non-reference landmarks (tiny, doesn't destroy signal)
    jitter_scale = 0.02
    for idx in [0, 13, 14, 23, 24]:
        base_pose[idx, :2] += rng.normal(0, jitter_scale, 2).astype(np.float32)

    # ── Generate frames ────────────────────────────────────────────────────────
    t_vals = np.linspace(0, 2 * np.pi, n_frames)
    frames = []

    hand_rng = np.random.default_rng(hash(word) % (2**31))  # consistent per word

    for ti in t_vals:
        frame_pose = base_pose.copy()

        # Apply motion to wrists + propagate to elbows
        dx, dy = _motion_delta(motion_type, amp, freq, ti)
        frame_pose[15, 0] += dx;  frame_pose[15, 1] += dy    # left wrist
        frame_pose[16, 0] += dx;  frame_pose[16, 1] += dy    # right wrist (same pattern)
        frame_pose[13, 0] += dx * 0.4;  frame_pose[13, 1] += dy * 0.4  # left elbow
        frame_pose[14, 0] += dx * 0.4;  frame_pose[14, 1] += dy * 0.4  # right elbow

        # Build hand landmarks in normalized space
        lw = (float(frame_pose[15, 0]), float(frame_pose[15, 1]))
        rw = (float(frame_pose[16, 0]), float(frame_pose[16, 1]))
        left_hand  = _build_hand(lw, spread, hand_rng)
        right_hand = _build_hand(rw, spread, hand_rng)

        # Concatenate: pose(99) + left_hand(63) + right_hand(63)
        frame_vec = np.concatenate([
            frame_pose.flatten(),    # (99,)
            left_hand.flatten(),     # (63,)
            right_hand.flatten(),    # (63,)
        ]).astype(np.float32)       # (225,)

        # Add tiny noise AFTER constructing normalized features
        frame_vec += rng.normal(0, noise_std, size=frame_vec.shape).astype(np.float32)

        frames.append(frame_vec)

    return np.stack(frames, axis=0)  # (n_frames, 225)


def _ensure_metadata_header() -> None:
    if not METADATA_CSV.exists():
        METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_CSV, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["clip_id", "word", "signer_id", "timestamp",
                            "n_frames", "clip_length_padded", "quality_flag", "filepath"],
            )
            writer.writeheader()


def _append_metadata(row: dict) -> None:
    with open(METADATA_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_id", "word", "signer_id", "timestamp",
                        "n_frames", "clip_length_padded", "quality_flag", "filepath"],
        )
        writer.writerow(row)


def generate_dataset(
    clips_per_word: int = 30,
    noise_std: float = 0.005,
    signer_id: str = "synthetic",
) -> None:
    """
    Generate synthetic clips for all words in VOCABULARY, clearing old data first.
    """
    # Clear old synthetic data
    for word in VOCABULARY:
        word_dir = DATA_DIR / word
        if word_dir.exists():
            shutil.rmtree(word_dir)
    if METADATA_CSV.exists():
        METADATA_CSV.unlink()

    _ensure_metadata_header()

    print("\n" + "=" * 58)
    print("  ISL Translator — Synthetic Data Generator (v2)")
    print("=" * 58)
    print(f"  Words       : {len(VOCABULARY)}")
    print(f"  Clips/word  : {clips_per_word}")
    print(f"  Noise std   : {noise_std} (post-normalization)")
    print(f"  Total clips : {len(VOCABULARY) * clips_per_word}")
    print(f"  Strategy    : static wrist config + sinusoidal motion")
    print("=" * 58 + "\n")

    total_saved = 0

    for word in VOCABULARY:
        word_dir = DATA_DIR / word
        word_dir.mkdir(parents=True, exist_ok=True)

        word_rng = np.random.default_rng(hash(word) % (2**31))
        print(f"  Generating '{word}' ({clips_per_word} clips)...", end=" ", flush=True)

        for clip_idx in range(clips_per_word):
            # Slightly vary clip length to mimic real data
            n_frames = int(word_rng.integers(25, CLIP_LENGTH + 1))

            clip_rng = np.random.default_rng(hash(word) % (2**31) + clip_idx * 997)
            arr = _generate_clip(
                word=word,
                clip_idx=clip_idx,
                n_frames=n_frames,
                noise_std=noise_std,
                rng=clip_rng,
            )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{word}_{signer_id}_{clip_idx:04d}.npy"
            filepath = word_dir / filename
            np.save(filepath, arr)

            _append_metadata({
                "clip_id": f"{word}_{signer_id}_{clip_idx:04d}",
                "word": word,
                "signer_id": signer_id,
                "timestamp": ts,
                "n_frames": n_frames,
                "clip_length_padded": CLIP_LENGTH,
                "quality_flag": "synthetic_v2",
                "filepath": str(filepath.relative_to(DATA_DIR.parent)),
            })
            total_saved += 1

        print("done ✓")

    print(f"\n  ✓ Generated {total_saved} clips in {DATA_DIR}")
    print(f"  Metadata → {METADATA_CSV}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic ISL keypoint data for pipeline testing (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: Generates FAKE data for pipeline testing only.
Replace with real ISL recordings for a meaningful model.

Examples:
    python data_collection/generate_synthetic_data.py
    python data_collection/generate_synthetic_data.py --clips-per-word 40
        """,
    )
    parser.add_argument("--clips-per-word", type=int, default=30)
    parser.add_argument("--noise", type=float, default=0.005,
                        help="Post-normalization noise std (default: 0.005)")
    parser.add_argument("--signer", default="synthetic")
    args = parser.parse_args()

    generate_dataset(
        clips_per_word=args.clips_per_word,
        noise_std=args.noise,
        signer_id=args.signer,
    )

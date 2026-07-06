"""
record_clip.py — GUI-based data collection tool for ISL Translator.

Usage:
    python data_collection/record_clip.py

Controls (in the webcam window):
    SPACE   → start / stop recording a clip
    Q       → quit

The GUI shows:
    - Live webcam feed with MediaPipe skeleton overlay
    - Dropdown to select the target word
    - Signer ID entry (used in metadata for multi-signer tracking)
    - Clip counter (how many clips saved for the current word)
    - Recording status indicator (red dot when recording)

Saved output:
    data/<word>/<word>_<signer>_<timestamp>.npy   — (T, 225) normalized array
    data/metadata.csv                              — one row per clip

Design notes:
    - Raw video frames are NEVER written to disk — only normalized keypoint
      arrays. This is intentional: privacy-preserving by design.
    - Clips shorter than MIN_CLIP_FRAMES are discarded with a warning.
    - The MediaPipe model runs locally — no network calls.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow running from project root: python data_collection/record_clip.py
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from data_collection.config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    CLIP_LENGTH, MIN_CLIP_FRAMES, TARGET_FPS,
    DATA_DIR, METADATA_CSV, VOCABULARY,
)
from data_collection.normalize import extract_keypoints, normalize_frame

# ── MediaPipe ─────────────────────────────────────────────────────────────────
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Colours (BGR for OpenCV) ──────────────────────────────────────────────────
COLOUR_BG        = (18, 18, 18)
COLOUR_RED       = (60, 60, 220)
COLOUR_GREEN     = (80, 200, 80)
COLOUR_BLUE      = (220, 100, 60)
COLOUR_YELLOW    = (0, 220, 220)
COLOUR_WHITE     = (240, 240, 240)
COLOUR_GREY      = (130, 130, 130)
COLOUR_RECORDING = (50, 50, 220)   # red dot


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _ensure_metadata_header() -> None:
    """Create metadata.csv with header row if it doesn't exist."""
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


def _count_clips(word: str) -> int:
    word_dir = DATA_DIR / word
    if not word_dir.exists():
        return 0
    return len(list(word_dir.glob("*.npy")))


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_landmarks(frame: np.ndarray, results) -> np.ndarray:
    """Draw MediaPipe skeleton overlay onto a BGR frame."""
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    # Left hand — green
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 160, 0), thickness=2),
        )
    # Right hand — blue
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 100, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(150, 70, 0), thickness=2),
        )
    return frame


def _draw_hud(
    frame: np.ndarray,
    word: str,
    signer_id: str,
    is_recording: bool,
    clip_count: int,
    n_frames_recorded: int,
    message: str = "",
) -> np.ndarray:
    """Overlay HUD information on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Word label
    cv2.putText(frame, f"Word: {word}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOUR_YELLOW, 2)
    # Signer
    cv2.putText(frame, f"Signer: {signer_id}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_GREY, 1)
    # Clip count
    cv2.putText(frame, f"Clips saved: {clip_count}", (w - 170, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_GREEN, 1)

    # Recording indicator
    if is_recording:
        cv2.circle(frame, (w - 20, 38), 10, COLOUR_RECORDING, -1)
        cv2.putText(frame, f"{n_frames_recorded}/{CLIP_LENGTH}", (w - 180, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_RED, 1)
    else:
        status_text = "SPACE=record  Q=quit  N=next word  P=prev word"
        cv2.putText(frame, status_text, (w - 420, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOUR_GREY, 1)

    # Bottom message bar
    if message:
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
        msg_colour = COLOUR_GREEN if "Saved" in message else COLOUR_RED
        cv2.putText(frame, message, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_colour, 2)

    return frame


# ── Main recording loop ───────────────────────────────────────────────────────

def run_recorder(initial_word: Optional[str] = None, signer_id: str = "signer_01") -> None:
    """
    Main function: opens webcam, runs MediaPipe Holistic, captures clips.

    Args:
        initial_word: starting word from VOCABULARY (defaults to first).
        signer_id: identifier for the signer (used in metadata + filename).
    """
    _ensure_metadata_header()

    word_idx = 0
    if initial_word and initial_word in VOCABULARY:
        word_idx = VOCABULARY.index(initial_word)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAMERA_INDEX}. "
              "Try changing CAMERA_INDEX in data_collection/config.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ISL Translator — Data Collection Tool")
    print("=" * 60)
    print(f"  Vocabulary: {VOCABULARY}")
    print(f"  Clip length: {CLIP_LENGTH} frames  |  Signer: {signer_id}")
    print("  Controls: SPACE=record  Q=quit  N/P=next/prev word")
    print("=" * 60 + "\n")

    frames_buffer: list[np.ndarray] = []
    is_recording = False
    message = ""
    message_timer = 0.0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame — retrying...")
                continue

            # Flip horizontally for mirror-like feel (intuitive for self-recording)
            frame = cv2.flip(frame, 1)

            # MediaPipe requires RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Draw skeleton overlay
            frame = _draw_landmarks(frame, results)

            current_word = VOCABULARY[word_idx]
            clip_count = _count_clips(current_word)

            # ── Record frame ───────────────────────────────────────────────
            if is_recording:
                raw_kp = extract_keypoints(results)
                norm_kp = normalize_frame(raw_kp)
                frames_buffer.append(norm_kp)

                # Auto-stop when CLIP_LENGTH reached
                if len(frames_buffer) >= CLIP_LENGTH:
                    is_recording = False
                    message, message_timer = _save_clip(
                        frames_buffer, current_word, signer_id
                    )
                    frames_buffer = []

            # ── Message timeout ────────────────────────────────────────────
            if time.time() - message_timer > 2.5:
                message = ""

            # ── Draw HUD ───────────────────────────────────────────────────
            frame = _draw_hud(
                frame,
                word=current_word,
                signer_id=signer_id,
                is_recording=is_recording,
                clip_count=_count_clips(current_word),  # refresh after possible save
                n_frames_recorded=len(frames_buffer),
                message=message,
            )

            cv2.imshow("ISL Data Collection", frame)

            # ── Keyboard handling ──────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                if not is_recording:
                    # Start recording
                    frames_buffer = []
                    is_recording = True
                    print(f"  [REC] Recording '{current_word}'...")
                else:
                    # Stop early
                    is_recording = False
                    if len(frames_buffer) < MIN_CLIP_FRAMES:
                        message = f"Clip too short ({len(frames_buffer)} frames) — discarded."
                        message_timer = time.time()
                        print(f"  [WARN] {message}")
                        frames_buffer = []
                    else:
                        message, message_timer = _save_clip(
                            frames_buffer, current_word, signer_id
                        )
                        frames_buffer = []

            elif key == ord("n"):
                word_idx = (word_idx + 1) % len(VOCABULARY)
                print(f"  Switched to: {VOCABULARY[word_idx]}")
                is_recording = False
                frames_buffer = []

            elif key == ord("p"):
                word_idx = (word_idx - 1) % len(VOCABULARY)
                print(f"  Switched to: {VOCABULARY[word_idx]}")
                is_recording = False
                frames_buffer = []

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Done] Recording session ended.")
    _print_summary()


def _save_clip(
    frames: list[np.ndarray],
    word: str,
    signer_id: str,
) -> tuple[str, float]:
    """
    Save a recorded clip to disk and append metadata.

    Returns:
        (message_string, timestamp_of_save)
    """
    word_dir = DATA_DIR / word
    word_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{word}_{signer_id}_{ts}.npy"
    filepath = word_dir / filename

    arr = np.stack(frames, axis=0)  # (T, 225)
    np.save(filepath, arr)

    clip_id = f"{word}_{signer_id}_{ts}"
    _append_metadata({
        "clip_id": clip_id,
        "word": word,
        "signer_id": signer_id,
        "timestamp": ts,
        "n_frames": len(frames),
        "clip_length_padded": CLIP_LENGTH,
        "quality_flag": "ok",
        "filepath": str(filepath.relative_to(DATA_DIR.parent)),
    })

    msg = f"Saved: {filename}  ({len(frames)} frames)"
    print(f"  [SAVE] {msg}")
    return msg, time.time()


def _print_summary() -> None:
    """Print a per-word clip count summary to console."""
    print("\n── Dataset summary ──────────────────────────────")
    total = 0
    for word in VOCABULARY:
        count = _count_clips(word)
        bar = "█" * min(count, 30)
        flag = " ✓" if count >= 15 else " ← need more"
        print(f"  {word:<20} {bar:<30} {count:>3}{flag}")
        total += count
    print(f"\n  Total clips: {total}")
    print("─────────────────────────────────────────────────\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ISL Translator — Data Collection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python data_collection/record_clip.py
    python data_collection/record_clip.py --word hello --signer anshika
    python data_collection/record_clip.py --list-words
        """,
    )
    parser.add_argument(
        "--word", "-w",
        choices=VOCABULARY,
        default=None,
        help="Start with this word selected (default: first in vocabulary)",
    )
    parser.add_argument(
        "--signer", "-s",
        default="signer_01",
        help="Signer ID for metadata (default: signer_01)",
    )
    parser.add_argument(
        "--list-words",
        action="store_true",
        help="Print vocabulary list and current clip counts, then exit",
    )
    args = parser.parse_args()

    if args.list_words:
        _print_summary()
        sys.exit(0)

    run_recorder(initial_word=args.word, signer_id=args.signer)

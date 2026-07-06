"""
ingest_videos.py — Extract ISL keypoints from pre-recorded video files.

Instead of recording live, you can provide video files (.mp4, .mov, .avi, .webm)
organized by word, and this script will:
  1. Run MediaPipe Holistic on every frame
  2. Extract + normalize keypoints (identical to the live recorder)
  3. Save .npy clips to data_collection/data/<word>/
  4. Update metadata.csv

Folder layout expected (--input-dir mode):
    videos/
        hello/
            recording1.mp4
            recording2.mov
        thank_you/
            clip1.mp4
        ...

Or use --file + --word for a single file.

Usage:
    # Process an entire folder tree
    python data_collection/ingest_videos.py --input-dir videos/

    # Process one file
    python data_collection/ingest_videos.py --file videos/hello/clip1.mp4 --word hello

    # Dry-run: just show what would be processed
    python data_collection/ingest_videos.py --input-dir videos/ --dry-run

    # Override signer name (written to metadata)
    python data_collection/ingest_videos.py --input-dir videos/ --signer anshika

Requirements:
    pip install mediapipe opencv-python
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.config import (
    VOCABULARY,
    CLIP_LENGTH,
    FEATURE_DIM,
    DATA_DIR,
    METADATA_CSV,
    N_POSE_LANDMARKS,
    N_HAND_LANDMARKS,
    MIN_CLIP_FRAMES,
)
from data_collection.normalize import normalize_frame, pad_or_truncate

# ── Supported video extensions ─────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}

# ── MediaPipe feature extraction ───────────────────────────────────────────────

def _extract_keypoints_from_video(
    video_path: Path,
    mp_holistic,  # mediapipe.solutions.holistic.Holistic instance
) -> np.ndarray | None:
    """
    Run MediaPipe Holistic on a video file and extract raw keypoints per frame.

    Returns:
        np.ndarray of shape (T, FEATURE_DIM) with NORMALIZED keypoints,
        or None if the video had too few valid frames (< MIN_CLIP_FRAMES).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [WARN] Could not open video: {video_path.name}")
        return None

    frames: list[np.ndarray] = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = mp_holistic.process(rgb)
        rgb.flags.writeable = True

        # ── Extract landmarks → flat vector (225,) ────────────────────────────
        raw = np.zeros(FEATURE_DIM, dtype=np.float32)

        # Pose: 33 × 3 = 0..98
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark[:N_POSE_LANDMARKS]):
                raw[i * 3]     = lm.x
                raw[i * 3 + 1] = lm.y
                raw[i * 3 + 2] = lm.z

        # Left hand: 21 × 3 = 99..161
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark[:N_HAND_LANDMARKS]):
                raw[99 + i * 3]     = lm.x
                raw[99 + i * 3 + 1] = lm.y
                raw[99 + i * 3 + 2] = lm.z

        # Right hand: 21 × 3 = 162..224
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark[:N_HAND_LANDMARKS]):
                raw[162 + i * 3]     = lm.x
                raw[162 + i * 3 + 1] = lm.y
                raw[162 + i * 3 + 2] = lm.z

        # Normalize (shoulder-relative, scale-invariant)
        normed = normalize_frame(raw)
        frames.append(normed)

    cap.release()

    if len(frames) < MIN_CLIP_FRAMES:
        print(f"    [SKIP] {video_path.name}: only {len(frames)} valid frames "
              f"(min {MIN_CLIP_FRAMES})")
        return None

    # Pad or truncate to CLIP_LENGTH
    clip = np.stack(frames, axis=0)          # (T, 225)
    clip = pad_or_truncate(clip, CLIP_LENGTH) # (CLIP_LENGTH, 225)
    return clip


# ── Metadata helpers ───────────────────────────────────────────────────────────

def _ensure_metadata_header() -> None:
    if not METADATA_CSV.exists():
        METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_CSV, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["clip_id", "word", "signer_id", "timestamp",
                            "n_frames", "clip_length_padded",
                            "quality_flag", "source_file", "filepath"],
            )
            writer.writeheader()


def _append_metadata(row: dict) -> None:
    with open(METADATA_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_id", "word", "signer_id", "timestamp",
                        "n_frames", "clip_length_padded",
                        "quality_flag", "source_file", "filepath"],
        )
        writer.writerow(row)


# ── Discovery helpers ──────────────────────────────────────────────────────────

def _discover_videos(input_dir: Path) -> dict[str, list[Path]]:
    """
    Walk input_dir looking for <word>/<video_file> structure.
    Only picks up words that are in VOCABULARY (case-insensitive match).

    Returns dict: { word: [video_path, ...] }
    """
    vocab_lower = {w.lower(): w for w in VOCABULARY}
    found: dict[str, list[Path]] = {}

    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        word_key = vocab_lower.get(subdir.name.lower())
        if word_key is None:
            print(f"  [SKIP] Folder '{subdir.name}' not in VOCABULARY — ignoring.")
            continue
        videos = sorted(
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
        if videos:
            found[word_key] = videos

    return found


# ── Main ingestion routine ─────────────────────────────────────────────────────

def ingest(
    video_map: dict[str, list[Path]],
    signer: str = "uploaded",
    dry_run: bool = False,
    start_clip_idx: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Process all videos in video_map and save keypoint clips to DATA_DIR.

    Args:
        video_map: { word: [video_path, ...] }
        signer: signer ID written to metadata
        dry_run: if True, print what would happen but don't save anything
        start_clip_idx: { word: N } — start clip numbering at N (to avoid overwriting)

    Returns:
        { word: clips_saved }
    """
    if not dry_run:
        try:
            import mediapipe as mp
        except ImportError:
            print("\n  ✗ mediapipe not installed. Run:\n"
                  "      pip install mediapipe opencv-python\n")
            sys.exit(1)

        try:
            import cv2  # noqa: F401
        except ImportError:
            print("\n  ✗ opencv-python not installed. Run:\n"
                  "      pip install opencv-python\n")
            sys.exit(1)

        mp_holistic_module = mp.solutions.holistic
        holistic_ctx = mp_holistic_module.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        holistic_ctx = None

    _ensure_metadata_header()

    if start_clip_idx is None:
        # Auto-detect existing clip counts per word
        start_clip_idx = {}
        for word in video_map:
            existing = list((DATA_DIR / word).glob("*.npy")) if (DATA_DIR / word).exists() else []
            start_clip_idx[word] = len(existing)

    total_videos = sum(len(v) for v in video_map.values())
    total_saved  = 0
    total_skipped = 0
    results: dict[str, int] = {}

    print(f"\n{'='*58}")
    print(f"  ISL Translator — Video Ingestion Tool")
    print(f"{'='*58}")
    print(f"  Words found : {len(video_map)}")
    print(f"  Total videos: {total_videos}")
    print(f"  Signer ID   : {signer}")
    if dry_run:
        print(f"  Mode        : DRY RUN (nothing will be saved)")
    print(f"{'='*58}\n")

    try:
        for word, video_paths in video_map.items():
            word_dir = DATA_DIR / word
            if not dry_run:
                word_dir.mkdir(parents=True, exist_ok=True)

            clip_idx = start_clip_idx.get(word, 0)
            saved_this_word = 0

            print(f"  [{word}]  {len(video_paths)} video(s)")

            for vpath in video_paths:
                print(f"    → {vpath.name}  ", end="", flush=True)

                if dry_run:
                    print("(dry run)")
                    saved_this_word += 1
                    continue

                t0 = time.time()
                clip = _extract_keypoints_from_video(vpath, holistic_ctx)
                elapsed = time.time() - t0

                if clip is None:
                    total_skipped += 1
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{word}_{signer}_{clip_idx:04d}.npy"
                save_path = word_dir / filename
                np.save(save_path, clip)

                n_frames = int((clip.any(axis=1)).sum())  # non-zero frames

                _append_metadata({
                    "clip_id":           f"{word}_{signer}_{clip_idx:04d}",
                    "word":              word,
                    "signer_id":         signer,
                    "timestamp":         ts,
                    "n_frames":          n_frames,
                    "clip_length_padded": CLIP_LENGTH,
                    "quality_flag":      "video_ingested",
                    "source_file":       vpath.name,
                    "filepath":          str(save_path.relative_to(DATA_DIR.parent)),
                })

                print(f"✓  ({elapsed:.1f}s)")
                clip_idx   += 1
                saved_this_word += 1
                total_saved += 1

            results[word] = saved_this_word
            print(f"    Saved {saved_this_word} clip(s) for '{word}'\n")

    finally:
        if holistic_ctx is not None:
            holistic_ctx.close()

    print(f"{'='*58}")
    print(f"  Done!  {total_saved} clips saved, {total_skipped} skipped")
    print(f"  Data directory: {DATA_DIR}")
    if not dry_run and total_saved > 0:
        print(f"\n  Next steps:")
        print(f"    python training/train.py")
        print(f"    python training/evaluate.py")
        print(f"    python training/export_onnx.py")
    print(f"{'='*58}\n")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ISL keypoints from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a whole folder tree
  python data_collection/ingest_videos.py --input-dir ~/Desktop/isl_videos/

  # Process a single file
  python data_collection/ingest_videos.py --file clip.mp4 --word hello

  # Dry-run (see what would happen, don't save)
  python data_collection/ingest_videos.py --input-dir videos/ --dry-run

  # Set signer name
  python data_collection/ingest_videos.py --input-dir videos/ --signer anshika

Supported formats: .mp4 .mov .avi .webm .mkv .m4v

Folder structure for --input-dir:
  videos/
    hello/
      clip1.mp4
      clip2.mov
    thank_you/
      recording.mp4
    ...
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir", type=Path,
        help="Root folder containing <word>/<video_file> subfolders",
    )
    group.add_argument(
        "--file", type=Path,
        help="Single video file to ingest",
    )
    parser.add_argument(
        "--word", type=str,
        help="Word label for --file mode (required when using --file)",
    )
    parser.add_argument(
        "--signer", default="uploaded",
        help="Signer ID written to metadata (default: uploaded)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without saving anything",
    )
    args = parser.parse_args()

    # ── Validate and build video map ───────────────────────────────────────────
    if args.file:
        if not args.word:
            parser.error("--word is required when using --file")
        if args.word not in VOCABULARY:
            print(f"[WARN] '{args.word}' is not in VOCABULARY.")
            print(f"       Valid words: {', '.join(VOCABULARY)}")
            print(f"       Add it to data_collection/config.py first, or proceed anyway.")
        if not args.file.exists():
            parser.error(f"File not found: {args.file}")
        video_map = {args.word: [args.file]}

    else:  # --input-dir
        if not args.input_dir.exists():
            parser.error(f"Directory not found: {args.input_dir}")
        video_map = _discover_videos(args.input_dir)
        if not video_map:
            print(f"\n  No videos found in {args.input_dir}")
            print(f"  Make sure subfolders match vocabulary words: {', '.join(VOCABULARY)}")
            sys.exit(1)

    ingest(
        video_map=video_map,
        signer=args.signer,
        dry_run=args.dry_run,
    )

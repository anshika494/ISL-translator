"""
config.py — Central configuration for the ISL Translator project.

This is the SINGLE file to edit when:
  - Adding new vocabulary words
  - Changing clip length / camera settings
  - Adjusting keypoint dimensions

Adding a new word:
  1. Add it to VOCABULARY below (uncomment or append).
  2. Re-run record_clip.py to collect clips for the new word.
  3. Re-run training/train.py — the new class is picked up automatically.
"""

# ── Vocabulary ────────────────────────────────────────────────────────────────
# V1: 10-word starter set for the initial working loop.
# Uncomment additional words to expand the vocabulary (Phase 2 / stretch goal).
#
# NOTE (fixed): "help" previously appeared TWICE in this list. That caused
# _load_all_clips() in training/dataset.py to glob the same data/help/*.npy
# directory twice, duplicating every "help" clip in the dataset and allowing
# the same physical clip to land in both the train and test splits (data
# leakage), which inflated reported accuracy for that class. Keep each word
# unique — there's no test for accidental duplicates, so double-check by eye
# when adding new words.

VOCABULARY = [
    "hello",
    "thank_you",
    "please",
    "sorry",
    "yes",
    "no",
    "help",
    "water",
    "food",
    "home",
    "accident",
    "doctor",
    "call",
    "hot",
    "lose",
    "pain",
    "thief",
    # ── Expand below for 20-30 word model ────────────────────────────────────
    # "name",
    # "good",
    # "bad",
    # "more",
    # "stop",
    # "come",
    # "go",
    # "wait",
    # "friend",
    # "family",
    # "doctor",
    # "emergency",
    # "pain",
    # "happy",
    # "sad",
    # "understand",
    # "dont_understand",
    # "again",
    # "finished",
    # ── Add your own custom words here ───────────────────────────────────────
    # Any word is fine. Use underscores for multi-word signs, e.g. "i_love_you".
    # After adding, upload videos for that word and retrain.
    # "i_love_you",
    # "namaste",
    # "school",
    # "mother",
    # "father",
]

assert len(VOCABULARY) == len(set(VOCABULARY)), (
    "Duplicate word(s) found in VOCABULARY — this silently duplicates clips "
    "in the dataset. Check config.py."
)

# ── Clip Settings ─────────────────────────────────────────────────────────────
# Number of frames per clip (pad/truncate to this length at inference time too)
CLIP_LENGTH: int = 50          # ~1.7 seconds at 30 fps — suits isolated signs

# Target recording FPS (MediaPipe processes at native camera fps; this is used
# as metadata and for display purposes — actual fps may vary slightly)
TARGET_FPS: int = 30

# Minimum frames to keep a clip (clips shorter than this are discarded)
MIN_CLIP_FRAMES: int = 15

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0          # 0 = default/built-in webcam; change for external
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480

# ── Keypoint Dimensions ───────────────────────────────────────────────────────
# MediaPipe Holistic outputs:
#   - Pose:       33 landmarks × 3 coords (x, y, z)  = 99 values
#   - Left hand:  21 landmarks × 3 coords             = 63 values
#   - Right hand: 21 landmarks × 3 coords             = 63 values
#   - (Face mesh omitted — 468 landmarks, too noisy for isolated word signs)
# Total flat feature vector per frame: 225 values
N_POSE_LANDMARKS: int = 33
N_HAND_LANDMARKS: int = 21
N_COORDS: int = 3              # (x, y, z) per landmark

POSE_DIM: int = N_POSE_LANDMARKS * N_COORDS      # 99
HAND_DIM: int = N_HAND_LANDMARKS * N_COORDS      # 63
FEATURE_DIM: int = POSE_DIM + HAND_DIM * 2       # 225  (pose + left + right)

# ── Data Paths ────────────────────────────────────────────────────────────────
import os
from pathlib import Path

# All paths are relative to the isl-translator/ project root
_HERE = Path(__file__).parent                     # data_collection/
PROJECT_ROOT = _HERE.parent                       # isl-translator/

DATA_DIR = _HERE / "data"                         # data_collection/data/
METADATA_CSV = DATA_DIR / "metadata.csv"

CHECKPOINTS_DIR = PROJECT_ROOT / "training" / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "training" / "plots"
ONNX_OUTPUT = PROJECT_ROOT / "frontend" / "public" / "model.onnx"

# Ensure critical directories exist when this config is imported
for _d in [DATA_DIR, CHECKPOINTS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Training Hyperparameters (referenced from training/ scripts) ──────────────
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15        # remainder

BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
MAX_EPOCHS: int = 150
EARLY_STOP_PATIENCE: int = 15

LSTM_HIDDEN_DIM: int = 128
LSTM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.3

RANDOM_SEED: int = 42

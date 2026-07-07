"""
websocket_handler.py — Sliding window keypoint buffer + gesture boundary detection.

This module maintains a rolling buffer of incoming keypoint frames and detects
when a gesture has likely completed (hand motion drops below a threshold),
then triggers inference.

Gesture boundary heuristic:
  - Track mean wrist velocity over the last N frames
  - If velocity drops below IDLE_THRESHOLD for IDLE_FRAMES consecutive frames,
    and we have at least MIN_GESTURE_FRAMES in the active window → gesture done

Fixed: frames where no pose/person is detected at all (e.g. the signer stepped
out of frame) are no longer counted as "idle" motion. Previously, two
consecutive all-zero frames produced a velocity of exactly 0.0, which read as
"hand at rest" — if this happened mid-gesture, it could trigger a boundary on
a truncated, mostly-empty sequence and emit a confident-looking but meaningless
prediction. Now such frames are skipped entirely (not appended to the buffer,
and they don't affect the idle/active counters) until a real pose reappears.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import deque
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_collection.config import CLIP_LENGTH, FEATURE_DIM
from data_collection.normalize import wrist_velocity, is_pose_present

# Tunable heuristic parameters
IDLE_VELOCITY_THRESHOLD = 0.015   # normalized wrist velocity below this = "idle"
IDLE_FRAMES_REQUIRED = 8          # consecutive idle frames needed to trigger boundary
MIN_GESTURE_FRAMES = 10           # don't trigger if too few frames captured
MAX_BUFFER_FRAMES = CLIP_LENGTH * 2  # hard limit to prevent runaway buffer


class GestureBuffer:
    """
    Stateful per-connection sliding window for gesture segmentation.

    Usage:
        buf = GestureBuffer(on_gesture=my_callback)
        # For each incoming frame:
        result = buf.push_frame(keypoint_array)
        # result is None normally, or a prediction dict when a gesture is detected
    """

    def __init__(
        self,
        on_gesture: Callable[[np.ndarray], dict],
        idle_threshold: float = IDLE_VELOCITY_THRESHOLD,
        idle_frames: int = IDLE_FRAMES_REQUIRED,
        min_frames: int = MIN_GESTURE_FRAMES,
    ) -> None:
        """
        Args:
            on_gesture: callback called with (sequence_array) when gesture boundary
                        detected; should return a prediction dict.
            idle_threshold: velocity below this = hand at rest.
            idle_frames: consecutive frames below threshold to trigger boundary.
            min_frames: minimum gesture length to accept.
        """
        self.on_gesture = on_gesture
        self.idle_threshold = idle_threshold
        self.idle_frames = idle_frames
        self.min_frames = min_frames

        self._buffer: deque[np.ndarray] = deque(maxlen=MAX_BUFFER_FRAMES)
        self._idle_count = 0
        self._is_active = False  # True once we've seen motion above threshold
        self._pose_missing = False  # True when the signer is not currently in frame

    def push_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        Add a normalized keypoint frame to the buffer and check for gesture boundary.

        Args:
            frame: np.ndarray of shape (FEATURE_DIM,) — single normalized frame.

        Returns:
            Prediction dict if gesture boundary detected, else None.
        """
        if frame.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected frame shape ({FEATURE_DIM},), got {frame.shape}")

        # Fixed: if no pose is detected at all (signer out of frame / poor
        # lighting / camera obstructed), don't feed this frame into the
        # buffer or the idle/active counters. Treat it as "no signal" rather
        # than "hand at rest", since zero-to-zero velocity would otherwise
        # look identical to a genuinely idle hand.
        if not is_pose_present(frame):
            self._pose_missing = True
            return None
        self._pose_missing = False

        prev_frame = self._buffer[-1] if self._buffer else None
        self._buffer.append(frame)

        if prev_frame is None:
            return None

        # Compute wrist velocity between consecutive frames
        vel = wrist_velocity(prev_frame, frame)

        if vel > self.idle_threshold:
            self._idle_count = 0
            self._is_active = True
        else:
            self._idle_count += 1

        # Gesture boundary: active phase ended, enough frames collected
        if (
            self._is_active
            and self._idle_count >= self.idle_frames
            and len(self._buffer) >= self.min_frames
        ):
            # Extract the gesture frames (excluding the trailing idle frames)
            gesture_frames = list(self._buffer)[:-self.idle_frames]
            if len(gesture_frames) >= self.min_frames:
                sequence = np.stack(gesture_frames, axis=0)
                result = self.on_gesture(sequence)
                self._reset()
                return result

        return None

    def _reset(self) -> None:
        """Clear buffer after a gesture is detected."""
        self._buffer.clear()
        self._idle_count = 0
        self._is_active = False

    @property
    def buffer_length(self) -> int:
        return len(self._buffer)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def pose_missing(self) -> bool:
        """True if the most recently pushed frame had no detected pose."""
        return self._pose_missing

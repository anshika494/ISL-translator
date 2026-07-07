"""
main.py — FastAPI backend for ISL Translator (WebSocket inference fallback).

This backend is used when client-side ONNX inference is not feasible.
The frontend connects via WebSocket, streams keypoint frames, and receives
prediction JSON back.

Endpoints:
    GET  /health         — health check + model info
    WS   /ws/infer       — streaming keypoint inference
    GET  /vocabulary     — returns the vocabulary list

Start:
    uvicorn backend.main:app --reload --port 8000
    # or from project root:
    python -m uvicorn backend.main:app --reload
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.config import VOCABULARY, FEATURE_DIM
from backend.inference import ISLInferenceEngine
from backend.websocket_handler import GestureBuffer

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ISL Translator API",
    description="Real-time Indian Sign Language recognition via WebSocket",
    version="1.0.0",
)

# Allow frontend dev server (localhost:5173) and any deployed origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load inference engine ─────────────────────────────────────────────────
# Loaded on first connection to avoid startup crash if model doesn't exist yet
_engine: ISLInferenceEngine | None = None


def get_engine() -> ISLInferenceEngine:
    global _engine
    if _engine is None:
        _engine = ISLInferenceEngine()
    return _engine


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Health check — also reports whether the ONNX model is loaded."""
    try:
        engine = get_engine()
        model_status = "loaded"
        n_classes = engine.n_classes
    except Exception as e:
        model_status = f"not_loaded: {e}"
        n_classes = 0

    return {
        "status": "ok",
        "model": model_status,
        "n_classes": n_classes,
        "vocabulary_size": len(VOCABULARY),
    }


@app.get("/vocabulary")
async def vocabulary() -> dict:
    """Return the current vocabulary list (word → label index)."""
    return {"vocabulary": VOCABULARY}


# ── WebSocket inference endpoint ───────────────────────────────────────────────

@app.websocket("/ws/infer")
async def ws_infer(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time keypoint streaming inference.

    Protocol:
        Client sends JSON per frame:
            {"type": "frame", "keypoints": [float, ...]}   (225 values)

        Server responds when gesture detected:
            {"type": "prediction", "word": str, "confidence": float,
             "top_k": [{"word": str, "confidence": float}, ...]}

        Client sends to reset:
            {"type": "reset"}
    """
    await websocket.accept()
    engine = get_engine()

    def _on_gesture(sequence: np.ndarray) -> dict:
        return engine.predict(sequence)

    buffer = GestureBuffer(on_gesture=_on_gesture)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            msg_type = msg.get("type")

            if msg_type == "frame":
                kp_list = msg.get("keypoints", [])
                if len(kp_list) != FEATURE_DIM:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Expected {FEATURE_DIM} keypoints, got {len(kp_list)}"
                    })
                    continue

                frame = np.array(kp_list, dtype=np.float32)
                result = buffer.push_frame(frame)

                if result is not None:
                    await websocket.send_json({"type": "prediction", **result})
                else:
                    # Status includes pose_missing so the frontend can show an
                    # accurate "no person detected" state instead of a generic
                    # "tracking active" badge that's only ever based on
                    # whether the model loaded.
                    await websocket.send_json({
                        "type": "status",
                        "buffer_length": buffer.buffer_length,
                        "is_active": buffer.is_active,
                        "pose_missing": buffer.pose_missing,
                    })

            elif msg_type == "reset":
                buffer._reset()
                await websocket.send_json({"type": "reset_ack"})

            else:
                # Fixed: previously unrecognized message types were silently
                # dropped, which made client-side debugging harder than it
                # needed to be.
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unrecognized message type: {msg_type!r}",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

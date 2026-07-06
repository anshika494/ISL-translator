# ISL Translator — Real-Time Indian Sign Language Recognition

> A full-stack, privacy-first web app that recognizes Indian Sign Language (ISL) gestures from webcam video in real time, translates them to text, and optionally speaks them aloud.

---

## Why this project matters

India has an estimated **18+ million deaf/hard-of-hearing people**, and Indian Sign Language (ISL) is linguistically distinct from ASL and BSL — yet almost no consumer tools support it. This project's dual contribution:

1. A working real-time ISL translator that runs entirely in the browser (no server needed)
2. A small, clean, open-sourceable ISL keypoint dataset that others can build on

**Privacy design choice**: all keypoint extraction happens in your browser via MediaPipe. No video frames, no images, and no raw keypoints are ever sent to a server.

---

## Demo

> 📸 *[Add a GIF/screenshot here after recording a demo]*

---

## Architecture

```
Webcam → MediaPipe Holistic (JS, in-browser)
       → Normalized keypoint vector (225 floats/frame)
       → ONNXRuntime-Web (model.onnx, in-browser)
       → Gesture boundary detection (wrist velocity heuristic)
       → Prediction: word + confidence
       → SpeechSynthesis API (optional TTS)
```

```
isl-translator/
├── data_collection/     # Record ISL clips (Python + MediaPipe)
├── training/            # Train BiLSTM classifier (PyTorch)
├── backend/             # FastAPI WebSocket fallback (optional)
├── frontend/            # React + Vite web app (primary path)
├── docs/                # Dataset card + model card
└── tests/               # Unit tests
```

---

## Quick Start

### Prerequisites
- Python ≥ 3.10
- Node.js ≥ 18
- Webcam

### 1. Set up Python environment

```bash
cd isl-translator
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Collect training data

```bash
# List current clip counts
python data_collection/record_clip.py --list-words

# Start recording (change --signer to your name)
python data_collection/record_clip.py --signer your_name
```

**Controls in the webcam window:**
| Key | Action |
|-----|--------|
| `SPACE` | Start / stop recording a clip |
| `N` | Next word |
| `P` | Previous word |
| `Q` | Quit |

**Target**: 15–20 clips per word. Vary lighting and clothing across sessions for better generalization.

### 3. Train the model

```bash
python training/train.py
# Results saved to training/checkpoints/ and training/plots/
```

### 4. Evaluate

```bash
python training/evaluate.py
# Confusion matrix → training/plots/confusion_matrix_bilstm.png
```

### 5. Export to ONNX

```bash
python training/export_onnx.py
# Exports to frontend/public/model.onnx + frontend/public/label_map.json
```

### 6. Run the web app

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Adding a New Sign

1. Open `data_collection/config.py` and uncomment (or add) the word to `VOCABULARY`
2. Run `record_clip.py` and record 15+ clips for the new word
3. Re-run `training/train.py` — the new class is picked up automatically
4. Re-run `export_onnx.py` to update the browser model
5. Refresh the frontend — done

---

## Accuracy

| Metric | Value |
|--------|-------|
| Top-1 test accuracy | *Run evaluate.py after training* |
| Top-3 test accuracy | *Run evaluate.py after training* |
| Vocabulary size | 10 (v1) |

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for full evaluation details and known limitations.

---

## Backend Fallback (Optional)

By default, all inference runs client-side in the browser. If you need server-side inference:

```bash
uvicorn backend.main:app --reload
# WebSocket endpoint: ws://localhost:8000/ws/infer
```

Set `VITE_WS_URL=ws://localhost:8000/ws/infer` in `frontend/.env.local` and update `App.tsx` to use `useWebSocket` instead of the ONNX engine.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dataset

Raw keypoint clips are stored in `data_collection/data/` (gitignored). See [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) for documentation.

**Format**: each clip is a NumPy array of shape `(T, 225)` — `T` frames of normalized keypoints.

---

## Limitations & Ethical Notes

- **Single signer**: v1 is trained on one signer. It will not generalize reliably to other signers, especially across regional ISL dialects.
- **Not a certified tool**: this is a research/portfolio project — not suitable for medical, legal, or safety-critical use. It is not a replacement for professional human interpreters.
- **Regional variation**: ISL has significant regional variation across India. This project addresses none of it in v1.
- **10-word vocabulary**: v1 covers 10 common signs. The full 29-word list is in `config.py`, ready to uncomment once more data is collected.

---

## Roadmap

- [ ] Expand to 20–30 word vocabulary
- [ ] Multi-signer dataset (seeking collaborators)
- [ ] Deploy to Vercel/Netlify (fully client-side, no server needed)
- [ ] Add Transformer encoder baseline for accuracy comparison
- [ ] Continuous phrase recognition (CTC-based)
- [ ] Reach out to ISL community for feedback and validation

---

## License

Code: MIT  
Dataset: CC-BY-NC (see [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md))

---

## Acknowledgements

Built with [MediaPipe](https://mediapipe.dev/), [PyTorch](https://pytorch.org/), [ONNX Runtime Web](https://onnxruntime.ai/), and [React](https://react.dev/).

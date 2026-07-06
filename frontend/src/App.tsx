/**
 * App.tsx — Main application layout for the ISL Translator.
 *
 * Architecture:
 *  - useMediaPipe: loads MediaPipe Holistic, extracts keypoints per frame
 *  - ONNXInferenceEngine: loads model.onnx, detects gesture boundaries,
 *    runs inference client-side (no server needed)
 *  - Webcam: renders video + canvas overlay
 *  - PredictionDisplay: shows predictions, sentence, TTS
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Webcam } from './components/Webcam';
import { KeypointOverlay } from './components/KeypointOverlay';
import { PredictionDisplay } from './components/PredictionDisplay';
import { useMediaPipe } from './hooks/useMediaPipe';
import { inferenceEngine, type Prediction, CLIP_LENGTH } from './lib/onnxInference';

export default function App() {
  const { isReady, error: mpError, latestKeypoints, processFrame } = useMediaPipe();

  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [bufferStatus, setBufferStatus] = useState({ bufferLength: 0, isActive: false });
  const [webcamError, setWebcamError] = useState<string | null>(null);

  const inferenceRunningRef = useRef(false);

  // Load ONNX model
  useEffect(() => {
    inferenceEngine
      .load('/model.onnx', '/label_map.json')
      .then(() => setIsModelLoaded(true))
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        setModelError(`Model not loaded: ${msg}. Train the model and export it first.`);
        console.warn('[App] ONNX model not available — demo mode only.', err);
      });
  }, []);

  // Push keypoints into inference engine on every frame
  useEffect(() => {
    if (!latestKeypoints || !isModelLoaded || inferenceRunningRef.current) return;

    inferenceRunningRef.current = true;
    inferenceEngine
      .pushFrameAsync(latestKeypoints)
      .then(({ prediction: pred, status }) => {
        setBufferStatus(status);
        if (pred) setPrediction(pred);
      })
      .finally(() => {
        inferenceRunningRef.current = false;
      });
  }, [latestKeypoints, isModelLoaded]);

  const handleWebcamFrame = useCallback(
    (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
      processFrame(video, canvas);
    },
    [processFrame]
  );

  const combinedError = webcamError || mpError;

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <span className="header-logo">🤟</span>
            <div>
              <h1 className="header-title">ISL Translator</h1>
              <p className="header-subtitle">Real-Time Indian Sign Language Recognition</p>
            </div>
          </div>
          <div className="header-links">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="header-link"
              id="github-link"
            >
              GitHub
            </a>
            <span className="header-badge">v1 · 10 signs</span>
          </div>
        </div>
      </header>

      {/* ── Error banner ───────────────────────────────────────────────── */}
      {combinedError && (
        <div className="error-banner" role="alert" id="error-banner">
          <span>⚠</span> {combinedError}
        </div>
      )}

      {/* ── Main content ───────────────────────────────────────────────── */}
      <main className="app-main">
        {/* Left: webcam pane */}
        <section className="webcam-pane" aria-label="Webcam feed">
          <div className="pane-header">
            <h2 className="pane-title">Camera</h2>
            <span className={`pane-status ${isReady ? 'pane-status--ok' : 'pane-status--loading'}`}>
              {isReady ? 'MediaPipe Active' : 'Loading MediaPipe…'}
            </span>
          </div>

          <div className="webcam-wrapper">
            <Webcam
              onFrame={handleWebcamFrame}
              isReady={isReady}
              onError={setWebcamError}
            />
            <KeypointOverlay
              isMediaPipeReady={isReady}
              isActive={bufferStatus.isActive}
              bufferLength={bufferStatus.bufferLength}
              maxBuffer={CLIP_LENGTH}
            />
          </div>

          <div className="webcam-legend">
            <span className="legend-item legend-item--blue">● Pose</span>
            <span className="legend-item legend-item--green">● Left hand</span>
            <span className="legend-item legend-item--orange">● Right hand</span>
          </div>

          <div className="webcam-tip">
            💡 Position yourself so your hands and shoulders are visible. Wait for tracking to activate, then sign naturally.
          </div>
        </section>

        {/* Right: prediction panel */}
        <section className="prediction-pane" aria-label="Translation output">
          <div className="pane-header">
            <h2 className="pane-title">Translation</h2>
          </div>
          <PredictionDisplay
            prediction={prediction}
            isModelLoaded={isModelLoaded}
            modelError={modelError}
          />
        </section>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────── */}
      <footer className="app-footer">
        <p>
          Built for India's 18M+ deaf/hard-of-hearing community ·{' '}
          <strong>Disclaimer:</strong> This is a research tool, not a certified interpreter.
          See <a href="/docs/MODEL_CARD.md" className="footer-link">Model Card</a> for limitations.
        </p>
        <p>
          Privacy: keypoint extraction runs entirely in your browser — no video or keypoints are sent to any server.
        </p>
      </footer>
    </div>
  );
}

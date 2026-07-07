/**
 * PredictionDisplay.tsx — Shows current prediction, confidence, sentence log,
 * and text-to-speech controls.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import type { Prediction } from '../lib/onnxInference';

interface PredictionDisplayProps {
  prediction: Prediction | null;
  isModelLoaded: boolean;
  modelError: string | null;
}

// Fixed (Bug #7): previously any repeated word was unconditionally suppressed
// forever until a different word interrupted it, so a user who legitimately
// signed the same word twice in a row (e.g. "no no") only ever got one word
// added and one spoken utterance. A short cooldown absorbs noisy re-triggers
// of the SAME physical gesture (the original goal) without permanently
// blocking a deliberate repeat.
const REPEAT_COOLDOWN_MS = 1500;

export function PredictionDisplay({
  prediction,
  isModelLoaded,
  modelError,
}: PredictionDisplayProps) {
  const [sentence, setSentence] = useState<string[]>([]);
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const lastPredictionRef = useRef<{ word: string; at: number } | null>(null);

  // Add prediction to sentence log and speak
  useEffect(() => {
    if (!prediction) return;

    const now = Date.now();
    const last = lastPredictionRef.current;
    const isNoisyRepeat =
      last !== null &&
      last.word === prediction.word &&
      now - last.at < REPEAT_COOLDOWN_MS;

    if (isNoisyRepeat) return;

    lastPredictionRef.current = { word: prediction.word, at: now };
    setSentence((prev) => [...prev.slice(-9), prediction.word]); // keep last 10

    if (ttsEnabled && 'speechSynthesis' in window) {
      const utt = new SpeechSynthesisUtterance(
        prediction.word.replace(/_/g, ' ')
      );
      utt.lang = 'en-IN';
      utt.rate = 0.9;
      window.speechSynthesis.speak(utt);
    }
  }, [prediction, ttsEnabled]);

  const clearSentence = useCallback(() => setSentence([]), []);

  const confidencePct = prediction ? Math.round(prediction.confidence * 100) : 0;

  return (
    <div className="prediction-panel">
      {/* Model status */}
      <div className="model-status">
        {modelError ? (
          <div className="status-badge status-badge--error">
            <span>⚠</span> {modelError}
          </div>
        ) : !isModelLoaded ? (
          <div className="status-badge status-badge--loading">
            <span className="spinner" /> Loading ONNX model…
          </div>
        ) : (
          <div className="status-badge status-badge--ok">
            <span>✓</span> Model ready — perform a sign to translate
          </div>
        )}
      </div>

      {/* Current prediction — large display */}
      <div className="current-prediction" id="current-prediction" aria-live="polite">
        {prediction ? (
          <>
            <div className="predicted-word" id="predicted-word">
              {prediction.word.replace(/_/g, ' ')}
            </div>
            <div className="confidence-row">
              <div className="confidence-bar-track">
                <div
                  className="confidence-bar-fill"
                  style={{
                    width: `${confidencePct}%`,
                    backgroundColor:
                      confidencePct > 75
                        ? 'var(--green)'
                        : confidencePct > 50
                        ? 'var(--yellow)'
                        : 'var(--red)',
                  }}
                />
              </div>
              <span className="confidence-label">{confidencePct}%</span>
            </div>
          </>
        ) : (
          <div className="no-prediction">
            <div className="no-prediction-icon">🤟</div>
            <p>Waiting for sign…</p>
          </div>
        )}
      </div>

      {/* Top-K alternatives */}
      {prediction && prediction.topK.length > 1 && (
        <div className="topk-panel">
          <h4>Alternatives</h4>
          <div className="topk-list">
            {prediction.topK.slice(1).map((item) => (
              <div key={item.word} className="topk-item">
                <span className="topk-word">{item.word.replace(/_/g, ' ')}</span>
                <span className="topk-conf">{Math.round(item.confidence * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sentence builder */}
      <div className="sentence-panel">
        <div className="sentence-header">
          <h4>Sentence</h4>
          <button className="btn-clear" onClick={clearSentence} id="clear-sentence-btn">
            Clear
          </button>
        </div>
        <div className="sentence-text" id="sentence-text" aria-live="polite">
          {sentence.length > 0
            ? sentence.map((w) => w.replace(/_/g, ' ')).join(' ')
            : <span className="sentence-placeholder">Your translated sentence will appear here…</span>
          }
        </div>
      </div>

      {/* TTS toggle */}
      <div className="tts-panel">
        <button
          className={`btn-tts ${ttsEnabled ? 'btn-tts--active' : ''}`}
          onClick={() => setTtsEnabled((v) => !v)}
          id="tts-toggle"
          aria-pressed={ttsEnabled}
          title="Toggle text-to-speech"
        >
          {ttsEnabled ? '🔊 Speech On' : '🔇 Speech Off'}
        </button>
        <span className="tts-note">Uses browser SpeechSynthesis API</span>
      </div>

      {/* Practice mode stub */}
      <div className="practice-panel practice-panel--disabled">
        <span className="coming-soon-badge">Coming Soon</span>
        <span>Practice Mode — pick a target word and get real-time feedback</span>
      </div>
    </div>
  );
}

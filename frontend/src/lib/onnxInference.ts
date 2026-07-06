/**
 * onnxInference.ts — Client-side ONNX model inference wrapper.
 *
 * Loads model.onnx from /public, manages a keypoint frame buffer,
 * detects gesture boundaries via wrist velocity, and runs inference
 * when a gesture is complete.
 *
 * Privacy note: all inference runs locally in the browser — no keypoint
 * data is ever sent to a server when using this module.
 */

import * as ort from 'onnxruntime-web';

// ── Constants matching Python config.py ───────────────────────────────────────
export const CLIP_LENGTH = 50;
export const FEATURE_DIM = 225;

// Gesture boundary detection parameters
const IDLE_VELOCITY_THRESHOLD = 0.015;
const IDLE_FRAMES_REQUIRED = 8;
const MIN_GESTURE_FRAMES = 10;
const MAX_BUFFER_FRAMES = CLIP_LENGTH * 2;

// Wrist landmark indices in the pose block (landmarks 15=left, 16=right)
const LEFT_WRIST_IDX = 15;
const RIGHT_WRIST_IDX = 16;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Prediction {
  word: string;
  confidence: number;
  topK: Array<{ word: string; confidence: number }>;
}

export interface BufferStatus {
  bufferLength: number;
  isActive: boolean;
}

// ── Normalization (mirrors Python normalize.py) ───────────────────────────────

/**
 * Normalize a single frame's keypoint vector.
 * Translates to shoulder midpoint origin, scales by shoulder width.
 */
export function normalizeFrame(keypoints: Float32Array): Float32Array {
  const kp = new Float32Array(keypoints);

  // Pose block layout: [x0,y0,z0, x1,y1,z1, ...]
  // Left shoulder = landmark 11 (indices 33, 34, 35)
  // Right shoulder = landmark 12 (indices 36, 37, 38)
  const lsX = kp[11 * 3];
  const lsY = kp[11 * 3 + 1];
  const rsX = kp[12 * 3];
  const rsY = kp[12 * 3 + 1];

  const midX = (lsX + rsX) / 2;
  const midY = (lsY + rsY) / 2;
  const shoulderWidth = Math.sqrt((lsX - rsX) ** 2 + (lsY - rsY) ** 2);

  if (shoulderWidth < 1e-6) return kp; // Pose not detected — pass through

  const nLandmarks = FEATURE_DIM / 3;
  for (let i = 0; i < nLandmarks; i++) {
    kp[i * 3] = (kp[i * 3] - midX) / shoulderWidth;
    kp[i * 3 + 1] = (kp[i * 3 + 1] - midY) / shoulderWidth;
    kp[i * 3 + 2] = kp[i * 3 + 2] / shoulderWidth; // scale z only
  }
  return kp;
}

/**
 * Compute mean wrist velocity between two consecutive normalized frames.
 */
export function wristVelocity(frameA: Float32Array, frameB: Float32Array): number {
  const lwAx = frameA[LEFT_WRIST_IDX * 3];
  const lwAy = frameA[LEFT_WRIST_IDX * 3 + 1];
  const rwAx = frameA[RIGHT_WRIST_IDX * 3];
  const rwAy = frameA[RIGHT_WRIST_IDX * 3 + 1];

  const lwBx = frameB[LEFT_WRIST_IDX * 3];
  const lwBy = frameB[LEFT_WRIST_IDX * 3 + 1];
  const rwBx = frameB[RIGHT_WRIST_IDX * 3];
  const rwBy = frameB[RIGHT_WRIST_IDX * 3 + 1];

  const leftVel = Math.sqrt((lwAx - lwBx) ** 2 + (lwAy - lwBy) ** 2);
  const rightVel = Math.sqrt((rwAx - rwBx) ** 2 + (rwAy - rwBy) ** 2);
  return (leftVel + rightVel) / 2;
}

/**
 * Pad or truncate a sequence to exactly CLIP_LENGTH frames.
 * Padding: zeros at end. Truncation: keep last CLIP_LENGTH frames.
 */
export function padOrTruncate(frames: Float32Array[]): Float32Array {
  const T = frames.length;
  const out = new Float32Array(CLIP_LENGTH * FEATURE_DIM); // initialized to 0

  if (T >= CLIP_LENGTH) {
    // Keep last CLIP_LENGTH frames
    const start = T - CLIP_LENGTH;
    for (let i = 0; i < CLIP_LENGTH; i++) {
      out.set(frames[start + i], i * FEATURE_DIM);
    }
  } else {
    // Pad end with zeros
    for (let i = 0; i < T; i++) {
      out.set(frames[i], i * FEATURE_DIM);
    }
  }
  return out;
}

// ── Softmax ───────────────────────────────────────────────────────────────────

function softmax(logits: Float32Array | number[]): Float32Array {
  const arr = Array.from(logits);
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return new Float32Array(exps.map((e) => e / sum));
}

// ── Main Inference Engine ─────────────────────────────────────────────────────

export class ONNXInferenceEngine {
  private session: ort.InferenceSession | null = null;
  private labelMap: Record<string, number> = {};
  private idxToWord: Record<number, string> = {};
  private isLoaded = false;

  // Gesture buffer state
  private buffer: Float32Array[] = [];
  private idleCount = 0;
  private isActive = false;

  async load(modelUrl = '/model.onnx', labelMapUrl = '/label_map.json'): Promise<void> {
    // Configure ONNX runtime to use WASM backend
    ort.env.wasm.numThreads = 1;

    try {
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
      });

      const resp = await fetch(labelMapUrl);
      this.labelMap = await resp.json();
      this.idxToWord = Object.fromEntries(
        Object.entries(this.labelMap).map(([word, idx]) => [idx, word])
      );

      this.isLoaded = true;
      console.log(
        `[ONNXInferenceEngine] Loaded. ${Object.keys(this.labelMap).length} classes.`
      );
    } catch (err) {
      console.error('[ONNXInferenceEngine] Failed to load model:', err);
      throw err;
    }
  }

  get loaded(): boolean {
    return this.isLoaded;
  }

  get vocabulary(): string[] {
    return Object.keys(this.labelMap);
  }

  /**
   * Push a single normalized keypoint frame into the gesture buffer.
   * Returns a Prediction if a gesture boundary is detected, else null.
   * Also returns a BufferStatus for UI progress display.
   */
  pushFrame(rawKeypoints: Float32Array): {
    prediction: Prediction | null;
    status: BufferStatus;
  } {
    if (!this.isLoaded || !this.session) {
      return { prediction: null, status: { bufferLength: 0, isActive: false } };
    }

    const frame = normalizeFrame(rawKeypoints);

    const prevFrame = this.buffer.length > 0 ? this.buffer[this.buffer.length - 1] : null;

    // Enforce max buffer size
    if (this.buffer.length >= MAX_BUFFER_FRAMES) {
      this.buffer.shift();
    }
    this.buffer.push(frame);

    // Check velocity to detect gesture boundary
    let prediction: Prediction | null = null;

    if (prevFrame) {
      const vel = wristVelocity(prevFrame, frame);

      if (vel > IDLE_VELOCITY_THRESHOLD) {
        this.idleCount = 0;
        this.isActive = true;
      } else {
        this.idleCount++;
      }

      if (
        this.isActive &&
        this.idleCount >= IDLE_FRAMES_REQUIRED &&
        this.buffer.length >= MIN_GESTURE_FRAMES
      ) {
        // Extract gesture frames (exclude trailing idle frames)
        const gestureFrames = this.buffer.slice(0, -IDLE_FRAMES_REQUIRED);
        if (gestureFrames.length >= MIN_GESTURE_FRAMES) {
          prediction = this._runInference(gestureFrames);
          this._resetBuffer();
        }
      }
    }

    return {
      prediction,
      status: {
        bufferLength: this.buffer.length,
        isActive: this.isActive,
      },
    };
  }

  private _runInference(frames: Float32Array[]): Prediction | null {
    if (!this.session) return null;

    try {
      const flat = padOrTruncate(frames);
      // Shape: [1, CLIP_LENGTH, FEATURE_DIM]
      const inputTensor = new ort.Tensor('float32', flat, [1, CLIP_LENGTH, FEATURE_DIM]);
      const feeds: Record<string, ort.Tensor> = { keypoints: inputTensor };

      // Synchronous-style: we use runSync if available, else we need async
      // Note: onnxruntime-web v1.20+ supports session.run() which returns a Promise
      // We handle this with a queued approach — see useONNXInference hook
      // For now, we expose an async predict method separately
      return null; // placeholder — actual call is in predictAsync
    } catch {
      return null;
    }
  }

  /**
   * Async inference on a complete gesture sequence.
   * Called internally by pushFrameAsync.
   */
  private async _predictAsync(frames: Float32Array[]): Promise<Prediction> {
    if (!this.session) throw new Error('Model not loaded');

    const flat = padOrTruncate(frames);
    const inputTensor = new ort.Tensor('float32', flat, [1, CLIP_LENGTH, FEATURE_DIM]);
    const results = await this.session.run({ keypoints: inputTensor });
    const logitsData = results['logits'].data as Float32Array;
    const probs = softmax(logitsData);

    const n = probs.length;
    const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);

    const topK = indices.slice(0, 3).map((i) => ({
      word: this.idxToWord[i] ?? `class_${i}`,
      confidence: probs[i],
    }));

    return { word: topK[0].word, confidence: topK[0].confidence, topK };
  }

  /**
   * Async version of pushFrame.
   * Use this in your React hook so inference doesn't block the main thread.
   */
  async pushFrameAsync(rawKeypoints: Float32Array): Promise<{
    prediction: Prediction | null;
    status: BufferStatus;
  }> {
    if (!this.isLoaded) {
      return { prediction: null, status: { bufferLength: 0, isActive: false } };
    }

    const frame = normalizeFrame(rawKeypoints);
    const prevFrame = this.buffer.length > 0 ? this.buffer[this.buffer.length - 1] : null;

    if (this.buffer.length >= MAX_BUFFER_FRAMES) this.buffer.shift();
    this.buffer.push(frame);

    let prediction: Prediction | null = null;

    if (prevFrame) {
      const vel = wristVelocity(prevFrame, frame);
      if (vel > IDLE_VELOCITY_THRESHOLD) {
        this.idleCount = 0;
        this.isActive = true;
      } else {
        this.idleCount++;
      }

      if (
        this.isActive &&
        this.idleCount >= IDLE_FRAMES_REQUIRED &&
        this.buffer.length >= MIN_GESTURE_FRAMES
      ) {
        const gestureFrames = this.buffer.slice(0, -IDLE_FRAMES_REQUIRED);
        if (gestureFrames.length >= MIN_GESTURE_FRAMES) {
          prediction = await this._predictAsync(gestureFrames);
          this._resetBuffer();
        }
      }
    }

    return {
      prediction,
      status: { bufferLength: this.buffer.length, isActive: this.isActive },
    };
  }

  resetBuffer(): void {
    this._resetBuffer();
  }

  private _resetBuffer(): void {
    this.buffer = [];
    this.idleCount = 0;
    this.isActive = false;
  }
}

// Singleton instance
export const inferenceEngine = new ONNXInferenceEngine();

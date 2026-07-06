/**
 * useMediaPipe.ts — React hook wrapping MediaPipe Holistic (Tasks Vision API).
 *
 * Manages:
 *  - Loading the MediaPipe WASM model
 *  - Per-frame landmark extraction from a video element
 *  - Drawing skeleton overlay onto a canvas element
 *  - Producing a flat Float32Array of raw (unnormalized) keypoints per frame
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  HolisticLandmarker,
  FilesetResolver,
  DrawingUtils,
} from '@mediapipe/tasks-vision';
import { FEATURE_DIM } from '../lib/onnxInference';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface MediaPipeState {
  isReady: boolean;
  error: string | null;
  latestKeypoints: Float32Array | null;
  processFrame: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => void;
}

// ── Raw keypoint extraction ────────────────────────────────────────────────────
// Result from HolisticLandmarker.detectForVideo() has shape:
//   { poseLandmarks: NormalizedLandmark[][], leftHandLandmarks: NormalizedLandmark[][], ... }

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractKeypoints(result: any): Float32Array {
  const out = new Float32Array(FEATURE_DIM); // 225 zeros

  // Pose: 33 × 3 = indices 0..98
  if (result.poseLandmarks?.length > 0) {
    const pose = result.poseLandmarks[0];
    for (let i = 0; i < Math.min(pose.length, 33); i++) {
      out[i * 3]     = pose[i].x;
      out[i * 3 + 1] = pose[i].y;
      out[i * 3 + 2] = pose[i].z ?? 0;
    }
  }

  // Left hand: 21 × 3 = indices 99..161
  if (result.leftHandLandmarks?.length > 0) {
    const lh = result.leftHandLandmarks[0];
    for (let i = 0; i < Math.min(lh.length, 21); i++) {
      out[99 + i * 3]     = lh[i].x;
      out[99 + i * 3 + 1] = lh[i].y;
      out[99 + i * 3 + 2] = lh[i].z ?? 0;
    }
  }

  // Right hand: 21 × 3 = indices 162..224
  if (result.rightHandLandmarks?.length > 0) {
    const rh = result.rightHandLandmarks[0];
    for (let i = 0; i < Math.min(rh.length, 21); i++) {
      out[162 + i * 3]     = rh[i].x;
      out[162 + i * 3 + 1] = rh[i].y;
      out[162 + i * 3 + 2] = rh[i].z ?? 0;
    }
  }

  return out;
}

// ── Skeleton drawing ───────────────────────────────────────────────────────────

const FINGER_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const UPPER_BODY_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function drawSkeleton(ctx: CanvasRenderingContext2D, result: any): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  function drawPoints(landmarks: { x: number; y: number }[], color: string, r = 3) {
    ctx.fillStyle = color;
    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawLines(
    landmarks: { x: number; y: number }[],
    connections: number[][],
    color: string,
    lineWidth = 2,
  ) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    for (const [a, b] of connections) {
      if (!landmarks[a] || !landmarks[b]) continue;
      ctx.beginPath();
      ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
      ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
      ctx.stroke();
    }
  }

  // Pose — blue
  if (result.poseLandmarks?.length > 0) {
    const pose = result.poseLandmarks[0];
    drawLines(pose, UPPER_BODY_CONNECTIONS, 'rgba(96, 165, 250, 0.8)', 2);
    drawPoints(pose.slice(11, 17), 'rgba(96, 165, 250, 0.9)', 4);
  }

  // Left hand — green
  if (result.leftHandLandmarks?.length > 0) {
    const lh = result.leftHandLandmarks[0];
    drawLines(lh, FINGER_CONNECTIONS, 'rgba(74, 222, 128, 0.8)', 2);
    drawPoints(lh, 'rgba(74, 222, 128, 0.9)', 3);
  }

  // Right hand — orange
  if (result.rightHandLandmarks?.length > 0) {
    const rh = result.rightHandLandmarks[0];
    drawLines(rh, FINGER_CONNECTIONS, 'rgba(251, 146, 60, 0.8)', 2);
    drawPoints(rh, 'rgba(251, 146, 60, 0.9)', 3);
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useMediaPipe(): MediaPipeState {
  const landmarkerRef = useRef<HolisticLandmarker | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);

  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latestKeypoints, setLatestKeypoints] = useState<Float32Array | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        const holistic = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minHandLandmarksConfidence: 0.5,
        });

        if (!cancelled) {
          landmarkerRef.current = holistic;
          setIsReady(true);
          console.log('[useMediaPipe] Loaded ✓');
        }
      } catch (err) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : String(err);
          setError(`Failed to load MediaPipe: ${msg}`);
          console.error('[useMediaPipe]', err);
        }
      }
    }

    loadMediaPipe();
    return () => { cancelled = true; };
  }, []);

  const processFrame = useCallback(
    (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
      if (!landmarkerRef.current || !isReady || video.readyState < 2) return;

      const currentTime = video.currentTime;
      if (currentTime === lastVideoTimeRef.current) return;
      lastVideoTimeRef.current = currentTime;

      const result = landmarkerRef.current.detectForVideo(video, performance.now());

      const ctx = canvas.getContext('2d');
      if (ctx) drawSkeleton(ctx, result);

      setLatestKeypoints(extractKeypoints(result));
    },
    [isReady]
  );

  return { isReady, error, latestKeypoints, processFrame };
}

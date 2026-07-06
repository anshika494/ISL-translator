/**
 * Webcam.tsx — Video capture + canvas overlay component.
 *
 * Handles:
 *  - getUserMedia webcam access
 *  - requestAnimationFrame loop calling MediaPipe processFrame
 *  - Canvas overlay for skeleton drawing
 */

import { useEffect, useRef, useCallback, forwardRef } from 'react';

interface WebcamProps {
  onFrame: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => void;
  isReady: boolean;
  onError: (msg: string) => void;
}

export const Webcam = forwardRef<HTMLVideoElement, WebcamProps>(
  ({ onFrame, isReady, onError }, _ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rafRef = useRef<number>(0);
    const streamRef = useRef<MediaStream | null>(null);

    // Start webcam
    useEffect(() => {
      async function startCamera() {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              width: { ideal: 640 },
              height: { ideal: 480 },
              facingMode: 'user',
            },
            audio: false,
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : 'Camera access denied';
          onError(`Webcam error: ${msg}`);
        }
      }
      startCamera();

      return () => {
        cancelAnimationFrame(rafRef.current);
        streamRef.current?.getTracks().forEach((t) => t.stop());
      };
    }, [onError]);

    // Animation frame loop
    const loop = useCallback(() => {
      if (videoRef.current && canvasRef.current && isReady) {
        // Sync canvas size to video
        if (canvasRef.current.width !== videoRef.current.videoWidth) {
          canvasRef.current.width = videoRef.current.videoWidth || 640;
          canvasRef.current.height = videoRef.current.videoHeight || 480;
        }
        onFrame(videoRef.current, canvasRef.current);
      }
      rafRef.current = requestAnimationFrame(loop);
    }, [onFrame, isReady]);

    useEffect(() => {
      rafRef.current = requestAnimationFrame(loop);
      return () => cancelAnimationFrame(rafRef.current);
    }, [loop]);

    return (
      <div className="webcam-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="webcam-video"
          id="webcam-video"
        />
        <canvas ref={canvasRef} className="webcam-canvas" id="skeleton-canvas" />
      </div>
    );
  }
);

Webcam.displayName = 'Webcam';

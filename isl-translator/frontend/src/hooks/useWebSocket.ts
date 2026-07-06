/**
 * useWebSocket.ts — WebSocket client hook for fallback backend inference.
 *
 * Used only when the backend FastAPI WebSocket path is enabled.
 * For client-side ONNX inference (default), this hook is not needed.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Prediction } from '../lib/onnxInference';

export interface WebSocketState {
  isConnected: boolean;
  error: string | null;
  sendFrame: (keypoints: Float32Array) => void;
  onPrediction: (callback: (pred: Prediction) => void) => void;
  disconnect: () => void;
}

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws/infer';
const RECONNECT_DELAY_MS = 2000;

export function useWebSocket(): WebSocketState {
  const wsRef = useRef<WebSocket | null>(null);
  const predictionCallbackRef = useRef<((pred: Prediction) => void) | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
      console.log('[useWebSocket] Connected to', WS_URL);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data as string);
        if (msg.type === 'prediction' && predictionCallbackRef.current) {
          predictionCallbackRef.current({
            word: msg.word,
            confidence: msg.confidence,
            topK: msg.top_k ?? [],
          });
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Auto-reconnect
      reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = (evt) => {
      setError('WebSocket connection error — is the backend running?');
      console.error('[useWebSocket] Error:', evt);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendFrame = useCallback((keypoints: Float32Array) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(
      JSON.stringify({
        type: 'frame',
        keypoints: Array.from(keypoints),
      })
    );
  }, []);

  const onPrediction = useCallback((callback: (pred: Prediction) => void) => {
    predictionCallbackRef.current = callback;
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    wsRef.current?.close();
    setIsConnected(false);
  }, []);

  return { isConnected, error, sendFrame, onPrediction, disconnect };
}

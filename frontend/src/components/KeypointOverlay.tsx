/**
 * KeypointOverlay.tsx — Visual indicator for tracking status.
 *
 * Shows colored badges for pose/hand detection status and
 * a live buffer progress bar during gesture capture.
 */

interface KeypointOverlayProps {
  isMediaPipeReady: boolean;
  isActive: boolean;
  bufferLength: number;
  maxBuffer: number;
}

export function KeypointOverlay({
  isMediaPipeReady,
  isActive,
  bufferLength,
  maxBuffer,
}: KeypointOverlayProps) {
  const progress = Math.min((bufferLength / maxBuffer) * 100, 100);

  return (
    <div className="keypoint-overlay">
      <div className="tracking-badges">
        <span className={`badge ${isMediaPipeReady ? 'badge--green' : 'badge--yellow'}`}>
          <span className="badge-dot" />
          {isMediaPipeReady ? 'Tracking Active' : 'Loading...'}
        </span>
        {isActive && (
          <span className="badge badge--red">
            <span className="badge-dot badge-dot--pulse" />
            Gesture Detected
          </span>
        )}
      </div>

      {isActive && bufferLength > 0 && (
        <div className="buffer-bar-container" title={`${bufferLength} frames captured`}>
          <div className="buffer-bar" style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  );
}

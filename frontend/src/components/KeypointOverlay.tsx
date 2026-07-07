/**
 * KeypointOverlay.tsx — Visual indicator for tracking status.
 *
 * Shows colored badges for pose/hand detection status and
 * a live buffer progress bar during gesture capture.
 */

interface KeypointOverlayProps {
  isMediaPipeReady: boolean;
  /**
   * True when a pose is actually detected in the current frame (a person is
   * visibly in view). Distinct from isMediaPipeReady, which only means the
   * model finished loading.
   */
  isPoseDetected: boolean;
  isActive: boolean;
  bufferLength: number;
  maxBuffer: number;
}

export function KeypointOverlay({
  isMediaPipeReady,
  isPoseDetected,
  isActive,
  bufferLength,
  maxBuffer,
}: KeypointOverlayProps) {
  const progress = Math.min((bufferLength / maxBuffer) * 100, 100);

  // Fixed (Bug #6): three distinct states instead of a single "loaded vs not"
  // badge that used to claim "Tracking Active" even when nobody was in frame.
  let statusBadgeClass = 'badge--yellow';
  let statusText = 'Loading…';
  let showPulseDot = false;

  if (isMediaPipeReady && isPoseDetected) {
    statusBadgeClass = 'badge--green';
    statusText = 'Tracking Active';
  } else if (isMediaPipeReady && !isPoseDetected) {
    statusBadgeClass = 'badge--red';
    statusText = 'No Person Detected';
    showPulseDot = true;
  }

  return (
    <div className="keypoint-overlay">
      <div className="tracking-badges">
        <span className={`badge ${statusBadgeClass}`}>
          <span className={`badge-dot ${showPulseDot ? 'badge-dot--pulse' : ''}`} />
          {statusText}
        </span>
        {isActive && isPoseDetected && (
          <span className="badge badge--red">
            <span className="badge-dot badge-dot--pulse" />
            Gesture Detected
          </span>
        )}
      </div>

      {isActive && isPoseDetected && bufferLength > 0 && (
        <div className="buffer-bar-container" title={`${bufferLength} frames captured`}>
          <div className="buffer-bar" style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  );
}

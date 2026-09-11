import { AlertTriangle, ArrowDown, Camera, CircleStop, Film, Images, Play, RefreshCw, RotateCcw } from 'lucide-react'
import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, errorMessage } from '../api'
import type { LiveTrack, PipelineCapture, PipelineConfig, PipelineCoordinate, PipelineSnapshot } from '../types'

interface Props {
  online: boolean
  onError: (message: string) => void
}

function hasCurrentPipelineSchema(config: PipelineConfig) {
  return Boolean(
    config.tracking
    && config.zones?.roi_left
    && config.zones.roi_right
    && config.zones.roi_top
    && config.zones.roi_bottom
    && config.zones.entry_zone_height
    && config.zones.exit_zone_height,
  )
}

function coordinatePixels(coordinate: PipelineCoordinate, extent: number) {
  return coordinate.normalized ? coordinate.value * extent : coordinate.value
}

function stateLabel(track: LiveTrack, minimumHits: number) {
  if (track.finalClass) return track.finalClass.replace('Class ', 'C')
  if (track.state === 'WAITING_FOR_CLASSIFICATION') return 'WAIT'
  if (track.state === 'WAITING_FOR_SEGMENTATION') return 'SEG'
  if (track.state === 'ACQUIRING') return 'ACQ'
  if (track.state === 'TENTATIVE') return `TENT ${track.confirmationHits}/${minimumHits}`
  return track.state.slice(0, 4)
}

function PipelineOverlay({ snapshot, config }: { snapshot: PipelineSnapshot; config: PipelineConfig }) {
  const width = snapshot.sourceWidth ?? 1
  const height = snapshot.sourceHeight ?? 1
  const left = coordinatePixels(config.zones.roi_left, width)
  const right = coordinatePixels(config.zones.roi_right, width)
  const top = coordinatePixels(config.zones.roi_top, height)
  const bottom = coordinatePixels(config.zones.roi_bottom, height)
  const entry = top + coordinatePixels(config.zones.entry_zone_height, height)
  const classification = bottom - coordinatePixels(config.zones.exit_zone_height, height)
  const debug = config.tracking.debug_diagnostics
  const overlayTracks = snapshot.tracks.filter((track) =>
    track.hasCurrentObservation && track.state !== 'LOST' && track.state !== 'COMPLETED' && track.state !== 'REJECTED' &&
    (debug || track.confirmed),
  )
  return (
    <svg className="live-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <rect x={left} y={top} width={Math.max(0, right - left)} height={Math.max(0, bottom - top)} className="zone roi-zone" />
      <rect x={left} y={top} width={Math.max(0, right - left)} height={Math.max(0, entry - top)} className="zone entry-zone" />
      <rect x={left} y={entry} width={Math.max(0, right - left)} height={Math.max(0, classification - entry)} className="zone acquisition-zone" />
      <rect x={left} y={classification} width={Math.max(0, right - left)} height={Math.max(0, bottom - classification)} className="zone exit-zone" />
      <line x1={left} x2={right} y1={entry} y2={entry} className="zone-line acquisition-line" />
      <line x1={left} x2={right} y1={classification} y2={classification} className="zone-line classification-line" />
      <line x1={left} x2={right} y1={bottom} y2={bottom} className="zone-line post-line" />
      {overlayTracks.map((track) => {
        const box = track.currentBoundingBox
        const tentative = track.state === 'TENTATIVE'
        return (
          <g key={track.trackId} className={tentative ? 'tentative-track' : ''}>
            <rect x={box.x} y={box.y} width={box.width} height={box.height} className="track-box" />
            <rect x={box.x} y={Math.max(0, box.y - 22)} width={Math.min(box.width, debug ? 220 : 150)} height={21} className="track-label-bg" />
            <text x={box.x + 5} y={Math.max(14, box.y - 7)} className="track-label">
              #{track.trackId} {stateLabel(track, config.tracking.minimum_confirmation_hits)} {track.viewsCaptured}/{config.capture.target_views}
              {debug ? ` ${track.motionStatus} ${track.sizeStatus}` : ''}
            </text>
          </g>
        )
      })}
      <g className="direction-marker">
        <line x1={width * 0.08} x2={width * 0.08} y1={height * 0.07} y2={height * 0.17} />
        <polyline points={`${width * 0.055},${height * 0.15} ${width * 0.08},${height * 0.17} ${width * 0.105},${height * 0.15}`} />
        <text x={width * 0.025} y={height * 0.045}>TOP → BOTTOM</text>
      </g>
    </svg>
  )
}

function TrackDetails({ track, targetViews, debug }: { track: LiveTrack; targetViews: number; debug: boolean }) {
  return (
    <article className={`live-track-card ${track.state.toLowerCase()}`}>
      <div className="live-track-title">
        <strong>Track #{track.trackId}</strong>
        <span>{track.finalClass ?? track.state.replaceAll('_', ' ')}</span>
      </div>
      <div className="track-progress"><i style={{ width: `${track.acquisitionZoneProgress * 100}%` }} /></div>
      <dl>
        <div><dt>Views</dt><dd>{track.viewsCaptured}/{targetViews}</dd></div>
        <div><dt>Segmented</dt><dd>{track.segmentationCompleted}</dd></div>
        <div><dt>Confidence</dt><dd>{track.finalClass ? `${(track.finalConfidence * 100).toFixed(1)}%` : '—'}</dd></div>
        <div><dt>Used</dt><dd>{track.viewsUsed || '—'}</dd></div>
      </dl>
      {debug && <small>{track.state === 'TENTATIVE' ? `${track.confirmationHits} confirmation hits · ` : ''}{track.motionStatus} · {track.sizeStatus}</small>}
      {track.incompleteEvidence && <p className="evidence-warning"><AlertTriangle size={12} /> Incomplete evidence</p>}
      {track.messages.length > 0 && <small title={track.messages.join('\n')}>{track.messages.at(-1)}</small>}
    </article>
  )
}

function CaptureLibrary({ captures, onRefresh }: { captures: PipelineCapture[]; onRefresh: () => void }) {
  const groups = useMemo(() => {
    const grouped = new Map<string, { runId: string; trackId: number; captures: PipelineCapture[] }>()
    for (const capture of captures) {
      const key = `${capture.runId}:${capture.trackId}`
      const group = grouped.get(key) ?? { runId: capture.runId, trackId: capture.trackId, captures: [] }
      group.captures.push(capture)
      grouped.set(key, group)
    }
    return [...grouped.values()]
  }, [captures])

  return (
    <section className="capture-library" aria-labelledby="capture-library-title">
      <div className="capture-library-heading">
        <div>
          <span className="eyebrow">Exact crops sent to segmentation</span>
          <h3 id="capture-library-title"><Images size={16} /> Capture library</h3>
        </div>
        <div>
          <span>{groups.length} tracks · {captures.length} views</span>
          <button className="icon-button" onClick={onRefresh} aria-label="Refresh capture library"><RefreshCw size={14} /></button>
        </div>
      </div>
      {groups.length === 0 ? (
        <div className="capture-library-empty">
          <Images size={24} />
          <strong>No captured views yet</strong>
          <span>Start the live conveyor. Confirmed tracks are saved here when they cross each configured capture point.</span>
        </div>
      ) : (
        <div className="capture-track-groups">
          {groups.map((group) => (
            <article className="capture-track-group" key={`${group.runId}:${group.trackId}`}>
              <header><strong>Track #{group.trackId}</strong><span>{group.captures.length} captured view{group.captures.length === 1 ? '' : 's'}</span></header>
              <div className="capture-grid">
                {group.captures.map((capture) => (
                  <a className="capture-card" href={capture.imageUrl} target="_blank" rel="noreferrer" key={capture.captureId}>
                    <img src={capture.imageUrl} alt={`Track ${capture.trackId}, view ${capture.viewId + 1}, frame ${capture.frameId}`} loading="lazy" />
                    <div><strong>View {capture.viewId + 1}</strong><span>Frame {capture.frameId}</span></div>
                    <dl>
                      <div><dt>Position</dt><dd>{Math.round(capture.zoneProgress * 100)}%</dd></div>
                      <div><dt>Sharpness</dt><dd>{capture.sharpness.toFixed(0)}</dd></div>
                      <div><dt>Source</dt><dd>{capture.usedTrackingPrediction ? 'Tracker' : `Detector ${Math.round(capture.detectionConfidence * 100)}%`}</dd></div>
                      <div><dt>Crop</dt><dd>{capture.cropWidth}×{capture.cropHeight}</dd></div>
                    </dl>
                  </a>
                ))}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

export function LiveConveyor({ online, onError }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const processedCanvasRef = useRef<HTMLCanvasElement>(null)
  const frameIdRef = useRef(0)
  const captureSignatureRef = useRef('')
  const streamRef = useRef<MediaStream | null>(null)
  const videoUrlRef = useRef<string | null>(null)
  const [config, setConfig] = useState<PipelineConfig | null>(null)
  const [snapshot, setSnapshot] = useState<PipelineSnapshot | null>(null)
  const [running, setRunning] = useState(false)
  const [sourceReady, setSourceReady] = useState(false)
  const [sourceName, setSourceName] = useState('No source selected')
  const [hasProcessedFrame, setHasProcessedFrame] = useState(false)
  const [captures, setCaptures] = useState<PipelineCapture[]>([])

  const loadCaptures = useCallback(async () => {
    try {
      setCaptures(await api.pipelineCaptures())
    } catch (error) {
      onError(errorMessage(error))
    }
  }, [onError])

  useEffect(() => {
    if (!online) return
    api.pipelineConfig()
      .then((nextConfig) => {
        if (!hasCurrentPipelineSchema(nextConfig)) {
          throw new Error('The native runtime uses an outdated pipeline schema. Rebuild and restart it, then refresh this page.')
        }
        setConfig(nextConfig)
      })
      .catch((error) => {
        setConfig(null)
        onError(errorMessage(error))
      })
  }, [online, onError])

  useEffect(() => {
    if (online) void loadCaptures()
  }, [online, loadCaptures])

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
  }, [])

  useEffect(() => {
    if (!running || !config) return
    let cancelled = false
    const interval = 1000 / config.windows.live_max_fps
    async function pump() {
      // This sequential pump is intentional: each pipeline request mutates one
      // tracker state machine, so frames must arrive in order and there should
      // never be multiple frame requests waiting behind slow inference.
      if (cancelled) return
      const video = videoRef.current
      const canvas = canvasRef.current
      const started = performance.now()
      try {
        if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0) {
          window.setTimeout(pump, 100)
          return
        }
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d', { alpha: false })?.drawImage(video, 0, 0, canvas.width, canvas.height)
        const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.82))
        if (!blob || cancelled) return
        frameIdRef.current += 1
        const next = await api.pipelineFrame(blob, frameIdRef.current)
        if (!cancelled) {
          const processed = processedCanvasRef.current
          if (processed) {
            // Keep the displayed pixels synchronized with the boxes returned
            // for this request, even though the webcam/video keeps advancing.
            processed.width = canvas.width
            processed.height = canvas.height
            processed.getContext('2d', { alpha: false })?.drawImage(canvas, 0, 0)
          }
          setSnapshot(next)
          setHasProcessedFrame(true)
          const captureSignature = next.tracks.flatMap((track) => track.capturedViews.map((view) => view.captureId)).filter(Boolean).sort().join('|')
          if (captureSignature !== captureSignatureRef.current) {
            captureSignatureRef.current = captureSignature
            void loadCaptures()
          }
        }
      } catch (error) {
        if (!cancelled) {
          setRunning(false)
          onError(errorMessage(error))
        }
        return
      }
      window.setTimeout(pump, Math.max(0, interval - (performance.now() - started)))
    }
    void pump()
    return () => { cancelled = true }
  }, [running, config, onError, loadCaptures])

  async function useWebcam() {
    try {
      streamRef.current?.getTracks().forEach((track) => track.stop())
      if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      streamRef.current = stream
      const video = videoRef.current
      if (video) {
        video.srcObject = stream
        video.removeAttribute('src')
        await video.play()
      }
      setSourceReady(true)
      setSourceName('Webcam')
      setHasProcessedFrame(false)
    } catch (error) {
      onError(errorMessage(error))
    }
  }

  async function chooseVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setRunning(false)
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    const url = URL.createObjectURL(file)
    videoUrlRef.current = url
    const video = videoRef.current
    if (video) {
      video.srcObject = null
      video.src = url
      video.loop = true
      await video.play()
    }
    setSourceReady(true)
    setSourceName(file.name)
    setHasProcessedFrame(false)
  }

  async function start() {
    if (!sourceReady) return
    try {
      await api.resetPipeline()
      frameIdRef.current = 0
      setSnapshot(null)
      setHasProcessedFrame(false)
      captureSignatureRef.current = ''
      setRunning(true)
      await videoRef.current?.play()
    } catch (error) {
      onError(errorMessage(error))
    }
  }

  async function reset() {
    setRunning(false)
    frameIdRef.current = 0
    setHasProcessedFrame(false)
    captureSignatureRef.current = ''
    try {
      setSnapshot(await api.resetPipeline())
    } catch (error) {
      onError(errorMessage(error))
    }
  }

  const tracks = useMemo(() => {
    const all = snapshot?.tracks ?? []
    if (config?.tracking?.debug_diagnostics) return all
    return all.filter((track) => track.confirmed || track.state === 'COMPLETED')
  }, [snapshot, config])

  return (
    <section className="panel live-conveyor" aria-labelledby="live-title">
      <div className="panel-heading live-heading">
        <div><span className="eyebrow">YOLO26n + ByteTrack + asynchronous segmentation</span><h2 id="live-title">Live conveyor</h2></div>
        <div className="live-actions">
          <button className="button secondary" onClick={() => void useWebcam()} disabled={!online || running}><Camera size={15} /> Webcam</button>
          <label className={`button secondary ${running ? 'disabled' : ''}`}><Film size={15} /> Video file<input type="file" accept="video/*" onChange={(event) => void chooseVideo(event)} disabled={running} /></label>
          <button className="button primary" onClick={() => void (running ? Promise.resolve(setRunning(false)) : start())} disabled={!online || !sourceReady}>
            {running ? <><CircleStop size={15} /> Stop</> : <><Play size={15} /> Start</>}
          </button>
          <button className="icon-button" onClick={() => void reset()} disabled={!online} aria-label="Reset live pipeline"><RotateCcw size={15} /></button>
        </div>
      </div>
      <div className="live-layout">
        <div>
          <div className="live-stage">
            <video ref={videoRef} className={hasProcessedFrame ? 'processed-source-video' : undefined} muted playsInline onLoadedData={() => setSourceReady(true)} />
            <canvas ref={processedCanvasRef} className={`live-processed-frame ${hasProcessedFrame ? 'visible' : ''}`} aria-label="Latest processed conveyor frame" />
            {snapshot && config && <PipelineOverlay snapshot={snapshot} config={config} />}
            {!sourceReady && <div className="live-placeholder"><ArrowDown size={28} /><strong>Select a webcam or recorded video</strong><span>The conveyor path runs from the top of the frame to the bottom.</span></div>}
          </div>
          <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
          <div className="source-caption"><span className={running ? 'live-dot' : ''} />{sourceName}{hasProcessedFrame ? ' · synchronized processed frame' : ''}</div>
          <div className="pipeline-metrics">
            <div><span>Detector FPS</span><strong>{snapshot?.detectorFps.toFixed(1) ?? '0.0'}</strong></div>
            <div><span>Queue</span><strong>{snapshot?.queueSize ?? 0}</strong></div>
            <div><span>Seg latency</span><strong>{snapshot ? `${snapshot.segmentationLatencyMs.toFixed(1)} ms` : '—'}</strong></div>
            <div><span>End-to-end</span><strong>{snapshot?.classificationLatencyMs ? `${snapshot.classificationLatencyMs.toFixed(1)} ms` : '—'}</strong></div>
            <div><span>Dropped jobs</span><strong>{snapshot?.droppedSegmentationJobs ?? 0}</strong></div>
            <div><span>Completed fruit</span><strong>{snapshot?.completedCount ?? 0}</strong></div>
          </div>
        </div>
        <aside className="track-panel">
          <div className="track-panel-heading"><strong>Tracked naartjies</strong><span>{tracks.length} active/retained</span></div>
          {tracks.length === 0 ? <div className="track-empty">Track details appear here without covering the fruit.</div> : tracks.map((track) => (
            <TrackDetails key={track.trackId} track={track} targetViews={config?.capture.target_views ?? 4} debug={config?.tracking?.debug_diagnostics ?? false} />
          ))}
        </aside>
      </div>
      <CaptureLibrary captures={captures} onRefresh={() => void loadCaptures()} />
    </section>
  )
}

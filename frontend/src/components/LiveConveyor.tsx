import { AlertTriangle, ArrowRight, Camera, CircleStop, Film, Play, RotateCcw } from 'lucide-react'
import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import { api, errorMessage } from '../api'
import type { LiveTrack, PipelineConfig, PipelineCoordinate, PipelineSnapshot } from '../types'

interface Props {
  online: boolean
  onError: (message: string) => void
}

function coordinatePixels(coordinate: PipelineCoordinate, width: number) {
  return coordinate.normalized ? coordinate.value * width : coordinate.value
}

function stateLabel(track: LiveTrack) {
  if (track.finalClass) return track.finalClass.replace('Class ', 'C')
  if (track.state === 'WAITING_FOR_CLASSIFICATION') return 'WAIT'
  if (track.state === 'WAITING_FOR_SEGMENTATION') return 'SEG'
  if (track.state === 'ACQUIRING') return 'ACQ'
  return track.state.slice(0, 4)
}

function PipelineOverlay({ snapshot, config }: { snapshot: PipelineSnapshot; config: PipelineConfig }) {
  const width = snapshot.sourceWidth ?? 1
  const height = snapshot.sourceHeight ?? 1
  const acquisition = coordinatePixels(config.zones.acquisition_zone_start_x, width)
  const classification = coordinatePixels(config.zones.classification_line_x, width)
  const post = coordinatePixels(config.zones.post_classification_zone, width)
  return (
    <svg className="live-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <rect x={acquisition} y={0} width={Math.max(0, classification - acquisition)} height={height} className="zone acquisition-zone" />
      <line x1={acquisition} x2={acquisition} y1={0} y2={height} className="zone-line acquisition-line" />
      <line x1={classification} x2={classification} y1={0} y2={height} className="zone-line classification-line" />
      <line x1={post} x2={post} y1={0} y2={height} className="zone-line post-line" />
      {snapshot.tracks.map((track) => {
        const box = track.currentBoundingBox
        const terminal = track.finalClassId !== null
        return (
          <g key={track.trackId} className={terminal ? 'classified-track' : ''}>
            <rect x={box.x} y={box.y} width={box.width} height={box.height} className="track-box" />
            <rect x={box.x} y={Math.max(0, box.y - 22)} width={Math.min(box.width, 132)} height={21} className="track-label-bg" />
            <text x={box.x + 5} y={Math.max(14, box.y - 7)} className="track-label">
              #{track.trackId} {stateLabel(track)} {track.viewsCaptured}/{config.capture.target_views}
            </text>
          </g>
        )
      })}
      <g className="direction-marker">
        <line x1={width * 0.04} x2={width * 0.14} y1={height * 0.08} y2={height * 0.08} />
        <polyline points={`${width * 0.12},${height * 0.055} ${width * 0.14},${height * 0.08} ${width * 0.12},${height * 0.105}`} />
        <text x={width * 0.04} y={height * 0.05}>CONVEYOR</text>
      </g>
    </svg>
  )
}

function TrackDetails({ track, targetViews }: { track: LiveTrack; targetViews: number }) {
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
      {track.incompleteEvidence && <p className="evidence-warning"><AlertTriangle size={12} /> Incomplete evidence</p>}
      {track.messages.length > 0 && <small title={track.messages.join('\n')}>{track.messages.at(-1)}</small>}
    </article>
  )
}

export function LiveConveyor({ online, onError }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameIdRef = useRef(0)
  const streamRef = useRef<MediaStream | null>(null)
  const videoUrlRef = useRef<string | null>(null)
  const [config, setConfig] = useState<PipelineConfig | null>(null)
  const [snapshot, setSnapshot] = useState<PipelineSnapshot | null>(null)
  const [running, setRunning] = useState(false)
  const [sourceReady, setSourceReady] = useState(false)
  const [sourceName, setSourceName] = useState('No source selected')

  useEffect(() => {
    if (!online) return
    api.pipelineConfig().then(setConfig).catch((error) => onError(errorMessage(error)))
  }, [online, onError])

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
  }, [])

  useEffect(() => {
    if (!running || !config) return
    let cancelled = false
    const interval = 1000 / config.windows.live_max_fps
    async function pump() {
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
        if (!cancelled) setSnapshot(next)
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
  }, [running, config, onError])

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
  }

  async function start() {
    if (!sourceReady) return
    try {
      await api.resetPipeline()
      frameIdRef.current = 0
      setSnapshot(null)
      setRunning(true)
      await videoRef.current?.play()
    } catch (error) {
      onError(errorMessage(error))
    }
  }

  async function reset() {
    setRunning(false)
    frameIdRef.current = 0
    try {
      setSnapshot(await api.resetPipeline())
    } catch (error) {
      onError(errorMessage(error))
    }
  }

  const tracks = useMemo(() => snapshot?.tracks ?? [], [snapshot])

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
            <video ref={videoRef} muted playsInline onLoadedData={() => setSourceReady(true)} />
            {snapshot && config && <PipelineOverlay snapshot={snapshot} config={config} />}
            {!sourceReady && <div className="live-placeholder"><ArrowRight size={28} /><strong>Select a webcam or recorded video</strong><span>The same frame API drives both physical and simulated conveyor runs.</span></div>}
          </div>
          <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
          <div className="source-caption"><span className={running ? 'live-dot' : ''} />{sourceName}</div>
          <div className="pipeline-metrics">
            <div><span>Detector FPS</span><strong>{snapshot?.detectorFps.toFixed(1) ?? '0.0'}</strong></div>
            <div><span>Queue</span><strong>{snapshot?.queueSize ?? 0}</strong></div>
            <div><span>Seg latency</span><strong>{snapshot ? `${snapshot.segmentationLatencyMs.toFixed(1)} ms` : '—'}</strong></div>
            <div><span>End-to-end</span><strong>{snapshot?.classificationLatencyMs ? `${snapshot.classificationLatencyMs.toFixed(1)} ms` : '—'}</strong></div>
            <div><span>Dropped jobs</span><strong>{snapshot?.droppedSegmentationJobs ?? 0}</strong></div>
          </div>
        </div>
        <aside className="track-panel">
          <div className="track-panel-heading"><strong>Tracked naartjies</strong><span>{tracks.length} active/retained</span></div>
          {tracks.length === 0 ? <div className="track-empty">Track details appear here without covering the fruit.</div> : tracks.map((track) => (
            <TrackDetails key={track.trackId} track={track} targetViews={config?.capture.target_views ?? 4} />
          ))}
        </aside>
      </div>
    </section>
  )
}

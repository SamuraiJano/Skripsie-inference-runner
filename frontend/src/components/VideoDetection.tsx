import { CircleStop, Clock3, Crosshair, Film, Layers3, Play, RotateCcw, ScanLine } from 'lucide-react'
import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import { api, errorMessage } from '../api'
import type { InferenceResult, StoredModel } from '../types'
import { boxColors, DetectionOverlay } from './InferenceWorkbench'

interface Props {
  models: StoredModel[]
  online: boolean
  onError: (message: string) => void
}

const maximumFrameDimension = 1920

export function VideoDetection({ models, online, onError }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const videoUrlRef = useRef<string | null>(null)
  const loopTokenRef = useRef(0)
  const processedFramesRef = useRef(0)
  const runStartedRef = useRef(0)
  const detectionModels = useMemo(() => models.filter((model) => model.task === 'yolo-detect'), [models])
  const [selectedModelId, setSelectedModelId] = useState('')
  const [sourceReady, setSourceReady] = useState(false)
  const [sourceName, setSourceName] = useState('No video selected')
  const [running, setRunning] = useState(false)
  const [confidence, setConfidence] = useState(0.25)
  const [iou, setIou] = useState(0.5)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [processedFrames, setProcessedFrames] = useState(0)
  const [processingFps, setProcessingFps] = useState(0)

  const selectedModel = useMemo(
    () => detectionModels.find((model) => model.id === selectedModelId),
    [detectionModels, selectedModelId],
  )

  useEffect(() => {
    if (selectedModelId && detectionModels.some((model) => model.id === selectedModelId)) return
    setSelectedModelId(detectionModels.find((model) => model.active)?.id ?? detectionModels[0]?.id ?? '')
  }, [detectionModels, selectedModelId])

  useEffect(() => () => {
    loopTokenRef.current += 1
    videoRef.current?.pause()
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
  }, [])

  function stop(pauseVideo = true) {
    loopTokenRef.current += 1
    setRunning(false)
    if (pauseVideo) videoRef.current?.pause()
  }

  function resetRun() {
    stop()
    const video = videoRef.current
    if (video) video.currentTime = 0
    processedFramesRef.current = 0
    setProcessedFrames(0)
    setProcessingFps(0)
    setResult(null)
  }

  function chooseVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (!file.type.startsWith('video/')) {
      onError('Choose a compatible local video file.')
      return
    }
    stop()
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    const url = URL.createObjectURL(file)
    videoUrlRef.current = url
    const video = videoRef.current
    if (video) {
      video.src = url
      video.load()
    }
    setSourceReady(false)
    setSourceName(file.name)
    setResult(null)
    processedFramesRef.current = 0
    setProcessedFrames(0)
    setProcessingFps(0)
  }

  async function processFrame(token: number) {
    if (token !== loopTokenRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.ended || video.paused) {
      if (token === loopTokenRef.current) setRunning(false)
      return
    }

    try {
      const scale = Math.min(1, maximumFrameDimension / Math.max(video.videoWidth, video.videoHeight))
      canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
      const context = canvas.getContext('2d', { alpha: false })
      if (!context) throw new Error('The browser could not create a video capture canvas.')
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      const frame = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.86))
      if (!frame) throw new Error('The browser could not encode the current video frame.')
      const nextResult = await api.infer(selectedModelId, frame, confidence, iou)
      if (token !== loopTokenRef.current) return
      setResult(nextResult)
      processedFramesRef.current += 1
      setProcessedFrames(processedFramesRef.current)
      const elapsedSeconds = Math.max(0.001, (performance.now() - runStartedRef.current) / 1000)
      setProcessingFps(processedFramesRef.current / elapsedSeconds)
      window.requestAnimationFrame(() => void processFrame(token))
    } catch (error) {
      if (token !== loopTokenRef.current) return
      setRunning(false)
      loopTokenRef.current += 1
      video.pause()
      onError(errorMessage(error))
    }
  }

  async function start() {
    const video = videoRef.current
    if (!video || !sourceReady || !selectedModel) return
    if (video.ended) video.currentTime = 0
    processedFramesRef.current = 0
    setProcessedFrames(0)
    setProcessingFps(0)
    runStartedRef.current = performance.now()
    const token = loopTokenRef.current + 1
    loopTokenRef.current = token
    setRunning(true)
    try {
      await video.play()
      void processFrame(token)
    } catch (error) {
      setRunning(false)
      loopTokenRef.current += 1
      onError(errorMessage(error))
    }
  }

  return (
    <section className="panel video-detection" data-testid="video-detection-screen" aria-labelledby="video-detection-title">
      <div className="panel-heading video-detection-heading">
        <div>
          <span className="eyebrow">Object detection only</span>
          <h2 id="video-detection-title">Video inference</h2>
        </div>
        <div className="video-actions">
          <label className={`button secondary ${running ? 'disabled' : ''}`}>
            <Film size={15} /> Choose video
            <input data-testid="video-file-input" type="file" accept="video/*" onChange={chooseVideo} disabled={running} />
          </label>
          <button data-testid="run-video-detection" className="button primary" onClick={() => { if (running) stop(); else void start() }} disabled={!online || !sourceReady || !selectedModel}>
            {running ? <><CircleStop size={15} /> Stop</> : <><Play size={15} fill="currentColor" /> Run detection</>}
          </button>
          <button className="icon-button" onClick={resetRun} disabled={!sourceReady} aria-label="Reset video inference"><RotateCcw size={15} /></button>
        </div>
      </div>

      <div className="video-detection-toolbar">
        <label className="video-model-select">
          <span>Detection model</span>
          <select
            value={selectedModelId}
            onChange={(event) => { stop(); setSelectedModelId(event.target.value); setResult(null) }}
            disabled={running || detectionModels.length === 0}
          >
            {detectionModels.length === 0 && <option value="">No object-detection model installed</option>}
            {detectionModels.map((model) => <option value={model.id} key={model.id}>{model.name} · {model.format.toUpperCase()} · {model.accelerator}</option>)}
          </select>
        </label>
        <label className="range-control">
          <span>Confidence <strong>{confidence.toFixed(2)}</strong></span>
          <input type="range" min="0.05" max="0.95" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} disabled={running} />
        </label>
        <label className="range-control">
          <span>NMS IoU <strong>{iou.toFixed(2)}</strong></span>
          <input type="range" min="0.1" max="0.9" step="0.05" value={iou} onChange={(event) => setIou(Number(event.target.value))} disabled={running} />
        </label>
      </div>

      {detectionModels.length === 0 && (
        <div className="video-model-warning"><Layers3 size={16} /> Import a model with the <strong>Object detection</strong> task before running video inference.</div>
      )}

      <div className="video-detection-layout">
        <div>
          <div className="video-detection-stage">
            <video
              ref={videoRef}
              controls
              muted
              playsInline
              preload="metadata"
              onLoadedData={() => setSourceReady(true)}
              onEnded={() => stop(false)}
              onPause={() => { if (running) stop(false) }}
            />
            {result && <DetectionOverlay detections={result.detections} width={result.sourceWidth} height={result.sourceHeight} />}
            {!sourceReady && <div className="video-detection-placeholder"><Film size={32} /><strong>Choose a local video</strong><span>Frames are decoded in the browser and sent sequentially to the selected detection model.</span></div>}
            {running && !result && <div className="video-starting"><ScanLine size={30} /><strong>Processing first frame</strong></div>}
          </div>
          <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
          <div className="source-caption"><span className={running ? 'live-dot' : ''} />{sourceName}</div>
        </div>

        <aside className="video-results" aria-live="polite">
          <div className="video-metrics">
            <div><Crosshair size={15} /><span>Detections</span><strong>{result?.detections.length ?? 0}</strong></div>
            <div><Clock3 size={15} /><span>Frame latency</span><strong>{result ? `${result.timing.backendTotalMs.toFixed(1)} ms` : '—'}</strong></div>
            <div><ScanLine size={15} /><span>Processing rate</span><strong>{processingFps.toFixed(1)} FPS</strong></div>
            <div><Film size={15} /><span>Frames processed</span><strong data-testid="video-frames-processed">{processedFrames}</strong></div>
          </div>

          <div className="video-result-heading"><strong>Current frame</strong><span>{selectedModel?.name ?? 'No model selected'}</span></div>
          {!result || result.detections.length === 0 ? (
            <div className="video-result-empty">{result ? 'No objects above the confidence threshold.' : 'Detection results will appear here while the video runs.'}</div>
          ) : (
            <div className="video-result-list">
              {result.detections.map((detection, index) => (
                <div key={`${detection.classId}-${index}`}>
                  <span className="class-swatch" style={{ background: boxColors[detection.classId % boxColors.length] }} />
                  <strong>{detection.label}</strong>
                  <span>{(detection.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>
    </section>
  )
}

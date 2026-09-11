import { CircleStop, Clock3, Crosshair, Film, Layers3, Play, RotateCcw, ScanLine } from 'lucide-react'
import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import { api, errorMessage } from '../api'
import type { GroundingMode, InferenceResult, StoredModel } from '../types'
import { boxColors, DetectionOverlay, PointOverlay } from './InferenceWorkbench'

interface Props {
  models: StoredModel[]
  online: boolean
  onError: (message: string) => void
}

const maximumFrameDimension = 1920
const groundingModeOptions: { value: GroundingMode; label: string }[] = [
  { value: 'detect', label: 'Detect named categories' },
  { value: 'ground-single', label: 'Ground one instance' },
  { value: 'ground-multi', label: 'Ground all matching instances' },
  { value: 'detect-text', label: 'Detect all scene text' },
  { value: 'ground-text', label: 'Locate specific text' },
  { value: 'gui-box', label: 'Ground GUI element (box)' },
  { value: 'gui-point', label: 'Ground GUI element (point)' },
  { value: 'point', label: 'Point to an object' },
]

function seekVideo(video: HTMLVideoElement, time: number) {
  return new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener('seeked', onSeeked)
      video.removeEventListener('error', onError)
    }
    const onSeeked = () => {
      cleanup()
      resolve()
    }
    const onError = () => {
      cleanup()
      reject(new Error('The browser could not seek to the next sampled frame.'))
    }
    video.addEventListener('seeked', onSeeked, { once: true })
    video.addEventListener('error', onError, { once: true })
    try {
      video.currentTime = time
    } catch (error) {
      cleanup()
      reject(error)
    }
  })
}

export function VideoDetection({ models, online, onError }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const processedCanvasRef = useRef<HTMLCanvasElement>(null)
  const videoUrlRef = useRef<string | null>(null)
  const loopTokenRef = useRef(0)
  const processedFramesRef = useRef(0)
  const runStartedRef = useRef(0)
  const videoModels = useMemo(
    () => models.filter((model) => model.task === 'yolo-detect' || model.task === 'vision-grounding'),
    [models],
  )
  const [selectedModelId, setSelectedModelId] = useState('')
  const [sourceReady, setSourceReady] = useState(false)
  const [sourceName, setSourceName] = useState('No video selected')
  const [running, setRunning] = useState(false)
  const [confidence, setConfidence] = useState(0.25)
  const [iou, setIou] = useState(0.5)
  const [groundingMode, setGroundingMode] = useState<GroundingMode>('detect')
  const [prompt, setPrompt] = useState('fruit')
  const [generationMode, setGenerationMode] = useState<'fast' | 'hybrid' | 'slow'>('hybrid')
  const [sampleInterval, setSampleInterval] = useState(1)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [processedFrames, setProcessedFrames] = useState(0)
  const [processingFps, setProcessingFps] = useState(0)

  const selectedModel = useMemo(
    () => videoModels.find((model) => model.id === selectedModelId),
    [videoModels, selectedModelId],
  )
  const isGrounding = selectedModel?.task === 'vision-grounding'
  const isLocateAnything = selectedModel?.format === 'locate-anything'
  const isGroundingDino = selectedModel?.format === 'grounding-dino'
  const promptRequired = isGroundingDino || groundingMode !== 'detect-text'
  const showsDetectionConfidence = !isGrounding || isGroundingDino
  const resultCount = (result?.detections.length ?? 0) + (result?.points.length ?? 0)
  const showProcessedFrame = running || Boolean(result)

  useEffect(() => {
    if (selectedModelId && videoModels.some((model) => model.id === selectedModelId)) return
    setSelectedModelId(videoModels.find((model) => model.active)?.id ?? videoModels[0]?.id ?? '')
  }, [videoModels, selectedModelId])

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
    // Exactly one request is in flight. Awaiting inference before scheduling
    // the next animation frame provides backpressure instead of building an
    // unbounded queue when model inference is slower than video playback.
    if (token !== loopTokenRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.ended || (!isGrounding && video.paused)) {
      if (token === loopTokenRef.current) setRunning(false)
      return
    }

    try {
      const groundingFrameDimension = Math.max(selectedModel?.inputWidth ?? 1280, selectedModel?.inputHeight ?? 1280)
      const frameLimit = isGrounding ? Math.min(maximumFrameDimension, groundingFrameDimension) : maximumFrameDimension
      const scale = Math.min(1, frameLimit / Math.max(video.videoWidth, video.videoHeight))
      canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
      const context = canvas.getContext('2d', { alpha: false })
      if (!context) throw new Error('The browser could not create a video capture canvas.')
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      const frame = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.86))
      if (!frame) throw new Error('The browser could not encode the current video frame.')
      const nextResult = await api.infer(
        selectedModelId,
        frame,
        confidence,
        iou,
        isGrounding ? { prompt: prompt.trim(), groundingMode: isGroundingDino ? 'detect' : groundingMode, generationMode } : undefined,
      )
      if (token !== loopTokenRef.current) return
      const processedCanvas = processedCanvasRef.current
      if (processedCanvas) {
        // Freeze the exact pixels that produced nextResult. Drawing overlays on
        // the still-playing <video> would make boxes appear on a later frame.
        processedCanvas.width = canvas.width
        processedCanvas.height = canvas.height
        processedCanvas.getContext('2d', { alpha: false })?.drawImage(canvas, 0, 0)
      }
      setResult(nextResult)
      processedFramesRef.current += 1
      setProcessedFrames(processedFramesRef.current)
      const elapsedSeconds = Math.max(0.001, (performance.now() - runStartedRef.current) / 1000)
      setProcessingFps(processedFramesRef.current / elapsedSeconds)
      if (isGrounding) {
        // Grounding models are expensive, so pause and seek by a fixed media-
        // time interval. Ordinary detection instead follows normal playback.
        const nextTime = video.currentTime + sampleInterval
        if (!Number.isFinite(video.duration) || nextTime >= video.duration) {
          setRunning(false)
          return
        }
        await seekVideo(video, nextTime)
        if (token !== loopTokenRef.current) return
      }
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
    const token = loopTokenRef.current + 1
    loopTokenRef.current = token
    setRunning(true)
    try {
      if (!selectedModel.active) await api.activateModel(selectedModel.id)
      if (token !== loopTokenRef.current) return
      runStartedRef.current = performance.now()
      setResult(null)
      if (isGrounding) video.pause()
      else await video.play()
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
          <span className="eyebrow">Object detection + visual grounding</span>
          <h2 id="video-detection-title">Video inference</h2>
        </div>
        <div className="video-actions">
          <label className={`button secondary ${running ? 'disabled' : ''}`}>
            <Film size={15} /> Choose video
            <input data-testid="video-file-input" type="file" accept="video/*" onChange={chooseVideo} disabled={running} />
          </label>
          <button data-testid="run-video-detection" className="button primary" onClick={() => { if (running) stop(); else void start() }} disabled={!online || !sourceReady || !selectedModel || (isGrounding && promptRequired && !prompt.trim())}>
            {running ? <><CircleStop size={15} /> Stop</> : <><Play size={15} fill="currentColor" /> {isGrounding ? 'Run grounding' : 'Run detection'}</>}
          </button>
          <button className="icon-button" onClick={resetRun} disabled={!sourceReady} aria-label="Reset video inference"><RotateCcw size={15} /></button>
        </div>
      </div>

      <div className={`video-detection-toolbar ${isGrounding ? 'grounding' : ''}`}>
        <label className="video-control video-model-select">
          <span>Video model</span>
          <select
            value={selectedModelId}
            onChange={(event) => { stop(); setSelectedModelId(event.target.value); setResult(null) }}
            disabled={running || videoModels.length === 0}
          >
            {videoModels.length === 0 && <option value="">No compatible video model installed</option>}
            {videoModels.map((model) => <option value={model.id} key={model.id}>{model.name} · {model.format.toUpperCase()} · {model.accelerator}</option>)}
          </select>
        </label>
        {!isGrounding && <label className="range-control">
          <span>Confidence <strong>{confidence.toFixed(2)}</strong></span>
          <input type="range" min="0.05" max="0.95" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} disabled={running} />
        </label>}
        {!isGrounding && <label className="range-control">
          <span>NMS IoU <strong>{iou.toFixed(2)}</strong></span>
          <input type="range" min="0.1" max="0.9" step="0.05" value={iou} onChange={(event) => setIou(Number(event.target.value))} disabled={running} />
        </label>}
        {isGrounding && <label className="video-control">
          <span>Grounding task</span>
          <select value={isGroundingDino ? 'detect' : groundingMode} onChange={(event) => { setGroundingMode(event.target.value as GroundingMode); setResult(null) }} disabled={running || isGroundingDino}>
            {(isGroundingDino ? groundingModeOptions.slice(0, 1) : groundingModeOptions).map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
          </select>
        </label>}
        {isLocateAnything && <label className="video-control">
          <span>Decoding</span>
          <select value={generationMode} onChange={(event) => { setGenerationMode(event.target.value as 'fast' | 'hybrid' | 'slow'); setResult(null) }} disabled={running}>
            <option value="hybrid">Hybrid (recommended)</option>
            <option value="fast">Fast / parallel</option>
            <option value="slow">Slow / precise</option>
          </select>
        </label>}
        {isGroundingDino && <label className="range-control">
          <span>Box threshold <strong>{confidence.toFixed(2)}</strong></span>
          <input type="range" min="0.05" max="0.95" step="0.05" value={confidence} onChange={(event) => { setConfidence(Number(event.target.value)); setResult(null) }} disabled={running} />
        </label>}
        {isGrounding && <label className="video-control video-grounding-prompt">
          <span>{isGroundingDino || groundingMode === 'detect' ? 'Categories (comma separated)' : 'Natural-language prompt'}</span>
          <input
            value={prompt}
            maxLength={1000}
            disabled={running || !promptRequired}
            onChange={(event) => { setPrompt(event.target.value); setResult(null) }}
            placeholder={isGroundingDino || groundingMode === 'detect' ? 'fruit, apple, banana, orange' : 'all ripe fruit on the conveyor'}
          />
          <small>For fruit, start with <strong>fruit</strong> or list exact categories such as <strong>apple, banana, orange</strong>.</small>
        </label>}
        {isGrounding && <label className="range-control video-sample-interval">
          <span>Sample every <strong>{sampleInterval.toFixed(2)} s</strong></span>
          <input type="range" min="0.25" max="5" step="0.25" value={sampleInterval} onChange={(event) => setSampleInterval(Number(event.target.value))} disabled={running} />
        </label>}
      </div>

      {videoModels.length === 0 && (
        <div className="video-model-warning"><Layers3 size={16} /> Import an <strong>object detection</strong> or <strong>visual grounding</strong> model before running video inference.</div>
      )}

      <div className="video-detection-layout">
        <div>
          <div className="video-detection-stage">
            <video
              ref={videoRef}
              className={showProcessedFrame ? 'processed-source-video' : undefined}
              controls
              muted
              playsInline
              preload="metadata"
              onLoadedData={() => setSourceReady(true)}
              onEnded={() => { if (!isGrounding) stop(false) }}
              onPause={() => { if (running && !isGrounding) stop(false) }}
            />
            <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
            <canvas ref={processedCanvasRef} className={showProcessedFrame ? 'processed-video-frame' : 'capture-canvas'} aria-label="Latest processed video frame" />
            {result && <DetectionOverlay detections={result.detections} width={result.sourceWidth} height={result.sourceHeight} />}
            {result && <PointOverlay points={result.points} width={result.sourceWidth} height={result.sourceHeight} />}
            {!sourceReady && <div className="video-detection-placeholder"><Film size={32} /><strong>Choose a local video</strong><span>Frames are decoded in the browser and sent sequentially to the selected detection or grounding model.</span></div>}
            {running && !result && <div className="video-starting"><ScanLine size={30} /><strong>Processing first frame</strong></div>}
          </div>
          <div className="source-caption"><span className={running ? 'live-dot' : ''} />{sourceName}</div>
        </div>

        <aside className="video-results" aria-live="polite">
          <div className="video-metrics">
            <div><Crosshair size={15} /><span>{isGrounding ? 'Regions / points' : 'Detections'}</span><strong>{resultCount}</strong></div>
            <div><Clock3 size={15} /><span>Frame latency</span><strong>{result ? `${result.timing.backendTotalMs.toFixed(1)} ms` : '—'}</strong></div>
            <div><ScanLine size={15} /><span>Processing rate</span><strong>{processingFps.toFixed(1)} FPS</strong></div>
            <div><Film size={15} /><span>Frames processed</span><strong data-testid="video-frames-processed">{processedFrames}</strong></div>
          </div>

          <div className="video-result-heading"><strong>Current frame</strong><span>{selectedModel?.name ?? 'No model selected'}</span></div>
          {!result || resultCount === 0 ? (
            <div className="video-result-empty">{result ? (isGrounding ? 'No matching regions or points were returned.' : 'No objects above the confidence threshold.') : 'Inference results will appear here while the video runs.'}</div>
          ) : (
            <div className="video-result-list">
              {result.detections.map((detection, index) => (
                <div key={`${detection.classId}-${index}`}>
                  <span className="class-swatch" style={{ background: boxColors[detection.classId % boxColors.length] }} />
                  <strong>{detection.label}</strong>
                  <span>{showsDetectionConfidence ? `${(detection.confidence * 100).toFixed(1)}%` : 'BOX'}</span>
                </div>
              ))}
              {result.points.map((point, index) => (
                <div key={`${point.label}-${index}`}>
                  <span className="class-swatch" style={{ background: boxColors[point.classId % boxColors.length] }} />
                  <strong>{point.label}</strong>
                  <span>POINT</span>
                </div>
              ))}
            </div>
          )}
          {isLocateAnything && result?.answer && <details className="video-grounding-answer"><summary>Raw Locate Anything answer</summary><code>{result.answer}</code></details>}
        </aside>
      </div>
    </section>
  )
}

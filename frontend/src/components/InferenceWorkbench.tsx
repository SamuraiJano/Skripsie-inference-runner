import { Clock3, Crosshair, ImagePlus, Layers3, Play, ScanLine, Sparkles, X } from 'lucide-react'
import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from 'react'
import { api, errorMessage } from '../api'
import type { InferenceResult, StoredModel } from '../types'

interface Props {
  activeModel: StoredModel | undefined
  online: boolean
  onError: (message: string) => void
}

const boxColors = ['#b8f26a', '#ffb84d', '#66d9ff', '#dca6ff', '#ff7f88']

function percent(value: number, total: number) {
  return `${Math.max(0, Math.min(100, (value / Math.max(1, total)) * 100))}%`
}

export function InferenceWorkbench({ activeModel, online, onError }: Props) {
  const [image, setImage] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [confidence, setConfidence] = useState(0.25)
  const [iou, setIou] = useState(0.5)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    if (!image) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(image)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [image])

  useEffect(() => setResult(null), [activeModel?.id, image])

  const fps = useMemo(() => result ? 1000 / Math.max(0.01, result.timing.totalMs) : 0, [result])

  function chooseFile(file?: File) {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      onError('Choose a JPG, PNG, or WebP image.')
      return
    }
    setImage(file)
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0])
    event.target.value = ''
  }

  function onDrop(event: DragEvent) {
    event.preventDefault()
    setDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  async function runInference() {
    if (!image || !activeModel) return
    setRunning(true)
    try {
      setResult(await api.infer(activeModel.id, image, confidence, iou))
    } catch (error) {
      onError(errorMessage(error))
    } finally {
      setRunning(false)
    }
  }

  return (
    <section className="panel workbench" aria-labelledby="workbench-title">
      <div className="panel-heading workbench-heading">
        <div>
          <span className="eyebrow">Single-image test</span>
          <h2 id="workbench-title">Inference workbench</h2>
        </div>
        {result && (
          <div className="latency-chip"><Clock3 size={14} /> {result.timing.totalMs.toFixed(1)} ms <span>{fps.toFixed(1)} FPS</span></div>
        )}
      </div>

      {!previewUrl ? (
        <label
          className={`image-dropzone ${dragging ? 'dragging' : ''}`}
          onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <span className="dropzone-icon"><ImagePlus size={27} /></span>
          <strong>Drop an image here</strong>
          <span>or click to choose JPG, PNG, or WebP</span>
          <small>Images stay on this device.</small>
          <input type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
        </label>
      ) : (
        <div className="preview-stage">
          <img src={previewUrl} alt="Selected inference input" />
          {result?.detections.map((detection, index) => {
            const color = boxColors[detection.classId % boxColors.length]
            return (
              <div
                className="detection-box"
                key={`${detection.classId}-${index}`}
                style={{
                  left: percent(detection.x, result.sourceWidth),
                  top: percent(detection.y, result.sourceHeight),
                  width: percent(detection.width, result.sourceWidth),
                  height: percent(detection.height, result.sourceHeight),
                  borderColor: color,
                }}
              >
                <span style={{ backgroundColor: color }}>{detection.label} {(detection.confidence * 100).toFixed(0)}%</span>
              </div>
            )
          })}
          {running && <div className="inference-overlay"><ScanLine size={34} /><strong>Running on device</strong></div>}
          <button className="preview-remove" onClick={() => setImage(null)} aria-label="Remove selected image"><X size={16} /></button>
        </div>
      )}

      <div className="control-strip">
        <label className="range-control">
          <span>Confidence <strong>{confidence.toFixed(2)}</strong></span>
          <input type="range" min="0.05" max="0.95" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} />
        </label>
        <label className="range-control">
          <span>NMS IoU <strong>{iou.toFixed(2)}</strong></span>
          <input type="range" min="0.1" max="0.9" step="0.05" value={iou} onChange={(event) => setIou(Number(event.target.value))} />
        </label>
        <button
          className="button primary run-button"
          disabled={!image || !activeModel || !online || running}
          onClick={runInference}
          data-testid="run-inference"
        >
          {running ? <span className="button-loader" /> : <Play size={16} fill="currentColor" />}
          {running ? 'Running…' : 'Run inference'}
        </button>
      </div>

      {!activeModel && <div className="workbench-hint"><Layers3 size={16} /> Add and activate a model to enable inference.</div>}

      {result && (
        <div className="results-section" data-testid="inference-results">
          <div className="result-summary">
            <div><Crosshair size={16} /><span>Detections</span><strong>{result.detections.length}</strong></div>
            <div><Sparkles size={16} /><span>Model compute</span><strong>{result.timing.inferenceMs.toFixed(1)} ms</strong></div>
            <div><ScanLine size={16} /><span>Pre + post</span><strong>{(result.timing.preprocessMs + result.timing.postprocessMs).toFixed(1)} ms</strong></div>
          </div>

          {result.detections.length > 0 && (
            <div className="result-list">
              {result.detections.map((detection, index) => (
                <div className="result-row" key={`${detection.label}-${index}`}>
                  <span className="class-swatch" style={{ background: boxColors[detection.classId % boxColors.length] }} />
                  <strong>{detection.label}</strong>
                  <span>class {detection.classId}</span>
                  <div className="confidence-bar"><i style={{ width: `${detection.confidence * 100}%` }} /></div>
                  <b>{(detection.confidence * 100).toFixed(1)}%</b>
                </div>
              ))}
            </div>
          )}

          {result.classifications.length > 0 && (
            <div className="classification-list">
              {result.classifications.map((item) => (
                <div key={item.classId}><strong>{item.label}</strong><span>{(item.score * 100).toFixed(2)}%</span></div>
              ))}
            </div>
          )}

          {result.tensors.length > 0 && (
            <div className="tensor-list">
              {result.tensors.map((tensor) => (
                <details key={tensor.name}>
                  <summary><strong>{tensor.name}</strong><span>[{tensor.shape.join(', ')}] · {tensor.totalValues} values</span></summary>
                  <code>{tensor.values.map((value) => value.toFixed(5)).join(', ')}</code>
                </details>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}


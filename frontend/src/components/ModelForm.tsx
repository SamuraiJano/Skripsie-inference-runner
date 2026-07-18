import { ChevronDown, Cpu, FileArchive, Info, Upload, X } from 'lucide-react'
import { FormEvent, useMemo, useState } from 'react'
import type { ModelManifestDraft, ModelTask } from '../types'

interface Props {
  open: boolean
  submitting: boolean
  onClose: () => void
  onSubmit: (manifest: ModelManifestDraft, param: File, bin: File) => Promise<void>
}

function FilePicker({ label, accept, file, onChange }: {
  label: string
  accept: string
  file: File | null
  onChange: (file: File | null) => void
}) {
  return (
    <label className={`file-picker ${file ? 'selected' : ''}`}>
      <FileArchive size={20} />
      <span><strong>{file?.name ?? label}</strong><small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : accept}</small></span>
      <Upload size={16} />
      <input type="file" accept={accept} onChange={(event) => onChange(event.target.files?.[0] ?? null)} required />
    </label>
  )
}

export function ModelForm({ open, submitting, onClose, onSubmit }: Props) {
  const [name, setName] = useState('')
  const [task, setTask] = useState<ModelTask>('yolo-detect')
  const [paramFile, setParamFile] = useState<File | null>(null)
  const [binFile, setBinFile] = useState<File | null>(null)
  const [labels, setLabels] = useState('')
  const [inputSize, setInputSize] = useState(640)
  const [threads, setThreads] = useState(4)
  const [inputBlob, setInputBlob] = useState('in0')
  const [outputBlobs, setOutputBlobs] = useState('out0')
  const [showAdvanced, setShowAdvanced] = useState(false)

  const parsedLabels = useMemo(() => labels.split(/[\n,]/).map((value) => value.trim()).filter(Boolean), [labels])

  if (!open) return null

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!paramFile || !binFile) return
    const manifest: ModelManifestDraft = {
      name: name.trim(),
      task,
      input: {
        width: inputSize,
        height: inputSize,
        blob: inputBlob.trim(),
        mean: [0, 0, 0],
        norm: [1 / 255, 1 / 255, 1 / 255],
      },
      outputs: outputBlobs.split(',').map((value) => value.trim()).filter(Boolean),
      labels: parsedLabels,
      runtime: { threads },
      classification: { applySoftmax: false, topK: 5 },
    }
    await onSubmit(manifest, paramFile, binFile)
  }

  const labelsRequired = task === 'yolo-detect'

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="model-drawer" role="dialog" aria-modal="true" aria-labelledby="add-model-title">
        <div className="drawer-heading">
          <div>
            <span className="eyebrow">NCNN model pair</span>
            <h2 id="add-model-title">Add a model</h2>
            <p>The model is stored locally and only loaded into RAM when active.</p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close model form"><X size={19} /></button>
        </div>

        <form onSubmit={submit}>
          <div className="form-grid two">
            <label className="field">
              <span>Model name</span>
              <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Naartjie detector" required />
            </label>
            <label className="field">
              <span>Output parser</span>
              <div className="select-wrap">
                <select value={task} onChange={(event) => setTask(event.target.value as ModelTask)}>
                  <option value="yolo-detect">YOLO detection</option>
                  <option value="classification">Classification</option>
                  <option value="raw">Raw tensors</option>
                </select>
                <ChevronDown size={15} />
              </div>
            </label>
          </div>

          <div className="form-grid two file-grid">
            <FilePicker label="Select .param file" accept=".param" file={paramFile} onChange={setParamFile} />
            <FilePicker label="Select .bin file" accept=".bin" file={binFile} onChange={setBinFile} />
          </div>

          <label className="field">
            <span>Class labels {labelsRequired ? <em>required for YOLO</em> : <em>optional</em>}</span>
            <textarea
              value={labels}
              onChange={(event) => setLabels(event.target.value)}
              placeholder={'healthy\nblemish\nrot'}
              rows={4}
              required={labelsRequired}
            />
            <small>{parsedLabels.length} labels · one per line</small>
          </label>

          <button type="button" className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)} aria-expanded={showAdvanced}>
            <Cpu size={16} /> Runtime and tensor settings <ChevronDown size={15} className={showAdvanced ? 'rotated' : ''} />
          </button>

          {showAdvanced && (
            <div className="advanced-panel">
              <div className="form-grid two">
                <label className="field">
                  <span>Square input size</span>
                  <input type="number" min="32" max="4096" step="32" value={inputSize} onChange={(event) => setInputSize(Number(event.target.value))} />
                </label>
                <label className="field">
                  <span>Inference threads</span>
                  <input type="number" min="1" max="4" value={threads} onChange={(event) => setThreads(Number(event.target.value))} />
                </label>
                <label className="field">
                  <span>Input blob</span>
                  <input value={inputBlob} onChange={(event) => setInputBlob(event.target.value)} required />
                </label>
                <label className="field">
                  <span>Output blobs</span>
                  <input value={outputBlobs} onChange={(event) => setOutputBlobs(event.target.value)} required />
                </label>
              </div>
              <div className="inline-note"><Info size={15} /> Defaults match standard Ultralytics NCNN exports. Change them only when your graph uses different blob names.</div>
            </div>
          )}

          <div className="drawer-actions">
            <button type="button" className="button secondary" onClick={onClose}>Cancel</button>
            <button className="button primary" disabled={submitting || (labelsRequired && parsedLabels.length === 0)}>
              {submitting ? <span className="button-loader" /> : <Upload size={16} />}
              {submitting ? 'Importing…' : 'Import model'}
            </button>
          </div>
        </form>
      </section>
    </div>
  )
}


import { FileArchive, Info, Upload, X } from 'lucide-react'
import { FormEvent, useMemo, useState } from 'react'
import type { DirectModelDraft, ModelFormatCapability, ModelManifestDraft, ModelPrecision, ModelTask } from '../types'

interface Props {
  open: boolean
  submitting: boolean
  formats: ModelFormatCapability[]
  onClose: () => void
  onSubmitArtifact: (draft: DirectModelDraft, model: File) => Promise<void>
  onSubmitNcnn: (manifest: ModelManifestDraft, param: File, bin: File) => Promise<void>
}

type ImportMode = 'direct' | 'ncnn'

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

function validateManifest(value: unknown): ModelManifestDraft {
  if (!value || typeof value !== 'object') throw new Error('Manifest must be a JSON object.')
  const manifest = value as Partial<ModelManifestDraft>
  if (manifest.schemaVersion !== 2) throw new Error('Manifest schemaVersion must be 2.')
  if (!manifest.name || !manifest.task || !manifest.precision) throw new Error('Manifest name, task, and precision are required.')
  if (manifest.framework !== 'ncnn' || manifest.runtime?.name !== 'ncnn') throw new Error('The package must describe an NCNN model.')
  if (manifest.runtime?.useVulkanCompute !== true) throw new Error('NCNN packages require runtime.useVulkanCompute=true.')
  if ((manifest.runtime?.gpuDevice ?? -1) < -1) throw new Error('runtime.gpuDevice must be -1 (automatic) or a Vulkan device index.')
  if (manifest.task === 'yolo-segment' && (manifest.outputs?.length ?? 0) < 2) {
    throw new Error('YOLO segmentation requires detection and prototype outputs.')
  }
  for (const field of ['checkpointSha256', 'datasetHash', 'splitHash', 'classSchemaHash'] as const) {
    if (!/^[0-9a-f]{64}$/i.test(String(manifest[field] ?? ''))) throw new Error(`${field} must be a SHA-256 digest.`)
  }
  if (!manifest.artifactVerification?.verified) throw new Error('Artifact verification must be recorded by the export harness.')
  return manifest as ModelManifestDraft
}

function extensionFor(filename: string) {
  const lower = filename.toLowerCase()
  if (lower.endsWith('.torchscript')) return '.torchscript'
  if (lower.endsWith('_edgetpu.tflite')) return '.tflite'
  const separator = lower.lastIndexOf('.')
  return separator < 0 ? '' : lower.slice(separator)
}

export function ModelForm({ open, submitting, formats, onClose, onSubmitArtifact, onSubmitNcnn }: Props) {
  const [mode, setMode] = useState<ImportMode>('direct')
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [task, setTask] = useState<Exclude<ModelTask, 'raw'>>('yolo-detect')
  const [precision, setPrecision] = useState<ModelPrecision>('fp32')
  const [inputSize, setInputSize] = useState(640)
  const [trusted, setTrusted] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const [manifestFile, setManifestFile] = useState<File | null>(null)
  const [manifest, setManifest] = useState<ModelManifestDraft | null>(null)
  const [manifestError, setManifestError] = useState<string | null>(null)
  const [paramFile, setParamFile] = useState<File | null>(null)
  const [binFile, setBinFile] = useState<File | null>(null)

  const directFormats = useMemo(() => formats.filter((format) => format.format !== 'ncnn'), [formats])
  const acceptedExtensions = useMemo(
    () => Array.from(new Set(directFormats.filter((format) => format.available).flatMap((format) => format.extensions))).join(','),
    [directFormats],
  )

  if (!open) return null

  function chooseModel(file: File | null) {
    setModelFile(file)
    setFileError(null)
    if (!file) return
    const extension = extensionFor(file.name)
    const matches = directFormats.filter((format) => format.extensions.includes(extension))
    if (matches.length === 0) {
      setFileError(`No installed backend recognises ${extension || 'this file type'}.`)
    } else if (!matches.some((format) => format.available)) {
      setFileError(`${matches.map((format) => format.runtime).join(' / ')} is not installed on this laptop.`)
    }
    if (!name) setName(file.name.replace(/\.(torchscript|onnx|engine|tflite|mnn|pte|xml|pb|pt|zip)$/i, ''))
  }

  async function chooseManifest(file: File | null) {
    setManifestFile(file)
    setManifest(null)
    setManifestError(null)
    if (!file) return
    try {
      setManifest(validateManifest(JSON.parse(await file.text())))
    } catch (error) {
      setManifestError(error instanceof Error ? error.message : 'Manifest could not be read.')
    }
  }

  async function submitDirect(event: FormEvent) {
    event.preventDefault()
    if (!modelFile || fileError || !name.trim()) return
    const executable = ['.pt', '.torchscript', '.pte'].includes(extensionFor(modelFile.name))
    if (executable && !trusted) {
      setFileError('Confirm that executable model checkpoints come from a trusted source.')
      return
    }
    await onSubmitArtifact({
      name: name.trim(), task, precision, trustedCheckpoint: trusted,
      inputWidth: inputSize, inputHeight: inputSize,
    }, modelFile)
  }

  async function submitNcnn(event: FormEvent) {
    event.preventDefault()
    if (!manifest || !paramFile || !binFile) return
    await onSubmitNcnn(manifest, paramFile, binFile)
  }

  const selectedExtension = extensionFor(modelFile?.name ?? '')
  const requiresTrust = ['.pt', '.torchscript', '.pte'].includes(selectedExtension)

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="model-drawer" role="dialog" aria-modal="true" aria-labelledby="add-model-title">
        <div className="drawer-heading">
          <div>
            <span className="eyebrow">Automatic backend selection</span>
            <h2 id="add-model-title">Add a model</h2>
            <p>Import a compatible artifact. The runner validates it against installed runtimes before storing it.</p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close model form"><X size={19} /></button>
        </div>

        <div className="import-tabs" role="tablist" aria-label="Model import type">
          <button className={mode === 'direct' ? 'active' : ''} onClick={() => setMode('direct')} role="tab">Model file</button>
          <button className={mode === 'ncnn' ? 'active' : ''} onClick={() => setMode('ncnn')} role="tab">NCNN package</button>
        </div>

        {mode === 'direct' ? (
          <form onSubmit={(event) => void submitDirect(event)}>
            <div className="form-grid file-grid">
              <FilePicker label="Select a model artifact" accept={acceptedExtensions || '.pt,.onnx,.engine,.torchscript'} file={modelFile} onChange={chooseModel} />
            </div>
            {fileError && <div className="inline-note manifest-error"><Info size={15} /> {fileError}</div>}

            <label className="field">
              <span>Display name</span>
              <input value={name} maxLength={100} onChange={(event) => setName(event.target.value)} placeholder="My model" required />
            </label>
            <div className="form-grid two">
              <label className="field">
                <span>Task</span>
                <select value={task} onChange={(event) => setTask(event.target.value as Exclude<ModelTask, 'raw'>)}>
                  <option value="yolo-detect">Object detection</option>
                  <option value="yolo-segment">Instance segmentation</option>
                  <option value="classification">Classification</option>
                </select>
              </label>
              <label className="field">
                <span>Artifact precision</span>
                <select value={precision} onChange={(event) => setPrecision(event.target.value as ModelPrecision)}>
                  <option value="fp32">FP32</option>
                  <option value="fp16">FP16</option>
                  <option value="int8">INT8</option>
                </select>
              </label>
            </div>
            <label className="field">
              <span>Inference size <em>(square pixels)</em></span>
              <input type="number" min={32} max={4096} step={32} value={inputSize} onChange={(event) => setInputSize(Number(event.target.value))} />
            </label>

            {requiresTrust && (
              <label className="trust-check">
                <input type="checkbox" checked={trusted} onChange={(event) => setTrusted(event.target.checked)} />
                <span>I trust the source of this executable checkpoint.</span>
              </label>
            )}

            <div className="format-capabilities" aria-label="Installed model runtimes">
              {directFormats.map((format) => (
                <span className={format.available ? 'available' : 'unavailable'} key={format.format}>
                  <strong>{format.format}</strong>{format.available ? `${format.runtime} · ${format.accelerator}` : `${format.runtime} missing`}
                </span>
              ))}
            </div>

            <div className="inline-note"><Info size={15} /> Folder-based exports such as OpenVINO or SavedModel must be uploaded as a ZIP preserving the export directory.</div>
            <div className="drawer-actions">
              <button type="button" className="button secondary" onClick={onClose}>Cancel</button>
              <button className="button primary" disabled={submitting || !modelFile || Boolean(fileError) || !name.trim()}>
                {submitting ? <span className="button-loader" /> : <Upload size={16} />}
                {submitting ? 'Validating…' : 'Validate and import'}
              </button>
            </div>
          </form>
        ) : (
          <form onSubmit={(event) => void submitNcnn(event)}>
            <div className="form-grid file-grid">
              <FilePicker label="Select manifest.json" accept=".json,application/json" file={manifestFile} onChange={(file) => void chooseManifest(file)} />
            </div>
            {manifestError && <div className="inline-note manifest-error"><Info size={15} /> {manifestError}</div>}
            {manifest && (
              <div className="manifest-summary">
                <div><span>Name</span><strong>{manifest.name}</strong></div>
                <div><span>Task</span><strong>{manifest.task}</strong></div>
                <div><span>Precision</span><strong>{manifest.precision.toUpperCase()}</strong></div>
                <div><span>Runtime</span><strong>NCNN / Vulkan GPU {manifest.runtime.gpuDevice}</strong></div>
              </div>
            )}
            <div className="form-grid two file-grid">
              <FilePicker label="Select .param file" accept=".param" file={paramFile} onChange={setParamFile} />
              <FilePicker label="Select .bin file" accept=".bin" file={binFile} onChange={setBinFile} />
            </div>
            <div className="drawer-actions">
              <button type="button" className="button secondary" onClick={onClose}>Cancel</button>
              <button className="button primary" disabled={submitting || !manifest || !paramFile || !binFile}>
                {submitting ? <span className="button-loader" /> : <Upload size={16} />}
                {submitting ? 'Importing…' : 'Import NCNN package'}
              </button>
            </div>
          </form>
        )}
      </section>
    </div>
  )
}

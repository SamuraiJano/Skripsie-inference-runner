export type ModelTask = 'classification' | 'yolo-detect' | 'raw'

export interface RuntimeHealth {
  status: 'ok'
  version: string
  engine: string
  ncnnAvailable: boolean
  architecture: string
  logicalCores: number
  recommendedThreads: number
  uptimeSeconds: number
  activeModelId: string | null
}

export interface StoredModel {
  id: string
  name: string
  task: ModelTask
  inputWidth: number
  inputHeight: number
  inputBlob: string
  outputBlobs: string[]
  threads: number
  labels: string[]
  active: boolean
  sizeBytes: number
  createdAt: string
}

export interface Detection {
  classId: number
  label: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
}

export interface ClassificationScore {
  classId: number
  label: string
  score: number
}

export interface RawTensor {
  name: string
  shape: number[]
  values: number[]
  totalValues: number
}

export interface InferenceResult {
  modelId: string
  task: ModelTask
  sourceWidth: number
  sourceHeight: number
  detections: Detection[]
  classifications: ClassificationScore[]
  tensors: RawTensor[]
  timing: {
    decodeMs: number
    preprocessMs: number
    inferenceMs: number
    postprocessMs: number
    totalMs: number
  }
}

export interface ModelManifestDraft {
  name: string
  task: ModelTask
  input: {
    width: number
    height: number
    blob: string
    mean: [number, number, number]
    norm: [number, number, number]
  }
  outputs: string[]
  labels: string[]
  runtime: {
    threads: number
  }
  classification: {
    applySoftmax: boolean
    topK: number
  }
}


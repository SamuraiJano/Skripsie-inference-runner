export type ModelTask = 'classification' | 'yolo-detect' | 'yolo-segment' | 'raw'
export type ModelPrecision = 'fp32' | 'fp16' | 'int8'

export interface ModelFormatCapability {
  format: string
  extensions: string[]
  available: boolean
  runtime: string
  accelerated: boolean
  accelerator: string
}

export interface RuntimeHealth {
  status: 'ok'
  version: string
  engine: string
  ncnnAvailable: boolean
  architecture: string
  platform: 'Windows'
  vulkanAvailable: boolean
  gpuCount: number
  gpuDeviceIndex: number
  gpuDevice: string
  logicalCores: number
  recommendedThreads: number
  uptimeSeconds: number
  activeModelId: string | null
  pythonBackendAvailable: boolean
  cudaAvailable: boolean
  cudaVersion: string | null
  torchVersion: string | null
  ultralyticsVersion: string | null
  formats: ModelFormatCapability[]
}

export interface StoredModel {
  id: string
  name: string
  task: ModelTask
  precision: ModelPrecision
  framework: string
  format: string
  artifactFile: string
  backend: string
  accelerator: string
  inputWidth: number
  inputHeight: number
  inputBlob: string
  outputBlobs: string[]
  threads: number
  labels: string[]
  active: boolean
  sizeBytes: number
  createdAt: string
  checkpointSha256: string
  datasetHash: string
  splitHash: string
  classSchemaHash: string
  calibrationId: string | null
  exporter: { name: string; version: string }
  runtime: { name: string; version: string; useVulkanCompute: boolean; gpuDevice: number }
  legacyUnverified: boolean
}

export interface DirectModelDraft {
  name: string
  task: Exclude<ModelTask, 'raw'>
  precision: ModelPrecision
  trustedCheckpoint: boolean
  inputWidth: number
  inputHeight: number
}

export interface MaskRle {
  encoding: 'rle-row-major'
  width: number
  height: number
  counts: number[]
}

export interface Detection {
  classId: number
  label: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
  trackId: number | null
  mask: MaskRle | null
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
    detectionNmsMs: number
    maskReconstructionMs: number
    trackingMs: number
    queueWaitMs: number
    gradingMs: number
    backendTotalMs: number
  }
  clientHttpRoundTripMs?: number
}

export interface ModelManifestDraft {
  name: string
  schemaVersion: 2
  task: ModelTask
  precision: ModelPrecision
  framework: 'ncnn'
  checkpointSha256: string
  datasetHash: string
  splitHash: string
  classSchemaHash: string
  exporter: { name: string; version: string }
  calibrationId: string | null
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
    name: 'ncnn'
    version: string
    threads: number
    useVulkanCompute: boolean
    gpuDevice: number
    useInt8Inference: boolean
    useInt8Storage: boolean
    useFp16Storage: boolean
    useFp16Arithmetic: boolean
  }
  artifactVerification: { verified: boolean; quantized: boolean }
  classification: {
    applySoftmax: boolean
    topK: number
  }
  maskThreshold: number
}

export interface PipelineCoordinate {
  value: number
  normalized: boolean
}

export interface PipelineConfig {
  models: {
    yolo26n_model_id: string
    segmentation_model_id: string
    yolo26n_model_path: string
    segmentation_model_path: string
  }
  zones: {
    acquisition_zone_start_x: PipelineCoordinate
    classification_line_x: PipelineCoordinate
    post_classification_zone: PipelineCoordinate
  }
  capture: {
    target_capture_percentages: number[]
    target_views: number
    minimum_frame_gap: number
    crop_margin: number
    sharpness_threshold: number
    minimum_crop_width: number
    minimum_crop_height: number
    maximum_permitted_overlap: number
  }
  classification: {
    stationary_delay_seconds: number
    timeout_seconds: number
    minimum_views: number
    confidence_threshold: number
  }
  segmentation: { queue_size: number; workers: number }
  windows: { cpu_threads: number; live_max_fps: number }
}

export type TrackState =
  | 'NEW'
  | 'ACQUIRING'
  | 'WAITING_FOR_SEGMENTATION'
  | 'WAITING_FOR_CLASSIFICATION'
  | 'CLASSIFIED'
  | 'EXITED'
  | 'FAILED'

export interface LiveTrack {
  trackId: number
  currentBoundingBox: Detection
  detectionConfidence: number
  firstSeenFrame: number
  lastSeenFrame: number
  state: TrackState
  acquisitionZoneProgress: number
  capturedViews: Array<{
    frameId: number
    viewId: number
    zoneProgress: number
    sharpness: number
    segmentationStatus: string
  }>
  segmentationResults: Array<{
    frameId: number
    viewId: number
    predictedClass: number
    confidence: number
    defectArea: number
    processingMs: number
    success: boolean
    message: string
  }>
  finalClass: string | null
  finalClassId: number | null
  finalConfidence: number
  incompleteEvidence: boolean
  classificationTimestamp: string
  messages: string[]
  viewsCaptured: number
  viewsUsed: number
  segmentationCompleted: number
  totalVisibleDefectArea: number
  classificationLatencyMs: number
}

export interface PipelineSnapshot {
  frameId: number
  sourceWidth?: number
  sourceHeight?: number
  tracks: LiveTrack[]
  queueSize: number
  droppedSegmentationJobs: number
  detectorFps: number
  segmentationLatencyMs: number
  classificationLatencyMs: number
  config: PipelineConfig
}

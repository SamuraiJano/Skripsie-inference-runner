export type ModelTask = 'classification' | 'yolo-detect' | 'yolo-segment' | 'vision-grounding' | 'raw'
export type ModelPrecision = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4' | 'unknown'

// These interfaces mirror JSON produced by the C++ public API (which may in
// turn normalize a Python model response). They document the wire contract but
// do not perform runtime validation, so backend and frontend changes must stay
// synchronized.

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
  family: string
  relativePath: string
  quantization: {
    label: string
    method: string
    quantized: boolean
  }
  runtimeAvailable: boolean
  compatible: boolean
  statusReason: string
  autoDetected: boolean
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

export interface GroundingPoint {
  classId: number
  label: string
  x: number
  y: number
}

export type GroundingMode =
  | 'detect'
  | 'ground-single'
  | 'ground-multi'
  | 'detect-text'
  | 'ground-text'
  | 'gui-box'
  | 'gui-point'
  | 'point'

export interface GroundingRequest {
  prompt: string
  groundingMode: GroundingMode
  generationMode: 'fast' | 'hybrid' | 'slow'
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
  points: GroundingPoint[]
  classifications: ClassificationScore[]
  tensors: RawTensor[]
  answer: string
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
    roi_left: PipelineCoordinate
    roi_right: PipelineCoordinate
    roi_top: PipelineCoordinate
    roi_bottom: PipelineCoordinate
    entry_zone_height: PipelineCoordinate
    exit_zone_height: PipelineCoordinate
  }
  tracking: {
    strict_entry_gating: boolean
    roi_boundary_tolerance_ratio: number
    minimum_bbox_width_ratio: number
    minimum_bbox_height_ratio: number
    minimum_bbox_area_ratio: number
    maximum_bbox_area_ratio: number
    maintenance_size_tolerance: number
    minimum_confirmation_hits: number
    motion_window_frames: number
    minimum_downward_displacement_ratio: number
    maximum_upward_displacement_ratio: number
    stationary_tolerance_ratio: number
    maximum_stationary_frames: number
    debug_diagnostics: boolean
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
  surface_mapping: {
    enabled: boolean
    shape_model: 'oblate' | 'sphere'
    rotation_primary_axis: 'x' | 'y'
    rotation_direction: number
    candidate_frame_stride: number
    maximum_candidate_views: number
    maximum_mapping_views: number
    minimum_mapping_views: number
    preferred_step_degrees: number
    maximum_step_degrees: number
    rotation_search_step_degrees: number
    secondary_axis_limit_degrees: number
    downsample_factor: number
    texture_percentile: number
    maximum_texture_points: number
    minimum_texture_points: number
    minimum_valid_point_fraction: number
    latitude_limit_degrees: number
    pre_roll_degrees: number
    post_roll_degrees: number
    target_coverage_degrees: number
    minimum_confident_coverage: number
    maximum_map_width: number
  }
}

export interface SurfaceMappingResult {
  status: 'complete' | 'partial' | 'failed' | 'fallback'
  coverage_degrees?: number
  observed_area_ratio?: number
  accepted_views?: number
  rejected_views?: number
  mean_step_degrees?: number
  standard_deviation_step_degrees?: number
  alignment_confidence?: number
  rgb_map_url?: string
  defect_overlay_url?: string
  observed_mask_url?: string
  alignment_confidence_url?: string
  failure_reasons?: string[]
  defects?: Record<string, { observed_surface_ratio: number; target_surface_ratio: number }>
  total_known_defect_coverage?: number
  incomplete_evidence?: boolean
}

export type TrackState =
  | 'TENTATIVE'
  | 'CONFIRMED'
  | 'ACQUIRING'
  | 'WAITING_FOR_SEGMENTATION'
  | 'WAITING_FOR_CLASSIFICATION'
  | 'LOST'
  | 'COMPLETED'
  | 'REJECTED'

export interface LiveTrack {
  trackId: number
  currentBoundingBox: Detection
  detectionConfidence: number
  firstSeenFrame: number
  lastSeenFrame: number
  state: TrackState
  confirmationHits: number
  confirmed: boolean
  hasCurrentObservation: boolean
  motionStatus: string
  sizeStatus: string
  acquisitionZoneProgress: number
  capturedViews: Array<{
    captureId: string
    imageUrl: string | null
    frameId: number
    viewId: number
    zoneProgress: number
    sharpness: number
    usedTrackingPrediction: boolean
    segmentationStatus: string
    cropWidth: number
    cropHeight: number
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
  classificationCompleted: boolean
  counted: boolean
  exited: boolean
  failureReason: string | null
  classificationLatencyMs: number
  candidateViews: number
  rejectedViews: number
  surfaceMapping: SurfaceMappingResult | null
  defects: Record<string, { observed_surface_ratio: number; target_surface_ratio: number }>
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
  completedCount: number
  classCounts: Record<string, number>
  trackingDiagnostics: string[]
  config: PipelineConfig
}

export interface PipelineCapture {
  captureId: string
  imageUrl: string
  runId: string
  trackId: number
  frameId: number
  viewId: number
  timestamp: string
  zoneProgress: number
  originalBoundingBox: Detection
  sharpness: number
  detectionConfidence: number
  usedTrackingPrediction: boolean
  cropWidth: number
  cropHeight: number
}

import { createServer } from 'node:http'
import { randomUUID } from 'node:crypto'

const startedAt = Date.now()
const models = [{
  id: 'demo-yolo-model',
  name: 'Demo YOLO detector',
  task: 'yolo-detect',
  precision: 'fp16',
  framework: 'ncnn',
  format: 'ncnn',
  artifactFile: '',
  backend: 'ncnn',
  accelerator: 'Vulkan',
  inputWidth: 640,
  inputHeight: 640,
  inputBlob: 'in0',
  outputBlobs: ['out0'],
  threads: 8,
  labels: ['class_1', 'class_2', 'class_3'],
  checkpointSha256: '1'.repeat(64),
  datasetHash: '2'.repeat(64),
  splitHash: '3'.repeat(64),
  classSchemaHash: '4'.repeat(64),
  calibrationId: null,
  exporter: { name: 'ultralytics', version: '8.4.95' },
  runtime: { name: 'ncnn', version: '1.0.20260526', useVulkanCompute: true, gpuDevice: 0 },
  legacyUnverified: false,
  active: true,
  sizeBytes: 12_485_120,
  createdAt: new Date().toISOString(),
}]

const pipelineConfig = {
  models: { yolo26n_model_id: 'demo-yolo-model', segmentation_model_id: 'demo-yolo-model', yolo26n_model_path: '', segmentation_model_path: '' },
  detection: { confidence_threshold: 0.25, iou_threshold: 0.5, naartjie_class_id: 0 },
  bytetrack: { high_threshold: 0.5, low_threshold: 0.1, match_iou: 0.3, max_missed_frames: 30 },
  zones: {
    acquisition_zone_start_x: { value: 0.1, normalized: true },
    classification_line_x: { value: 0.75, normalized: true },
    post_classification_zone: { value: 0.95, normalized: true },
  },
  capture: {
    target_capture_percentages: [0.2, 0.4, 0.6, 0.8], target_views: 4, minimum_frame_gap: 2,
    crop_margin: 0.1, sharpness_threshold: 60, minimum_crop_width: 32, minimum_crop_height: 32, maximum_permitted_overlap: 0.25,
  },
  segmentation: { queue_size: 16, workers: 2 },
  classification: { stationary_delay_seconds: 0.1, timeout_seconds: 2, minimum_views: 2, confidence_threshold: 0.55 },
  retention: { stale_track_frames: 45, release_after_seconds: 5 },
  output: { results_directory: 'data/results', logging_directory: 'data/logs', debug_save_images: false },
  windows: { cpu_threads: 8, live_max_fps: 30 },
}
const formats = [
  { format: 'ncnn', extensions: ['.param', '.bin'], available: true, runtime: 'NCNN', accelerated: true, accelerator: 'Vulkan' },
  { format: 'pt', extensions: ['.pt'], available: true, runtime: 'PyTorch', accelerated: true, accelerator: 'CUDA' },
  { format: 'torchscript', extensions: ['.torchscript'], available: true, runtime: 'PyTorch', accelerated: true, accelerator: 'CUDA' },
  { format: 'onnx', extensions: ['.onnx'], available: true, runtime: 'ONNX Runtime', accelerated: true, accelerator: 'CUDA' },
  { format: 'engine', extensions: ['.engine'], available: false, runtime: 'TensorRT', accelerated: false, accelerator: 'CUDA' },
]
let pipelineFrame = 0

function pipelineSnapshot() {
  if (pipelineFrame === 0) return { frameId: 0, tracks: [], queueSize: 0, droppedSegmentationJobs: 0, detectorFps: 0, segmentationLatencyMs: 0, classificationLatencyMs: 0, config: pipelineConfig }
  const progress = Math.min(1, pipelineFrame / 32)
  const views = Math.min(4, Math.floor(progress * 5))
  const classified = progress >= 0.82
  return {
    frameId: pipelineFrame, sourceWidth: 1280, sourceHeight: 720,
    tracks: [{
      trackId: 1,
      currentBoundingBox: { classId: 0, label: 'naartjie', confidence: 0.94, x: 65 + progress * 1040, y: 250, width: 150, height: 145, trackId: 1, mask: null },
      detectionConfidence: 0.94, firstSeenFrame: 1, lastSeenFrame: pipelineFrame,
      state: classified ? 'CLASSIFIED' : progress >= 0.75 ? 'WAITING_FOR_CLASSIFICATION' : 'ACQUIRING',
      acquisitionZoneProgress: progress,
      capturedViews: Array.from({ length: views }, (_, index) => ({ frameId: index * 5 + 1, viewId: index, zoneProgress: (index + 1) * 0.2, sharpness: 96, segmentationStatus: 'completed' })),
      segmentationResults: [], finalClass: classified ? 'Class 2' : null, finalClassId: classified ? 1 : null,
      finalConfidence: classified ? 0.87 : 0, incompleteEvidence: false, classificationTimestamp: classified ? new Date().toISOString() : '',
      messages: [], viewsCaptured: views, viewsUsed: classified ? 4 : 0, segmentationCompleted: views,
      totalVisibleDefectArea: classified ? 214 : 0, classificationLatencyMs: classified ? 780 : 0,
    }],
    queueSize: Math.max(0, views - 2), droppedSegmentationJobs: 0, detectorFps: 7.8,
    segmentationLatencyMs: views ? 84.2 : 0, classificationLatencyMs: classified ? 780 : 0, config: pipelineConfig,
  }
}

function json(response, status, value) {
  response.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' })
  response.end(JSON.stringify(value))
}

function drain(request) {
  return new Promise((resolve, reject) => {
    let size = 0
    request.on('data', (chunk) => {
      size += chunk.length
      if (size > 300 * 1024 * 1024) reject(new Error('request is too large'))
    })
    request.on('end', resolve)
    request.on('error', reject)
  })
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1')
  try {
    if (request.method === 'GET' && url.pathname === '/api/health') {
      return json(response, 200, {
        status: 'ok', version: '1.0.0-mock', engine: 'Multi-backend mock', ncnnAvailable: true,
        architecture: process.arch, platform: 'Windows', vulkanAvailable: true,
        gpuCount: 1, gpuDeviceIndex: 0, gpuDevice: 'NVIDIA GeForce RTX 4060 Laptop GPU',
        logicalCores: 16, recommendedThreads: 8,
        pythonBackendAvailable: true, cudaAvailable: true, cudaVersion: '12.4',
        torchVersion: '2.6.0+cu124', ultralyticsVersion: '8.4.95', formats,
        uptimeSeconds: (Date.now() - startedAt) / 1000,
        activeModelId: models.find((model) => model.active)?.id ?? null,
    })
    }
    if (request.method === 'GET' && url.pathname === '/api/capabilities') return json(response, 200, { formats })
    if (request.method === 'GET' && url.pathname === '/api/models') return json(response, 200, models)
    if (request.method === 'GET' && url.pathname === '/api/pipeline/config') return json(response, 200, pipelineConfig)
    if (request.method === 'GET' && url.pathname === '/api/pipeline') return json(response, 200, pipelineSnapshot())
    if (request.method === 'POST' && url.pathname === '/api/pipeline/reset') {
      pipelineFrame = 0
      return json(response, 200, pipelineSnapshot())
    }
    if (request.method === 'POST' && url.pathname === '/api/pipeline/frame') {
      await drain(request)
      pipelineFrame = (pipelineFrame % 40) + 1
      return json(response, 200, pipelineSnapshot())
    }

    if (request.method === 'POST' && url.pathname === '/api/models') {
      await drain(request)
      const model = {
        ...models[0], id: randomUUID(), name: 'Imported model', active: models.length === 0,
        sizeBytes: 8_600_000, createdAt: new Date().toISOString(),
      }
      models.unshift(model)
      return json(response, 201, model)
    }

    const activate = url.pathname.match(/^\/api\/models\/([A-Za-z0-9_-]+)\/activate$/)
    if (request.method === 'POST' && activate) {
      const selected = models.find((model) => model.id === activate[1])
      if (!selected) return json(response, 404, { error: 'model not found' })
      models.forEach((model) => { model.active = model.id === selected.id })
      return json(response, 200, selected)
    }

    const remove = url.pathname.match(/^\/api\/models\/([A-Za-z0-9_-]+)$/)
    if (request.method === 'DELETE' && remove) {
      const index = models.findIndex((model) => model.id === remove[1])
      if (index < 0) return json(response, 404, { error: 'model not found' })
      models.splice(index, 1)
      return json(response, 200, { deleted: remove[1] })
    }

    if (request.method === 'POST' && url.pathname === '/api/infer') {
      await drain(request)
      const model = models.find((candidate) => candidate.active)
      if (!model) return json(response, 400, { error: 'no model is active' })
      return json(response, 200, {
        modelId: model.id,
        task: model.task,
        sourceWidth: 1280,
        sourceHeight: 720,
        detections: [
          { classId: 0, label: 'class_1', confidence: 0.93, x: 126, y: 94, width: 390, height: 430, trackId: null, mask: null },
          { classId: 1, label: 'class_2', confidence: 0.78, x: 710, y: 205, width: 260, height: 285, trackId: null, mask: null },
        ],
        classifications: [], tensors: [],
        timing: {
          decodeMs: 3.2, preprocessMs: 5.6, inferenceMs: 72.4, detectionNmsMs: 2.8,
          maskReconstructionMs: 8.3, trackingMs: 0, queueWaitMs: 0, gradingMs: 0, backendTotalMs: 92.3,
        },
      })
    }
    json(response, 404, { error: 'endpoint not found' })
  } catch (error) {
    json(response, 500, { error: error instanceof Error ? error.message : 'mock server error' })
  }
})

server.listen(8080, '127.0.0.1', () => {
  console.log('Edge Runner mock API: http://127.0.0.1:8080')
})

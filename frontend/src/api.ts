import type { GroundingRequest, InferenceResult, PipelineCapture, PipelineConfig, PipelineSnapshot, RuntimeHealth, StoredModel } from './types'

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // All browser calls are same-origin /api requests. In development Vite
  // proxies them to :8080; in production the C++ service serves React and the
  // API from :8080. The browser never connects to the private Python :8091.
  const response = await fetch(path, init)
  const contentType = response.headers.get('content-type') ?? ''
  const payload = contentType.includes('application/json') ? await response.json() : await response.text()

  if (!response.ok) {
    const message = typeof payload === 'object' && payload && 'error' in payload
      ? String(payload.error)
      : `Request failed (${response.status})`
    throw new ApiError(message, response.status)
  }
  return payload as T
}

export const api = {
  health: () => request<RuntimeHealth>('/api/health'),
  models: () => request<StoredModel[]>('/api/models'),
  rescanModels: () => request<StoredModel[]>('/api/models/rescan', { method: 'POST' }),
  activateModel: (id: string) => request<StoredModel>(`/api/models/${encodeURIComponent(id)}/activate`, { method: 'POST' }),
  infer: async (modelId: string, image: Blob, confidence: number, iou: number, grounding?: GroundingRequest) => {
    // Do not set Content-Type manually for FormData. fetch adds multipart/form-
    // data together with the generated boundary required by the C++ parser.
    const body = new FormData()
    body.append('modelId', modelId)
    body.append('confidence', String(confidence))
    body.append('iou', String(iou))
    if (grounding) {
      body.append('prompt', grounding.prompt)
      body.append('groundingMode', grounding.groundingMode)
      body.append('generationMode', grounding.generationMode)
    }
    body.append('image', image, image instanceof File ? image.name : 'video-frame.jpg')
    const started = performance.now()
    const result = await request<InferenceResult>('/api/infer', { method: 'POST', body })
    result.clientHttpRoundTripMs = performance.now() - started
    return result
  },
  pipeline: () => request<PipelineSnapshot>('/api/pipeline'),
  pipelineConfig: () => request<PipelineConfig>('/api/pipeline/config'),
  pipelineCaptures: () => request<PipelineCapture[]>('/api/pipeline/captures'),
  resetPipeline: () => request<PipelineSnapshot>('/api/pipeline/reset', { method: 'POST' }),
  pipelineFrame: (image: Blob, frameId: number) => {
    // A monotonically increasing frame ID lets the stateful native pipeline
    // reject reordered/duplicate frames before updating any tracks.
    const body = new FormData()
    body.append('frameId', String(frameId))
    body.append('image', image, `frame-${frameId}.jpg`)
    return request<PipelineSnapshot>('/api/pipeline/frame', { method: 'POST', body })
  },
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong'
}

import type { DirectModelDraft, InferenceResult, ModelManifestDraft, PipelineConfig, PipelineSnapshot, RuntimeHealth, StoredModel } from './types'

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
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
  addModel: (manifest: ModelManifestDraft, paramFile: File, binFile: File) => {
    const body = new FormData()
    body.append('manifest', new Blob([JSON.stringify(manifest)], { type: 'application/json' }), 'manifest.json')
    body.append('param', paramFile)
    body.append('bin', binFile)
    return request<StoredModel>('/api/models', { method: 'POST', body })
  },
  addArtifact: (draft: DirectModelDraft, modelFile: File) => {
    const body = new FormData()
    body.append('model', modelFile)
    body.append('name', draft.name)
    body.append('task', draft.task)
    body.append('precision', draft.precision)
    body.append('trustedCheckpoint', String(draft.trustedCheckpoint))
    body.append('inputWidth', String(draft.inputWidth))
    body.append('inputHeight', String(draft.inputHeight))
    return request<StoredModel>('/api/models', { method: 'POST', body })
  },
  activateModel: (id: string) => request<StoredModel>(`/api/models/${encodeURIComponent(id)}/activate`, { method: 'POST' }),
  deleteModel: (id: string) => request<{ deleted: string }>(`/api/models/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  infer: async (modelId: string, image: Blob, confidence: number, iou: number) => {
    const body = new FormData()
    body.append('modelId', modelId)
    body.append('confidence', String(confidence))
    body.append('iou', String(iou))
    body.append('image', image, image instanceof File ? image.name : 'video-frame.jpg')
    const started = performance.now()
    const result = await request<InferenceResult>('/api/infer', { method: 'POST', body })
    result.clientHttpRoundTripMs = performance.now() - started
    return result
  },
  pipeline: () => request<PipelineSnapshot>('/api/pipeline'),
  pipelineConfig: () => request<PipelineConfig>('/api/pipeline/config'),
  resetPipeline: () => request<PipelineSnapshot>('/api/pipeline/reset', { method: 'POST' }),
  pipelineFrame: (image: Blob, frameId: number) => {
    const body = new FormData()
    body.append('frameId', String(frameId))
    body.append('image', image, `frame-${frameId}.jpg`)
    return request<PipelineSnapshot>('/api/pipeline/frame', { method: 'POST', body })
  },
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong'
}

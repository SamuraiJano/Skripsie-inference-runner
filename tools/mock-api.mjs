import { createServer } from 'node:http'
import { randomUUID } from 'node:crypto'

const startedAt = Date.now()
const models = [{
  id: 'demo-yolo-model',
  name: 'Demo YOLO detector',
  task: 'yolo-detect',
  inputWidth: 640,
  inputHeight: 640,
  inputBlob: 'in0',
  outputBlobs: ['out0'],
  threads: 4,
  labels: ['healthy', 'blemish', 'rot'],
  active: true,
  sizeBytes: 12_485_120,
  createdAt: new Date().toISOString(),
}]

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
        status: 'ok', version: '0.1.0-mock', engine: 'Development mock', ncnnAvailable: false,
        architecture: process.arch, logicalCores: 4, recommendedThreads: 4,
        uptimeSeconds: (Date.now() - startedAt) / 1000,
        activeModelId: models.find((model) => model.active)?.id ?? null,
      })
    }
    if (request.method === 'GET' && url.pathname === '/api/models') return json(response, 200, models)

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
          { classId: 0, label: 'healthy', confidence: 0.93, x: 126, y: 94, width: 390, height: 430 },
          { classId: 1, label: 'blemish', confidence: 0.78, x: 710, y: 205, width: 260, height: 285 },
        ],
        classifications: [], tensors: [],
        timing: { decodeMs: 3.2, preprocessMs: 5.6, inferenceMs: 72.4, postprocessMs: 2.8, totalMs: 84.0 },
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


import { Box, Check, Cpu, MoreVertical, Play, Plus, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { ModelPrecision, StoredModel } from '../types'

interface Props {
  models: StoredModel[]
  busyModelId: string | null
  onActivate: (id: string) => void
  onDelete: (id: string) => void
  onAdd: () => void
}

type PrecisionFilter = 'all' | ModelPrecision

function formatBytes(bytes: number) {
  if (bytes <= 0) return '—'
  const mb = bytes / (1024 * 1024)
  return `${mb < 10 ? mb.toFixed(1) : mb.toFixed(0)} MB`
}

export function ModelLibrary({ models, busyModelId, onActivate, onDelete, onAdd }: Props) {
  const [menuId, setMenuId] = useState<string | null>(null)
  const [precision, setPrecision] = useState<PrecisionFilter>('all')
  const visibleModels = useMemo(
    () => precision === 'all' ? models : models.filter((model) => model.precision === precision),
    [models, precision],
  )

  return (
    <section className="panel model-library" aria-labelledby="model-library-title">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Local storage</span>
          <h2 id="model-library-title">Model library</h2>
        </div>
        <div className="library-actions">
          <select
            value={precision}
            onChange={(event) => setPrecision(event.target.value as PrecisionFilter)}
            aria-label="Filter models by precision"
          >
            <option value="all">All precision</option>
            <option value="fp32">FP32</option>
            <option value="fp16">FP16</option>
            <option value="int8">INT8</option>
          </select>
          <button className="icon-button" onClick={onAdd} aria-label="Add a model" data-testid="open-model-form">
            <Plus size={18} />
          </button>
        </div>
      </div>

      {models.length === 0 ? (
        <div className="empty-state compact">
          <Box size={24} />
          <strong>No models stored</strong>
          <span>Add a compatible model file or a verified NCNN package.</span>
          <button className="text-button" onClick={onAdd}>Add first model</button>
        </div>
      ) : visibleModels.length === 0 ? (
        <div className="empty-state compact">
          <Box size={24} />
          <strong>No {precision.toUpperCase()} models</strong>
          <span>Choose another precision filter or import a verified artifact.</span>
        </div>
      ) : (
        <div className="model-list">
          {visibleModels.map((model) => (
            <article className={`model-card ${model.active ? 'active' : ''}`} key={model.id}>
              <div className="model-icon"><Cpu size={18} /></div>
              <div className="model-copy">
                <div className="model-title-row">
                  <strong title={model.name}>{model.name}</strong>
                  {model.active && <span className="active-badge"><Check size={11} /> active</span>}
                </div>
                <span>{model.task.replace('-', ' ')} · {model.precision.toUpperCase()} · {model.inputWidth}×{model.inputHeight}</span>
                <span>{model.format.toUpperCase()} · {model.accelerator} · {formatBytes(model.sizeBytes)}</span>
                <span title={model.backend}>{model.backend} runtime · {model.threads} helper threads</span>
              </div>
              <div className="model-actions">
                {!model.active && (
                  <button
                    className="icon-button subtle"
                    disabled={busyModelId !== null}
                    onClick={() => onActivate(model.id)}
                    aria-label={`Activate ${model.name}`}
                  >
                    {busyModelId === model.id ? <span className="mini-loader" /> : <Play size={15} />}
                  </button>
                )}
                <button
                  className="icon-button subtle"
                  onClick={() => setMenuId(menuId === model.id ? null : model.id)}
                  aria-label={`More options for ${model.name}`}
                  aria-expanded={menuId === model.id}
                >
                  <MoreVertical size={16} />
                </button>
                {menuId === model.id && (
                  <div className="model-menu">
                    <button onClick={() => { setMenuId(null); onDelete(model.id) }}>
                      <Trash2 size={14} /> Delete model
                    </button>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

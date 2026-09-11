import { AlertCircle, Boxes, Check, FolderOpen, Play, RefreshCw, Search, SlidersHorizontal } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { StoredModel } from '../types'

interface Props {
  models: StoredModel[]
  busyModelId: string | null
  scanning: boolean
  onActivate: (id: string) => void
  onRescan: () => void
}

function formatBytes(bytes: number) {
  if (bytes <= 0) return 'Unknown size'
  const gib = bytes / (1024 ** 3)
  if (gib >= 1) return `${gib.toFixed(gib < 10 ? 2 : 1)} GiB`
  const mib = bytes / (1024 ** 2)
  return `${mib < 10 ? mib.toFixed(1) : mib.toFixed(0)} MiB`
}

function taskLabel(task: StoredModel['task']) {
  return {
    'yolo-detect': 'Detection',
    'yolo-segment': 'Segmentation',
    classification: 'Classification',
    'vision-grounding': 'Vision grounding',
    raw: 'Auto task',
  }[task]
}

export function ModelLibrary({ models, busyModelId, scanning, onActivate, onRescan }: Props) {
  const [query, setQuery] = useState('')
  const [format, setFormat] = useState('all')
  const [precision, setPrecision] = useState('all')

  const formats = useMemo(
    () => [...new Set(models.map((model) => model.format))].sort(),
    [models],
  )
  const precisions = useMemo(
    () => [...new Set(models.map((model) => model.precision))].sort(),
    [models],
  )
  const groups = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    const visible = models.filter((model) => {
      const matchesQuery = !normalizedQuery || [model.name, model.family, model.relativePath, model.task, model.format]
        .some((value) => value.toLowerCase().includes(normalizedQuery))
      return matchesQuery
        && (format === 'all' || model.format === format)
        && (precision === 'all' || model.precision === precision)
    })
    const byFamily = new Map<string, StoredModel[]>()
    visible.forEach((model) => {
      const family = model.family || 'Other models'
      byFamily.set(family, [...(byFamily.get(family) ?? []), model])
    })
    return [...byFamily.entries()]
      .sort(([first], [second]) => first.localeCompare(second))
      .map(([family, familyModels]) => ({
        family,
        models: familyModels.sort((first, second) => Number(second.active) - Number(first.active) || first.name.localeCompare(second.name)),
      }))
  }, [format, models, precision, query])
  const visibleCount = groups.reduce((count, group) => count + group.models.length, 0)

  return (
    <section className="panel model-library" aria-labelledby="model-library-title">
      <div className="panel-heading model-library-heading">
        <div>
          <span className="eyebrow">Automatic discovery · models/</span>
          <h2 id="model-library-title">Model library</h2>
        </div>
        <div className="model-library-summary">
          <span><strong>{models.length}</strong> detected</span>
          <span><strong>{models.filter((model) => model.runtimeAvailable && model.compatible).length}</strong> ready</span>
          <button className="button secondary rescan-button" onClick={onRescan} disabled={scanning}>
            <RefreshCw size={14} className={scanning ? 'spin' : ''} />
            {scanning ? 'Scanning…' : 'Rescan folder'}
          </button>
        </div>
      </div>

      <div className="model-toolbar">
        <label className="model-search">
          <Search size={14} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search name, path, task…" aria-label="Search detected models" />
        </label>
        <label className="model-filter">
          <SlidersHorizontal size={13} />
          <select value={format} onChange={(event) => setFormat(event.target.value)} aria-label="Filter models by format">
            <option value="all">All formats</option>
            {formats.map((value) => <option key={value} value={value}>{value.toUpperCase()}</option>)}
          </select>
        </label>
        <label className="model-filter">
          <select value={precision} onChange={(event) => setPrecision(event.target.value)} aria-label="Filter models by weight precision">
            <option value="all">All weights</option>
            {precisions.map((value) => <option key={value} value={value}>{value.toUpperCase()}</option>)}
          </select>
        </label>
        <span className="filter-result">{visibleCount} shown</span>
      </div>

      {models.length === 0 ? (
        <div className="empty-state compact model-empty">
          <FolderOpen size={28} />
          <strong>No compatible models detected</strong>
          <span>Place a supported model file or export package anywhere under <code>models/</code>, then rescan.</span>
          <button className="text-button" onClick={onRescan}>Scan models folder</button>
        </div>
      ) : visibleCount === 0 ? (
        <div className="empty-state compact model-empty">
          <Search size={25} />
          <strong>No models match these filters</strong>
          <span>Clear the search or choose another format and weight precision.</span>
        </div>
      ) : (
        <div className="model-groups">
          {groups.map((group) => (
            <section className="model-family" key={group.family} aria-label={`${group.family} model family`}>
              <div className="model-family-heading">
                <span><FolderOpen size={13} /> {group.family}</span>
                <small>{group.models.length} model{group.models.length === 1 ? '' : 's'}</small>
              </div>
              <div className="model-list">
                {group.models.map((model) => {
                  const ready = model.runtimeAvailable && model.compatible
                  const unavailableReason = model.statusReason || `${model.runtime.name || model.backend} runtime is not installed`
                  return (
                    <article className={`model-card ${model.active ? 'active' : ''} ${ready ? '' : 'unavailable'}`} key={model.id}>
                      <div className="model-icon"><Boxes size={18} /></div>
                      <div className="model-copy">
                        <div className="model-title-row">
                          <strong title={model.name}>{model.name}</strong>
                          {model.active && <span className="active-badge"><Check size={11} /> active</span>}
                          {!ready && <span className="unavailable-badge" title={unavailableReason}><AlertCircle size={10} /> unavailable</span>}
                        </div>
                        <div className="model-badges">
                          <span>{taskLabel(model.task)}</span>
                          <span>{model.format.toUpperCase()}</span>
                          <span className={model.quantization.quantized ? 'quantized' : ''}>{model.precision.toUpperCase()}</span>
                        </div>
                        <dl className="model-details">
                          <div><dt>Weight format</dt><dd title={model.quantization.method}>{model.quantization.label}</dd></div>
                          <div><dt>Runtime</dt><dd>{model.runtime.name} · {model.accelerator}</dd></div>
                          <div><dt>Input</dt><dd>{model.inputWidth}×{model.inputHeight}</dd></div>
                          <div><dt>Size</dt><dd>{formatBytes(model.sizeBytes)}</dd></div>
                        </dl>
                        <span className="model-path" title={model.relativePath}>{model.relativePath}</span>
                      </div>
                      <div className="model-actions">
                        {model.active ? (
                          <span className="active-indicator"><Check size={13} /> In use</span>
                        ) : (
                          <button
                            className="button secondary activate-button"
                            disabled={busyModelId !== null || !ready}
                            onClick={() => onActivate(model.id)}
                            title={ready ? `Activate ${model.name}` : unavailableReason}
                          >
                            {busyModelId === model.id ? <span className="mini-loader" /> : <Play size={13} />}
                            Activate
                          </button>
                        )}
                      </div>
                    </article>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
      )}
    </section>
  )
}

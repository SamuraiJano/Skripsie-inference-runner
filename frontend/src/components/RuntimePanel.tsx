import { Activity, Cpu, Gauge, HardDrive, Server } from 'lucide-react'
import type { RuntimeHealth, StoredModel } from '../types'

interface Props {
  health: RuntimeHealth | null
  activeModel: StoredModel | undefined
}

function duration(seconds: number) {
  if (seconds < 60) return `${Math.floor(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

export function RuntimePanel({ health, activeModel }: Props) {
  return (
    <section className="panel runtime-panel" aria-labelledby="runtime-title">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Device</span>
          <h2 id="runtime-title">Runtime</h2>
        </div>
        <Activity size={18} className={health ? 'success-icon' : 'muted-icon'} />
      </div>

      <div className="runtime-grid">
        <div className="runtime-stat">
          <Cpu size={17} />
          <span>GPU device</span>
          <strong>{health?.gpuDevice ?? 'Unavailable'}</strong>
        </div>
        <div className="runtime-stat">
          <Gauge size={17} />
          <span>Vulkan device</span>
          <strong>{health ? `${health.gpuDeviceIndex} / ${health.gpuCount}` : '—'}</strong>
        </div>
        <div className="runtime-stat">
          <Server size={17} />
          <span>Inference engine</span>
          <strong>{health?.engine ?? 'Offline'}</strong>
        </div>
        <div className="runtime-stat">
          <HardDrive size={17} />
          <span>Uptime</span>
          <strong>{health ? duration(health.uptimeSeconds) : '—'}</strong>
        </div>
      </div>

      <div className="runtime-note">
        <span className="note-dot" />
        <p>
          <strong>Windows GPU profile</strong>
          {activeModel
            ? `${activeModel.format.toUpperCase()} / ${activeModel.precision.toUpperCase()} through ${activeModel.backend} (${activeModel.accelerator}).`
            : `${health?.formats.filter((format) => format.available).length ?? 0} model formats are ready on this laptop.`}
        </p>
      </div>
    </section>
  )
}

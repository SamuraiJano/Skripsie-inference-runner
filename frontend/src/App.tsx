import { AlertTriangle, Box, Code2, Cpu, Layers3, Plus, RefreshCw, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, errorMessage } from './api'
import { InferenceWorkbench } from './components/InferenceWorkbench'
import { ModelForm } from './components/ModelForm'
import { ModelLibrary } from './components/ModelLibrary'
import { RuntimePanel } from './components/RuntimePanel'
import { StatusPill } from './components/StatusPill'
import type { ModelManifestDraft, RuntimeHealth, StoredModel } from './types'

function App() {
  const [health, setHealth] = useState<RuntimeHealth | null>(null)
  const [models, setModels] = useState<StoredModel[]>([])
  const [loading, setLoading] = useState(true)
  const [formOpen, setFormOpen] = useState(false)
  const [importing, setImporting] = useState(false)
  const [busyModelId, setBusyModelId] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const activeModel = useMemo(() => models.find((model) => model.active), [models])

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true)
    try {
      const [nextHealth, nextModels] = await Promise.all([api.health(), api.models()])
      setHealth(nextHealth)
      setModels(nextModels)
    } catch {
      setHealth(null)
    } finally {
      if (!quiet) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(true), 5000)
    return () => window.clearInterval(timer)
  }, [refresh])

  async function addModel(manifest: ModelManifestDraft, param: File, bin: File) {
    setImporting(true)
    try {
      await api.addModel(manifest, param, bin)
      setFormOpen(false)
      await refresh(true)
      setNotice(`${manifest.name} was imported successfully.`)
    } catch (error) {
      setNotice(errorMessage(error))
    } finally {
      setImporting(false)
    }
  }

  async function activateModel(id: string) {
    setBusyModelId(id)
    try {
      await api.activateModel(id)
      await refresh(true)
    } catch (error) {
      setNotice(errorMessage(error))
    } finally {
      setBusyModelId(null)
    }
  }

  async function deleteModel(id: string) {
    try {
      await api.deleteModel(id)
      await refresh(true)
    } catch (error) {
      setNotice(errorMessage(error))
    }
  }

  const status = loading ? 'loading' : health ? 'online' : 'offline'

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Edge Runner home">
          <span className="brand-mark"><Cpu size={19} /></span>
          <span><strong>EDGE</strong> RUNNER</span>
        </a>
        <div className="topbar-actions">
          <StatusPill state={status} />
          <button className="icon-button" onClick={() => void refresh()} aria-label="Refresh runtime"><RefreshCw size={17} /></button>
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <span className="eyebrow hero-eyebrow">Raspberry Pi 4 · local inference</span>
            <h1>Load once.<br /><em>Infer at the edge.</em></h1>
            <p>Run optimized NCNN models on-device with a native C++ hot path and a small React control surface.</p>
          </div>
          <div className="hero-actions">
            <button className="button primary" onClick={() => setFormOpen(true)}><Plus size={17} /> Add model</button>
            <span><span className="live-dot" /> No cloud. No Python runtime.</span>
          </div>
        </section>

        {!health && !loading && (
          <section className="offline-banner">
            <AlertTriangle size={18} />
            <div><strong>The native runtime is offline.</strong><span>Start the C++ server on port 8080, or run the included local mock while developing the UI.</span></div>
          </section>
        )}

        <div className="dashboard-grid">
          <ModelLibrary models={models} busyModelId={busyModelId} onActivate={activateModel} onDelete={deleteModel} onAdd={() => setFormOpen(true)} />
          <InferenceWorkbench activeModel={activeModel} online={Boolean(health)} onError={setNotice} />
          <RuntimePanel health={health} activeModel={activeModel} />
        </div>

        <section className="architecture-strip" aria-label="Application architecture">
          <div><span>01</span><Box size={18} /><p><strong>React UI</strong>Upload and inspect locally</p></div>
          <i />
          <div><span>02</span><Layers3 size={18} /><p><strong>C++ service</strong>Decode, queue, preprocess</p></div>
          <i />
          <div><span>03</span><Cpu size={18} /><p><strong>NCNN / ARM</strong>NEON + INT8-ready compute</p></div>
        </section>
      </main>

      <footer>
        <span>Edge Runner 0.1 · built for aarch64</span>
        <a href="https://github.com/Tencent/ncnn" target="_blank" rel="noreferrer"><Code2 size={14} /> NCNN runtime</a>
      </footer>

      <ModelForm open={formOpen} submitting={importing} onClose={() => setFormOpen(false)} onSubmit={addModel} />

      {notice && (
        <div className="toast" role="alert">
          <span>{notice}</span>
          <button onClick={() => setNotice(null)} aria-label="Dismiss notification"><X size={15} /></button>
        </div>
      )}
    </div>
  )
}

export default App

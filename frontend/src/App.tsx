import { AlertTriangle, Box, Code2, Cpu, Film, FolderOpen, Layers3, RefreshCw, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, errorMessage } from './api'
import { InferenceWorkbench } from './components/InferenceWorkbench'
import { LiveConveyor } from './components/LiveConveyor'
import { ModelLibrary } from './components/ModelLibrary'
import { RuntimePanel } from './components/RuntimePanel'
import { StatusPill } from './components/StatusPill'
import { VideoDetection } from './components/VideoDetection'
import type { RuntimeHealth, StoredModel } from './types'

type AppScreen = 'dashboard' | 'video'

function screenFromHash(): AppScreen {
  return window.location.hash === '#/video' ? 'video' : 'dashboard'
}

function App() {
  const [health, setHealth] = useState<RuntimeHealth | null>(null)
  const [models, setModels] = useState<StoredModel[]>([])
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [busyModelId, setBusyModelId] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [screen, setScreen] = useState<AppScreen>(screenFromHash)

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

  useEffect(() => {
    const syncScreen = () => setScreen(screenFromHash())
    window.addEventListener('hashchange', syncScreen)
    return () => window.removeEventListener('hashchange', syncScreen)
  }, [])

  async function rescanModels() {
    setScanning(true)
    try {
      const nextModels = await api.rescanModels()
      setModels(nextModels)
      setNotice(`${nextModels.length} model${nextModels.length === 1 ? '' : 's'} detected in models/.`)
    } catch (error) {
      setNotice(errorMessage(error))
    } finally {
      setScanning(false)
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

  const status = loading ? 'loading' : health ? 'online' : 'offline'

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-primary">
          <a className="brand" href="#/" aria-label="Inference Runner Windows home">
            <span className="brand-mark"><Cpu size={19} /></span>
            <span><strong>INFERENCE</strong> RUNNER</span>
          </a>
          <nav className="screen-nav" aria-label="Inference screens">
            <a href="#/" className={screen === 'dashboard' ? 'active' : ''} aria-current={screen === 'dashboard' ? 'page' : undefined}><Box size={14} /><span>Dashboard</span></a>
            <a href="#/video" className={screen === 'video' ? 'active' : ''} aria-current={screen === 'video' ? 'page' : undefined}><Film size={14} /><span>Video detection</span></a>
          </nav>
        </div>
        <div className="topbar-actions">
          <StatusPill state={status} />
          <button className="icon-button" onClick={() => void refresh()} aria-label="Refresh runtime"><RefreshCw size={17} /></button>
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <span className="eyebrow hero-eyebrow">Windows · RTX GPU inference</span>
            {screen === 'dashboard' ? (
              <>
                <h1>Load once.<br /><em>Run on the RTX.</em></h1>
                <p>Load compatible model formats through the best runtime available on this laptop, including PyTorch CUDA and native NCNN Vulkan.</p>
              </>
            ) : (
              <>
                <h1>Video in.<br /><em>Detections out.</em></h1>
                <p>Run an object detector, Grounding DINO, or Locate Anything over a local video with sequential, GPU-backed frame inference and live overlays.</p>
              </>
            )}
          </div>
          <div className="hero-actions">
            {screen === 'dashboard' && (
              <div className="model-source-status">
                <FolderOpen size={18} />
                <span><strong>models/</strong>{models.length} automatically detected</span>
              </div>
            )}
            <span><span className="live-dot" /> Local execution. Automatic backend selection.</span>
          </div>
        </section>

        {!health && !loading && (
          <section className="offline-banner">
            <AlertTriangle size={18} />
            <div><strong>The native runtime is offline.</strong><span>Start the C++ server on port 8080, or run the included local mock while developing the UI.</span></div>
          </section>
        )}

        {screen === 'dashboard' ? (
          <>
            <div className="dashboard-grid">
              <ModelLibrary models={models} busyModelId={busyModelId} scanning={scanning} onActivate={activateModel} onRescan={rescanModels} />
              <InferenceWorkbench activeModel={activeModel} online={Boolean(health)} onError={setNotice} />
              <RuntimePanel health={health} activeModel={activeModel} />
              <LiveConveyor online={Boolean(health)} onError={setNotice} />
            </div>

            <section className="architecture-strip" aria-label="Application architecture">
              <div><span>01</span><Box size={18} /><p><strong>Model discovery</strong>Scan and inspect models/</p></div>
              <i />
              <div><span>02</span><Layers3 size={18} /><p><strong>C++ service</strong>Decode, queue, preprocess</p></div>
              <i />
              <div><span>03</span><Cpu size={18} /><p><strong>CUDA / Vulkan</strong>Format-aware GPU inference</p></div>
            </section>
          </>
        ) : (
          <VideoDetection models={models} online={Boolean(health)} onError={setNotice} />
        )}
      </main>

      <footer>
        <span>Inference Runner Windows 1.0 · built for x86-64</span>
        <a href="https://docs.ultralytics.com/modes/predict/" target="_blank" rel="noreferrer"><Code2 size={14} /> Supported formats</a>
      </footer>

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

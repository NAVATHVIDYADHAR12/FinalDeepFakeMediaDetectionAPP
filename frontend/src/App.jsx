import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, NavLink, Route, Routes, useLocation } from 'react-router-dom'

import { api, REMOTE_BACKEND_CONFIGURED } from './api.js'
import Assistant from './components/Assistant.jsx'
import NavBar from './components/NavBar.jsx'
import ScrollTop from './components/ScrollTop.jsx'
import Landing from './pages/Landing.jsx'
import SignUp from './pages/SignUp.jsx'
import { ScrollContext } from './ScrollContext.js'
import LetterReveal from './components/LetterReveal.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Scanner from './pages/Scanner.jsx'
import Report from './pages/Report.jsx'
import History from './pages/History.jsx'
import Models from './pages/Models.jsx'
import Identity from './pages/Identity.jsx'
import System from './pages/System.jsx'
import TextCheck from './pages/TextCheck.jsx'
import ComingSoon from './pages/ComingSoon.jsx'
import LiquidScrollProgress from './components/LiquidScrollProgress.jsx'

/* Navigation mirrors the product design. Entries marked `soon` are honest
   placeholders - the prototype detects image and video, and says so plainly
   rather than showing dead controls that look functional. */
const NAV = [
  {
    section: 'Analyze',
    items: [
      { to: '/dashboard', label: 'Dashboard',        icon: '◲' },
      { to: '/scan',      label: 'Media Scanner',    icon: '⊕' },
      { to: '/scan?type=image', label: 'Image Verification', icon: '▣' },
      { to: '/scan?type=video', label: 'Video Verification', icon: '▶' },
      { to: '/soon/audio',    label: 'Audio Verification', icon: '♪', soon: true },
      { to: '/soon/provenance', label: 'Provenance Checker', icon: '⚿', soon: true },
      { to: '/soon/camera',   label: 'Live Camera',        icon: '◉', soon: true },
      { to: '/soon/mic',      label: 'Live Microphone',    icon: '◎', soon: true },
    ],
  },
  {
    section: 'Text Analysis',
    items: [
      { to: '/text', label: 'Text Analysis', icon: '¶' },
    ],
  },
  {
    section: 'Intelligence',
    items: [
      { to: '/models',        label: 'Model Comparison',   icon: '◫' },
      { to: '/identity',      label: 'Face Recognition',   icon: '☺' },
      { to: '/soon/threats',  label: 'Threat Intelligence', icon: '⚡', soon: true },
      { to: '/soon/analytics', label: 'Analytics',         icon: '◲', soon: true },
    ],
  },
  {
    section: 'History',
    items: [
      { to: '/history',       label: 'Scan History',    icon: '≡' },
      { to: '/soon/evidence', label: 'Evidence Library', icon: '⛁', soon: true },
    ],
  },
  {
    section: 'System',
    items: [
      { to: '/system',        label: 'System Status',   icon: '⚙' },
      { to: '/soon/api',      label: 'API Access',      icon: '⌘', soon: true },
    ],
  },
]

function Sidebar() {
  const location = useLocation()
  const current = location.pathname + location.search

  return (
    <aside className="w-full min-w-[180px] border-r flex flex-col h-full overflow-y-auto"
           style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}>
      {/* The brand mark lives in the floating nav bar now; the sidebar keeps
          only its tagline so the column still has a header. */}
      {/* The brand mark lives here on app pages — not beside the nav bar. */}
      <div className="px-4 pt-5 pb-4">
        <Link to="/" aria-label="OmniGuard AI — home" className="block press mb-3">
          <img src="/logo.png" alt="OmniGuard AI" width="238" height="96"
               className="w-full h-auto object-contain"
               style={{ filter: 'drop-shadow(0 0 16px rgba(0,240,255,.35))' }} />
        </Link>
        <LetterReveal as="div" text="SEE BEYOND THE REAL"
                      className="font-display text-[10px] tracking-[0.18em] headline-animated" />
      </div>

      <nav className="flex-1 px-2.5 pb-4">
        {NAV.map((group) => (
          <div key={group.section} className="mb-4">
            <div className="px-2.5 mb-1.5 text-[10px] font-semibold tracking-wider uppercase"
                 style={{ color: 'var(--ink-muted)' }}>
              {group.section}
            </div>
            <ul className="space-y-0.5">
              {group.items.map((item, i) => {
                const active = current === item.to
                return (
                  <li key={item.to} className="slide-in-left" style={{ '--i': i }}>
                    <NavLink to={item.to}
                             className="flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] transition-all duration-200 hover:translate-x-0.5"
                             style={{
                               background: active ? 'linear-gradient(90deg, var(--brand-2), transparent)' : 'transparent',
                               color: active ? 'var(--ink)' : 'var(--ink-2)',
                             }}>
                      <span className="w-4 text-center shrink-0" aria-hidden="true">{item.icon}</span>
                      <span className="flex-1 truncate">{item.label}</span>
                      {item.soon && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded"
                              style={{ background: 'var(--surface-3)', color: 'var(--ink-muted)' }}>
                          soon
                        </span>
                      )}
                    </NavLink>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </nav>
    </aside>
  )
}

export default function App() {
  const [health, setHealth] = useState(null)
  const location = useLocation()
  const mainRef = useRef(null)
  const resizeRef = useRef(null)
  const [sidebarHidden, setSidebarHidden] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem('omniguard-sidebar-hidden') === 'true'
  })
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    if (typeof window === 'undefined') return 228
    const saved = Number(window.localStorage.getItem('omniguard-sidebar-width'))
    return Number.isFinite(saved) && saved >= 180 && saved <= 360 ? saved : 228
  })
  const [resizingSidebar, setResizingSidebar] = useState(false)

  // GSAP and the scroll hooks need the actual <main> element. A ref alone is
  // null on first render, so consumers would never receive it — the element is
  // mirrored into state via a callback ref to trigger the re-render.
  const [scrollEl, setScrollEl] = useState(null)
  const attachMain = useCallback((node) => {
    mainRef.current = node
    setScrollEl(node)
  }, [])

  useEffect(() => {
    let alive = true
    let timer = null
    // `offline` is distinct from null: null means "not asked yet", offline
    // means the request was made and the backend is not there. The UI needs
    // to tell those apart to show the right message on a static deploy.
    // api.health() already falls back to the browser engine, which reports
    // engine:'browser'. A rejection here means something unexpected broke.
    const poll = async () => {
      let retryMs = 15_000
      try {
        const h = await api.health()
        if (alive) setHealth(h)
      } catch (error) {
        retryMs = 3_000
        if (alive) setHealth(REMOTE_BACKEND_CONFIGURED
          ? {
              engine: 'remote',
              service_unavailable: true,
              models_loaded: false,
              model_count: 0,
              error: error?.message || 'Detection service is not responding',
            }
          : { engine: 'browser', models_loaded: false, model_count: 0 })
      } finally {
        if (alive) timer = setTimeout(poll, retryMs)
      }
    }
    poll()
    return () => { alive = false; clearTimeout(timer) }
  }, [])

  // A new page should start at the top, not wherever the previous one was left.
  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0 })
  }, [location.pathname])

  useEffect(() => {
    window.localStorage.setItem('omniguard-sidebar-hidden', String(sidebarHidden))
    window.localStorage.setItem('omniguard-sidebar-width', String(sidebarWidth))
  }, [sidebarHidden, sidebarWidth])

  useEffect(() => {
    if (!resizingSidebar) return
    const onMove = (event) => {
      const start = resizeRef.current
      if (!start) return
      setSidebarWidth(Math.max(180, Math.min(360, start.width + event.clientX - start.x)))
    }
    const onUp = () => {
      setResizingSidebar(false)
      resizeRef.current = null
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [resizingSidebar])

  const beginSidebarResize = (event) => {
    event.preventDefault()
    resizeRef.current = { x: event.clientX, width: sidebarWidth }
    setResizingSidebar(true)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const isLanding = location.pathname === '/'
  const isAuthPage = ['/signup', '/login'].includes(location.pathname)
  const fullBleed = isLanding || isAuthPage

  return (
    <ScrollContext.Provider value={scrollEl}>
      <div className="flex h-full relative">
        {/* The landing page runs full-bleed; the sidebar belongs to the app. */}
        {!fullBleed && (
          <>
            <div className="sidebar-shell shrink-0 relative h-full"
                 style={{
                   width: sidebarHidden ? 0 : sidebarWidth,
                   opacity: sidebarHidden ? 0 : 1,
                   overflow: 'hidden',
                   transition: resizingSidebar ? 'none' : 'width .28s cubic-bezier(.22,1,.36,1), opacity .2s ease',
                 }}
                 aria-hidden={sidebarHidden}
                 inert={sidebarHidden ? '' : undefined}>
              <div className="h-full" style={{ width: sidebarWidth }}>
                <Sidebar />
              </div>
              {!sidebarHidden && (
                <div role="separator" aria-label="Resize sidebar" aria-orientation="vertical"
                     tabIndex={0} className="sidebar-resize-handle"
                     title="Drag to resize · double-click to reset"
                     onPointerDown={beginSidebarResize}
                     onDoubleClick={() => setSidebarWidth(228)}
                     onKeyDown={(event) => {
                       if (event.key === 'ArrowLeft') setSidebarWidth((width) => Math.max(180, width - 12))
                       if (event.key === 'ArrowRight') setSidebarWidth((width) => Math.min(360, width + 12))
                     }} />
              )}
            </div>
            <button type="button"
                    onClick={() => setSidebarHidden((hidden) => !hidden)}
                    className={`sidebar-toggle press ${sidebarHidden ? 'is-open-button' : ''}`}
                    style={{ left: sidebarHidden ? 12 : sidebarWidth - 15 }}
                    aria-label={sidebarHidden ? 'Open sidebar' : 'Hide sidebar'}
                    aria-expanded={!sidebarHidden}
                    title={sidebarHidden ? 'Open sidebar' : 'Hide sidebar'}>
              <span aria-hidden="true">{sidebarHidden ? '☰' : '‹'}</span>
            </button>
          </>
        )}

        <div className="flex-1 flex flex-col min-w-0 relative">
          <div className="aurora" aria-hidden="true" />
          <div className="grid-field" aria-hidden="true" />

          <main ref={attachMain}
                className={`flex-1 overflow-y-auto relative z-10 ${isLanding ? 'px-6' : 'px-6 pb-6'}`}>

            {isLanding && <LiquidScrollProgress />}

            {/* The nav bar lives inside the scroll container so `sticky` works,
                and outside the keyed wrapper below so it does not remount — and
                therefore does not replay its entry animation — on every route
                change. */}
            <NavBar health={health} isLanding={isLanding} />

            {/* Keying on the path remounts the subtree on navigation, which is
                what replays the entry animation for each page. */}
            <div key={location.pathname} className="page-enter pt-4">
              <Routes location={location}>
                <Route path="/"            element={<Landing />} />
                <Route path="/dashboard"   element={<Dashboard health={health} />} />
                <Route path="/scan"        element={<Scanner health={health} />} />
                <Route path="/report/:id"  element={<Report />} />
                <Route path="/history"     element={<History />} />
                <Route path="/models"      element={<Models />} />
                <Route path="/identity"    element={<Identity />} />
                <Route path="/system"      element={<System />} />
                <Route path="/text"        element={<TextCheck />} />
                <Route path="/signup"      element={<SignUp />} />
                <Route path="/login"       element={<SignUp />} />
                <Route path="/soon/:topic" element={<ComingSoon />} />
                <Route path="*"            element={<ComingSoon notFound />} />
              </Routes>
            </div>
          </main>

          <ScrollTop targetRef={mainRef} />
          <Assistant />
        </div>
      </div>
    </ScrollContext.Provider>
  )
}

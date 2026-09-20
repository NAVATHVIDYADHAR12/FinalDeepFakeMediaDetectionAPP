import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom'

import { useScroller } from '../ScrollContext.js'
import AuthButton from './AuthButton.jsx'
import { prefersReducedMotion } from '../hooks.js'
import { SECTIONS } from '../pages/Landing.jsx'

/**
 * The whole top bar: brand mark on the left, a floating liquid-glass nav pill
 * beside it, engine status on the right — one row, not stacked.
 *
 * The pill hugs its contents rather than stretching, so it reads as a floating
 * control instead of a full-width chrome bar.
 *
 * The bar hides on scroll down and returns the moment you scroll up, which
 * gives long pages their full height back without putting navigation more than
 * one gesture away.
 */

const PRIMARY = [
  { to: '/',          label: 'Home',      icon: '⌂' },
  { to: '/dashboard', label: 'Dashboard', icon: '◲' },
  { to: '/scan',      label: 'Scanner',   icon: '⊕' },
  { to: '/models',    label: 'Models',    icon: '◫' },
  { to: '/identity',  label: 'Face ID',   icon: '☺' },
  { to: '/text',      label: 'Text',      icon: '¶' },
  { to: '/history',   label: 'History',   icon: '≡' },
]

function useHideOnScroll(scroller, { threshold = 120, delta = 6 } = {}) {
  const [hidden, setHidden] = useState(false)
  const lastY = useRef(0)

  useEffect(() => {
    const node = scroller
    if (!node) return

    let ticking = false
    const onScroll = () => {
      if (ticking) return
      ticking = true
      requestAnimationFrame(() => {
        const y = node.scrollTop
        const diff = y - lastY.current

        // Ignore sub-pixel jitter, and never hide near the top of the page.
        if (Math.abs(diff) > delta) {
          setHidden(y > threshold && diff > 0)
          lastY.current = y
        }
        ticking = false
      })
    }

    node.addEventListener('scroll', onScroll, { passive: true })
    return () => node.removeEventListener('scroll', onScroll)
  }, [scroller, threshold, delta])

  return hidden
}

function FeaturesMenu({ onJump, scroller }) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ top: 0, right: 0 })
  const triggerRef = useRef(null)
  const closeTimer = useRef(null)

  // A small close delay keeps the menu usable while the pointer crosses the
  // gap between the trigger and the panel.
  const show = () => { clearTimeout(closeTimer.current); setOpen(true) }
  const hide = () => { closeTimer.current = setTimeout(() => setOpen(false), 180) }

  const place = useCallback(() => {
    const rect = triggerRef.current?.getBoundingClientRect()
    if (rect) {
      setPos({
        top: rect.bottom + 10,
        right: Math.max(12, window.innerWidth - rect.right),
      })
    }
  }, [])

  useLayoutEffect(() => { if (open) place() }, [open, place])

  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && setOpen(false)
    // The bar itself slides away on scroll, so a menu anchored to it must close.
    const onScroll = () => setOpen(false)

    window.addEventListener('keydown', onKey)
    window.addEventListener('resize', place)
    scroller?.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('resize', place)
      scroller?.removeEventListener('scroll', onScroll)
    }
  }, [open, place, scroller])

  useEffect(() => () => clearTimeout(closeTimer.current), [])

  return (
    <li className="shrink-0" onMouseEnter={show} onMouseLeave={hide}>
      <button
        ref={triggerRef}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="true"
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[13px] whitespace-nowrap on-glass transition-all duration-200"
        style={{
          border: `1px solid ${open ? 'rgba(0,240,255,.45)' : 'transparent'}`,
          background: open ? 'rgba(0,240,255,.14)' : 'transparent',
          color: open ? 'var(--ink)' : 'var(--ink-2)',
        }}
      >
        <span aria-hidden="true">✦</span>
        Features
        <span aria-hidden="true"
              style={{
                display: 'inline-block',
                transition: 'transform .2s ease',
                transform: open ? 'rotate(180deg)' : 'none',
                fontSize: '9px',
              }}>▼</span>
      </button>

      {/* Rendered into <body>. The nav pill scrolls horizontally, and
          `overflow-x: auto` clips on BOTH axes — an absolutely positioned menu
          inside it gets cut off entirely. A portal escapes that clipping
          context, so the menu is positioned from the trigger's viewport rect. */}
      {open && createPortal(
        <div
          role="menu"
          onMouseEnter={show}
          onMouseLeave={hide}
          className="fixed w-64 liquid-glass p-2 pop-in"
          style={{ top: pos.top, right: pos.right, zIndex: 60 }}
        >
          {SECTIONS.map((s, i) => (
            <button
              key={s.id}
              role="menuitem"
              onClick={() => { setOpen(false); onJump(s.id) }}
              className="w-full text-left px-3 py-2 rounded-lg text-[13px] flex items-center gap-2.5 fade-in"
              style={{ '--i': i, color: 'var(--ink)' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(0,240,255,.14)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
            >
              <span className="text-[10px]" style={{ color: 'var(--brand)' }} aria-hidden="true">◆</span>
              {s.label}
            </button>
          ))}
        </div>,
        document.body
      )}
    </li>
  )
}

export default function NavBar({ health, isLanding = false }) {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const scroller = useScroller()
  const hidden = useHideOnScroll(scroller)

  const ok = health?.models_loaded

  /** Scroll to a landing section, navigating home first if we are elsewhere. */
  const jumpTo = (id) => {
    const go = () => {
      const el = document.getElementById(id)
      el?.scrollIntoView({
        behavior: prefersReducedMotion() ? 'auto' : 'smooth',
        block: 'start',
      })
    }
    if (isLanding) go()
    else { navigate('/'); setTimeout(go, 420) }
  }

  return (
    <header
      className="sticky top-0 z-40 pt-4 pb-3"
      style={{
        transform: hidden ? 'translateY(calc(-100% - 1rem))' : 'translateY(0)',
        opacity: hidden ? 0 : 1,
        transition: 'transform .38s cubic-bezier(0.22, 1, 0.36, 1), opacity .28s ease',
        willChange: 'transform',
      }}
    >
      {/* Three tracks: brand hugs the left, nav is centred in the flexible
          middle, status hugs the right. The nav therefore stays optically
          centred regardless of how wide the brand or status get. */}
      <div className="grid items-center gap-4"
           style={{ gridTemplateColumns: 'auto minmax(0,1fr) auto' }}>

        {/* The brand mark sits beside the nav on the landing page only. On app
            pages it lives at the top of the sidebar, so it is not duplicated. */}
        {isLanding ? (
          <Link to="/" aria-label="OmniGuard AI — home" className="shrink-0 press">
            <img src="/logo.png" alt="OmniGuard AI" width="238" height="96"
                 className="h-14 w-auto object-contain"
                 style={{ filter: 'drop-shadow(0 0 18px rgba(0,240,255,.35))' }} />
          </Link>
        ) : <span aria-hidden="true" />}

        {/* The pill is inline-flex and centred, so it is only as wide as its
            links — not a full-width bar with dead space on the right. */}
        <div className="flex justify-center min-w-0">
          <ul className="liquid-glass inline-flex items-center gap-1 px-2 py-1.5 rounded-full max-w-full overflow-x-auto">
            {PRIMARY.map((item, i) => {
              const active = pathname === item.to
              return (
                <li key={item.to} className="slide-in-left shrink-0" style={{ '--i': i }}>
                  <NavLink
                    to={item.to}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[13px] whitespace-nowrap transition-all duration-200 on-glass"
                    style={active ? {
                      background: 'linear-gradient(140deg, rgba(0,240,255,.22), rgba(0,240,255,.08))',
                      border: '1px solid rgba(0,240,255,.4)',
                      color: 'var(--ink)',
                      boxShadow: '0 0 18px -6px rgba(0,240,255,.5)',
                    } : {
                      border: '1px solid transparent',
                      color: 'var(--ink-2)',
                    }}
                  >
                    <span aria-hidden="true">{item.icon}</span>
                    {item.label}
                  </NavLink>
                </li>
              )
            })}

            {/* Section anchors only make sense on the page that has them. */}
            {isLanding && <FeaturesMenu onJump={jumpTo} scroller={scroller} />}
          </ul>
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
        <div className="flex items-center gap-2 text-xs px-2.5 py-1.5 rounded-full on-glass"
             style={{
               background: 'rgba(3,7,18,.45)',
               border: '1px solid rgba(226,232,240,.10)',
               backdropFilter: 'blur(10px)',
             }}>
          <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'glow-pulse' : 'pulse-soft'}`}
                style={{ background: ok ? 'var(--good)' : 'var(--warning)' }}
                aria-hidden="true" />
          <span style={{ color: 'var(--ink-2)' }} className="whitespace-nowrap hidden md:inline">
            {health == null ? 'Connecting…'
              : health.service_unavailable ? 'Service waking…'
              : health.engine === 'browser' ? 'Standalone mode'
              : ok ? `${health.model_count} model${health.model_count === 1 ? '' : 's'} online`
                   : 'No models'}
          </span>
        </div>

          <AuthButton />
        </div>
      </div>
    </header>
  )
}

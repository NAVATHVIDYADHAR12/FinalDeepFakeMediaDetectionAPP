import { useEffect, useState } from 'react'

import { useScroller } from '../ScrollContext.js'

/** A GPU-cheap liquid progress rail for the app's inner scroll container. */
export default function LiquidScrollProgress() {
  const scroller = useScroller()
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!scroller) return
    let frame = 0
    const update = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        const distance = scroller.scrollHeight - scroller.clientHeight
        setProgress(distance > 0 ? Math.min(1, scroller.scrollTop / distance) : 0)
      })
    }
    update()
    scroller.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    return () => {
      cancelAnimationFrame(frame)
      scroller.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
    }
  }, [scroller])

  return (
    <div className={`liquid-progress ${progress > 0.01 ? 'is-visible' : ''}`}
         aria-hidden="true">
      <div className="liquid-progress__fill" style={{ transform: `scaleX(${progress})` }}>
        <span className="liquid-progress__bead" />
      </div>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'

/** True once the user has asked the OS to reduce motion. */
export function prefersReducedMotion() {
  return typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

/**
 * Reveal-on-scroll.
 *
 * Returns a ref and a flag that flips once the element scrolls into view, so
 * panels below the fold animate when you actually reach them instead of
 * finishing their animation unseen while the page loads.
 *
 * Unobserves after firing — these animations play once, not on every scroll past.
 */
export function useInView({ threshold = 0.12, rootMargin = '0px 0px -40px 0px' } = {}) {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const node = ref.current
    if (!node) return

    // No IntersectionObserver, or motion is unwanted: show immediately.
    if (typeof IntersectionObserver === 'undefined' || prefersReducedMotion()) {
      setInView(true)
      return
    }

    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setInView(true)
        observer.unobserve(node)
      }
    }, { threshold, rootMargin })

    observer.observe(node)
    return () => observer.disconnect()
  }, [threshold, rootMargin])

  return [ref, inView]
}

/** Tracks every viewport entry/exit so animations can deliberately replay. */
export function useRepeatInView({ threshold = 0.28, rootMargin = '0px 0px -8% 0px' } = {}) {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    if (typeof IntersectionObserver === 'undefined' || prefersReducedMotion()) {
      setInView(true)
      return
    }
    const observer = new IntersectionObserver(
      ([entry]) => setInView(entry.isIntersecting),
      { threshold, rootMargin },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [threshold, rootMargin])

  return [ref, inView]
}

/**
 * Counts a number up to its target.
 *
 * Uses an ease-out curve so it decelerates into the final value rather than
 * stopping dead. Skipped entirely under reduced motion, and re-runs whenever
 * the target changes so live-updating stats stay correct.
 */
export function useCountUp(target, { duration = 900, start = false, resetOnStop = false } = {}) {
  const [value, setValue] = useState(0)
  const frameRef = useRef(null)

  useEffect(() => {
    const end = Number(target) || 0

    if (!start) {
      if (resetOnStop) setValue(0)
      return
    }
    if (prefersReducedMotion() || end === 0) {
      setValue(end)
      return
    }

    const t0 = performance.now()
    const tick = (now) => {
      const progress = Math.min(1, (now - t0) / duration)
      const eased = 1 - Math.pow(1 - progress, 3)
      setValue(end * eased)
      if (progress < 1) frameRef.current = requestAnimationFrame(tick)
      else setValue(end)
    }

    frameRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frameRef.current)
  }, [target, duration, start, resetOnStop])

  return value
}

/** Tracks how far the given scroll container has been scrolled. */
export function useScrollPosition(elementRef) {
  const [scrolled, setScrolled] = useState(0)

  useEffect(() => {
    const node = elementRef?.current
    if (!node) return

    let ticking = false
    const onScroll = () => {
      if (ticking) return
      ticking = true
      requestAnimationFrame(() => {
        setScrolled(node.scrollTop)
        ticking = false
      })
    }

    node.addEventListener('scroll', onScroll, { passive: true })
    return () => node.removeEventListener('scroll', onScroll)
  }, [elementRef])

  return scrolled
}

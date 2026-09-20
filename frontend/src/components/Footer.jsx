import { Link } from 'react-router-dom'
import Reveal from './Reveal.jsx'

const COLUMNS = [
  { heading: 'Verify', links: [
    { label: 'Media scanner', to: '/scan' }, { label: 'Video verification', to: '/scan?type=video' },
    { label: 'Text & plagiarism', to: '/text' }, { label: 'Face recognition', to: '/identity' },
  ]},
  { heading: 'Inspect', links: [
    { label: 'Dashboard', to: '/dashboard' }, { label: 'Model comparison', to: '/models' },
    { label: 'Scan history', to: '/history' }, { label: 'System status', to: '/system' },
  ]},
  { heading: 'Resources', links: [
    { label: 'Technical documentation', href: '/Documentation.html' },
    { label: 'Plain-text specification', href: '/Documentation.txt' },
    { label: 'API reference', href: '/docs' }, { label: 'Known limitations', to: '/#honesty' },
  ]},
]

function FooterLink({ item }) {
  const className = 'footer-link group inline-flex items-center gap-2 text-[13px]'
  const content = <><span className="footer-link__arrow" aria-hidden="true">↗</span>{item.label}</>
  return item.href
    ? <a href={item.href} target="_blank" rel="noopener" className={className}>{content}</a>
    : <Link to={item.to} className={className}>{content}</Link>
}

export default function Footer() {
  return (
    <footer className="future-footer relative mt-10 border-t" style={{ borderColor: 'var(--border)' }}>
      <div className="accent-rule" aria-hidden="true" />
      <div className="future-footer__orb future-footer__orb--one" aria-hidden="true" />
      <div className="future-footer__orb future-footer__orb--two" aria-hidden="true" />

      <div className="max-w-6xl mx-auto px-6 pt-16 pb-8 relative">
        <Reveal from="scale" className="future-footer__cta panel p-7 sm:p-9 mb-14 overflow-hidden">
          <div className="future-footer__scan" aria-hidden="true" />
          <div className="relative flex flex-col md:flex-row md:items-center justify-between gap-7">
            <div>
              <div className="text-[10px] tracking-[.24em] font-display mb-3" style={{ color: 'var(--good)' }}>EVIDENCE, NOT GUESSWORK</div>
              <h2 className="font-display font-bold headline text-2xl sm:text-3xl mb-2">Question the pixels. Keep the proof.</h2>
              <p className="text-[13px] max-w-xl leading-relaxed" style={{ color: 'var(--ink-2)' }}>
                Three neural networks, forensic signals and explainable heatmaps converge into one reviewable report.
              </p>
            </div>
            <Link to="/scan" className="spark-sweep press shrink-0 rounded-full px-6 py-3 text-sm font-semibold"
                  style={{ background: 'linear-gradient(135deg,var(--brand),var(--brand-2))', color: 'var(--on-accent)' }}>
              <span>Start verification →</span>
            </Link>
          </div>
        </Reveal>

        <Reveal from="up" stagger={0.1} className="grid gap-10 md:grid-cols-[1.45fr_repeat(3,1fr)]">
          <div>
            <img src="/logo.png" alt="OmniGuard AI" width="238" height="96" className="h-11 w-auto object-contain mb-5"
                 style={{ filter: 'drop-shadow(0 0 18px rgba(0,240,255,.34))' }} />
            <p className="text-[13px] leading-relaxed max-w-xs" style={{ color: 'var(--ink-2)' }}>
              Media intelligence designed to show its working. Private by default, calibrated for uncertainty, built for human review.
            </p>
            <div className="flex flex-wrap gap-2 mt-5">
              <span className="footer-chip"><i className="glow-pulse" />Models online</span>
              <span className="footer-chip">CPU ready</span><span className="footer-chip">Local-first</span>
            </div>
          </div>
          {COLUMNS.map((column) => (
            <nav key={column.heading} aria-label={column.heading}>
              <h3 className="font-display text-[10px] tracking-[.2em] mb-5" style={{ color: 'var(--brand)' }}>{column.heading.toUpperCase()}</h3>
              <ul className="space-y-3">{column.links.map((item) => <li key={item.label}><FooterLink item={item} /></li>)}</ul>
            </nav>
          ))}
        </Reveal>

        <div className="future-footer__wordmark" aria-hidden="true">OMNIGUARD</div>
        <div className="pt-6 border-t flex flex-wrap gap-4 items-center justify-between text-[11px] relative"
             style={{ borderColor: 'var(--border)', color: 'var(--ink-muted)' }}>
          <p>© {new Date().getFullYear()} OmniGuard AI · PyTorch · ONNX Runtime · OpenCV · React</p>
          <p className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--warning)' }} />A verdict is evidence, never proof.</p>
        </div>
      </div>
    </footer>
  )
}

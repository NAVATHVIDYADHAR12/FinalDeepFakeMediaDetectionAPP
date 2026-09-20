import { verdictMeta } from '../api.js'
import { Sparkline } from './charts.jsx'
import { useCountUp, useInView } from '../hooks.js'

/* Shared presentational primitives. */

/**
 * @param index  position in a group; staggers the reveal animation so panels
 *               cascade in rather than all popping at once.
 */
export function Panel({ title, action, children, className = '', bodyClass = '', index = 0 }) {
  const [ref, inView] = useInView()

  return (
    <section ref={ref}
             className={`panel lift ${inView ? 'reveal' : 'opacity-0'} ${className}`}
             style={{ '--i': index }}>
      {(title || action) && (
        <header className="flex items-center justify-between px-5 pt-4 pb-3">
          <h2 className="text-[15px] font-semibold">{title}</h2>
          {action}
        </header>
      )}
      <div className={`px-5 pb-5 ${bodyClass}`}>{children}</div>
    </section>
  )
}

/** Verdict pill. Color is always accompanied by an icon and the label text,
 *  so the state survives colorblindness, print and forced-colors mode. */
export function VerdictBadge({ verdict, size = 'sm' }) {
  const meta = verdictMeta(verdict)
  const pad = size === 'lg' ? 'px-3 py-1.5 text-sm' : 'px-2.5 py-1 text-xs'
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full font-medium ${pad}`}
          style={{
            color: meta.color,
            background: `color-mix(in srgb, ${meta.color} 14%, transparent)`,
            border: `1px solid color-mix(in srgb, ${meta.color} 32%, transparent)`,
          }}>
      <span aria-hidden="true" className="font-bold">{meta.icon}</span>
      {meta.label}
    </span>
  )
}

/**
 * @param value  a number counts up when the tile scrolls into view; a string
 *               renders verbatim, so callers that pre-format still work.
 */
export function StatTile({ label, value, sub, color = 'var(--brand)', icon, trend, index = 0 }) {
  const [ref, inView] = useInView()
  const isNumber = typeof value === 'number'
  const counted = useCountUp(isNumber ? value : 0, { start: inView && isNumber })

  return (
    <div ref={ref}
         className={`panel p-4 flex flex-col gap-3 lift ${inView ? 'reveal' : 'opacity-0'}`}
         style={{ '--i': index }}>
      <div className="flex items-center gap-2.5">
        <span className="w-8 h-8 rounded-lg flex items-center justify-center text-sm"
              style={{ background: `color-mix(in srgb, ${color} 18%, transparent)`, color }}
              aria-hidden="true">
          {icon}
        </span>
        <span className="text-[13px]" style={{ color: 'var(--ink-2)' }}>{label}</span>
      </div>

      <div className="flex items-end justify-between gap-2">
        <div>
          <div className="text-[28px] leading-none figure">
            {isNumber ? Math.round(counted).toLocaleString() : value}
          </div>
          {sub && <div className="text-xs mt-1.5" style={{ color }}>{sub}</div>}
        </div>
        {trend?.length > 1 && <Sparkline values={trend} color={color} />}
      </div>
    </div>
  )
}

export function Spinner({ label }) {
  return (
    <div className="flex items-center justify-center gap-3 py-10" style={{ color: 'var(--ink-muted)' }}>
      <span className="spin w-4 h-4 rounded-full border-2 border-current border-t-transparent" />
      {label && <span className="text-sm">{label}</span>}
    </div>
  )
}

export function EmptyState({ icon = '◍', title, body, action }) {
  return (
    <div className="text-center py-12 px-6">
      <div className="text-3xl mb-3" aria-hidden="true" style={{ color: 'var(--ink-muted)' }}>{icon}</div>
      <p className="font-medium mb-1">{title}</p>
      {body && <p className="text-sm max-w-md mx-auto" style={{ color: 'var(--ink-muted)' }}>{body}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

/** Blocking notice shown when the backend has no trained models yet. It states
 *  the exact next step rather than just reporting failure. */
export function ModelsMissing() {
  return (
    <div className="panel p-6 rise" style={{ borderColor: 'color-mix(in srgb, var(--warning) 40%, transparent)' }}>
      <div className="flex items-start gap-3">
        <span className="text-lg" style={{ color: 'var(--warning)' }} aria-hidden="true">⚠</span>
        <div>
          <h3 className="font-semibold mb-1">No trained models loaded</h3>
          <p className="text-sm mb-3" style={{ color: 'var(--ink-2)' }}>
            Detection is unavailable until the deepfake classifiers are in place.
          </p>
          <ol className="text-sm space-y-1.5 list-decimal ml-4" style={{ color: 'var(--ink-2)' }}>
            <li>Open <code className="px-1 rounded" style={{ background: 'var(--surface-2)' }}>notebooks/OmniGuard_Training.ipynb</code> in Google Colab</li>
            <li>Set <strong>Runtime → Change runtime type → T4 GPU</strong>, then <strong>Run all</strong></li>
            <li>Unzip the downloaded <code className="px-1 rounded" style={{ background: 'var(--surface-2)' }}>omniguard_models.zip</code> into <code className="px-1 rounded" style={{ background: 'var(--surface-2)' }}>backend/models/</code></li>
            <li>Restart the server</li>
          </ol>
        </div>
      </div>
    </div>
  )
}

/** A hosted backend can briefly be unreachable while a free Render instance
 * wakes. This is never presented as browser-only detection: hosted scans must
 * either use the trained service or fail honestly. App.jsx retries every 15s. */
export function ServiceUnavailableNotice({ detail }) {
  return (
    <div className="panel p-6 rise"
         style={{ borderColor: 'color-mix(in srgb, var(--warning) 40%, transparent)' }}>
      <div className="flex items-start gap-3">
        <span className="spin mt-0.5 w-4 h-4 rounded-full border-2 border-current border-t-transparent"
              style={{ color: 'var(--warning)' }} aria-hidden="true" />
        <div>
          <h3 className="font-semibold mb-1">Detection service is waking up</h3>
          <p className="text-sm" style={{ color: 'var(--ink-2)' }}>
            The hosted model service can take about 50 seconds to resume after inactivity.
            OmniGuard is reconnecting automatically; uploads stay disabled until the trained
            classifiers are online, so no scan can silently fall back to a model-free verdict.
          </p>
          {detail && <p className="text-[11px] mt-2" style={{ color: 'var(--ink-muted)' }}>{detail}</p>}
        </div>
      </div>
    </div>
  )
}

/**
 * Shown when the Python service is not reachable and the app is running on its
 * own. It is not an error state: metadata, compression and provenance checks
 * all still run in the browser. Only the neural verdict is missing, and the
 * notice says exactly that rather than implying nothing works.
 */
export function StandaloneNotice() {
  return (
    <div className="panel p-6 rise"
         style={{ borderColor: 'color-mix(in srgb, var(--warning) 38%, transparent)' }}>
      <div className="flex items-start gap-3">
        <span className="text-lg" style={{ color: 'var(--warning)' }} aria-hidden="true">◐</span>
        <div>
          <h3 className="font-semibold mb-1">Standalone mode — analysis runs in your browser</h3>
          <p className="text-sm mb-3" style={{ color: 'var(--ink-2)' }}>
            No detection service is connected, so the app is using its built-in engine.
            These checks are real and computed from the file itself:
          </p>
          <ul className="text-sm space-y-1.5 ml-1 mb-3" style={{ color: 'var(--ink-2)' }}>
            <li>✓ EXIF metadata — camera, editing software, GPS, stripping</li>
            <li>✓ Error Level Analysis — compression inconsistency</li>
            <li>✓ C2PA provenance — content credential detection</li>
            <li>✓ Scan history, stored locally in your browser</li>
          </ul>
          <p className="text-sm mb-2" style={{ color: 'var(--ink-2)' }}>
            What is <strong>not</strong> available is the neural verdict: that needs the
            trained CNN ensemble, which is far too large to ship to a browser. Reports are
            marked <em>Not verified</em> rather than being given a guessed score.
          </p>
          <p className="text-[12px]" style={{ color: 'var(--ink-muted)' }}>
            For full detection — deepfake scoring, heatmaps, face recognition and video —
            clone the repository and run <code className="px-1 rounded"
            style={{ background: 'var(--surface-2)' }}>START.bat</code>.
          </p>
        </div>
      </div>
    </div>
  )
}

export function Findings({ items }) {
  const tone = { high: 'var(--critical)', medium: 'var(--warning)', info: 'var(--accent)' }
  const glyph = { high: '✕', medium: '!', info: 'i' }
  return (
    <ul className="space-y-2.5">
      {items.map((f, i) => (
        <li key={i} className="flex items-start gap-2.5 text-[13px] fade-in" style={{ '--i': i }}>
          <span className="mt-[3px] w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold shrink-0"
                style={{
                  color: tone[f.severity],
                  background: `color-mix(in srgb, ${tone[f.severity]} 18%, transparent)`,
                }}
                aria-hidden="true">
            {glyph[f.severity]}
          </span>
          <span style={{ color: 'var(--ink-2)' }}>{f.text}</span>
        </li>
      ))}
    </ul>
  )
}

export function Timeline({ steps }) {
  return (
    <ol className="flex gap-1 overflow-x-auto pb-2">
      {steps.map((s, i) => (
        <li key={i} className="flex flex-col items-center min-w-[104px] text-center fade-in"
            style={{ '--i': i }}>
          <div className="flex items-center w-full">
            <span className="h-px flex-1" style={{ background: i === 0 ? 'transparent' : 'var(--border-bright)' }} />
            <span className="w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-bold shrink-0"
                  style={{ background: 'var(--surface-3)', color: 'var(--brand)', border: '1px solid var(--border-bright)' }}>
              {i + 1}
            </span>
            <span className="h-px flex-1" style={{ background: i === steps.length - 1 ? 'transparent' : 'var(--border-bright)' }} />
          </div>
          <div className="text-[11px] mt-2 leading-tight" style={{ color: 'var(--ink-2)' }}>{s.stage}</div>
          <div className="text-[10px] tnum mt-0.5" style={{ color: 'var(--ink-muted)' }}>
            {s.elapsed_ms.toFixed(0)} ms
          </div>
        </li>
      ))}
    </ol>
  )
}

export function KeyValue({ rows }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-2 text-[13px]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt style={{ color: 'var(--ink-muted)' }}>{k}</dt>
          <dd className="tnum text-right sm:text-left">{v ?? '—'}</dd>
        </div>
      ))}
    </dl>
  )
}

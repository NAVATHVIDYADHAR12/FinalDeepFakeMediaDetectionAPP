import { useEffect, useState } from 'react'

import { api } from '../api.js'
import { EmptyState, KeyValue, Panel, Spinner } from '../components/ui.jsx'

export default function System() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => { api.systemInfo().then(setInfo).catch((e) => setError(e.message)) }, [])

  if (error) return <EmptyState icon="⚠" title="Cannot reach the backend" body={error} />
  if (!info) return <Spinner label="Loading system status…" />

  const d = info.detector
  const ai = info.ai_detector ?? { ready: false, model_count: 0, models: [] }

  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Detection Engine">
          <KeyValue rows={[
            ['Engine', info.engine === 'browser' ? 'browser (standalone)' : 'local Python service'],
            ['Face-forgery classifiers', d.ready ? `${d.model_count} model(s)` : 'none'],
            ['AI-image classifiers', ai.ready ? `${ai.model_count} model(s)` : 'none'],
            ['Face analyzer', info.face_analyzer_ready ? 'ready (YuNet + SFace)' : 'unavailable'],
            ['Face model data', d.trained_on ?? '—'],
            ['AI-image model data', ai.trained_on ?? '—'],
          ]} />
          {d.ready && (
            <ul className="mt-4 pt-3 border-t space-y-1.5 text-[13px]" style={{ borderColor: 'var(--border)' }}>
              {d.models.map((m) => (
                <li key={m.arch} className="flex justify-between">
                  <span>{m.name}</span>
                  <span className="tnum" style={{ color: 'var(--ink-muted)' }}>
                    {m.metrics?.accuracy != null ? `${(m.metrics.accuracy * 100).toFixed(1)}% acc` : 'no metrics'}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {ai.ready && (
            <ul className="mt-4 pt-3 border-t space-y-1.5 text-[13px]" style={{ borderColor: 'var(--border)' }}>
              {ai.models.map((m) => (
                <li key={`ai-${m.arch}`} className="flex justify-between">
                  <span>{m.name} <span style={{ color: 'var(--ink-muted)' }}>(AI image)</span></span>
                  <span className="tnum" style={{ color: 'var(--ink-muted)' }}>
                    {m.metrics?.balanced_accuracy != null
                      ? `${(m.metrics.balanced_accuracy * 100).toFixed(1)}% balanced acc`
                      : m.metrics?.accuracy != null ? `${(m.metrics.accuracy * 100).toFixed(1)}% acc` : 'no metrics'}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Thresholds & Limits">
          <KeyValue rows={[
            ['Suspicious at', `fake probability ≥ ${info.thresholds.suspicious}`],
            ['Fake at', `fake probability ≥ ${info.thresholds.fake}`],
            ['Max upload', `${info.limits.max_upload_mb} MB`],
            ['Video frames sampled', info.limits.video_max_frames],
          ]} />
        </Panel>
      </div>

      <Panel title="Supported Formats">
        <div className="grid sm:grid-cols-2 gap-5 text-[13px]">
          <div>
            <div className="mb-1.5" style={{ color: 'var(--ink-muted)' }}>Images</div>
            <div className="tnum">{info.supported.image.join('  ')}</div>
          </div>
          <div>
            <div className="mb-1.5" style={{ color: 'var(--ink-muted)' }}>Videos</div>
            <div className="tnum">{info.supported.video.join('  ')}</div>
          </div>
        </div>
      </Panel>

      <Panel title="Documentation">
        <p className="text-[13px] mb-4" style={{ color: 'var(--ink-muted)' }}>
          Full technical report — architecture, model training, engineering decisions,
          security, testing and limitations. Available as a styled page, plain text,
          or markdown.
        </p>
        <div className="flex flex-wrap gap-3">
          <a href="/Documentation.html" target="_blank" rel="noopener"
             className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-[13px] press"
             style={{
               background: 'linear-gradient(135deg, var(--brand), var(--brand-2))',
               color: 'var(--on-accent)', fontWeight: 600,
             }}>
            Open documentation
          </a>
          <a href="/Documentation.txt" target="_blank" rel="noopener"
             className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-[13px] press"
             style={{
               background: 'var(--surface-2)',
               border: '1px solid var(--border-bright)', color: 'var(--ink-2)',
             }}>
            Plain text
          </a>
          <a href="/Documentation.md" target="_blank" rel="noopener"
             className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-[13px] press"
             style={{
               background: 'var(--surface-2)',
               border: '1px solid var(--border-bright)', color: 'var(--ink-2)',
             }}>
            Markdown
          </a>
          <a href="/docs" target="_blank" rel="noopener"
             className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-[13px] press"
             style={{
               background: 'var(--surface-2)',
               border: '1px solid var(--border-bright)', color: 'var(--ink-2)',
             }}>
            Interactive API docs
          </a>
        </div>
      </Panel>

      <Panel title="Scope & Limitations">
        <ul className="space-y-2.5 text-[13px]" style={{ color: 'var(--ink-2)' }}>
          {[
            'Separate model banks analyze face manipulation and full-frame AI generation; their evidence is kept distinct in each report.',
            'The current AI-image model was trained on GPT Image 2 and Nano Banana 2 examples. Other generators and post-processing can reduce accuracy.',
            'C2PA is checked for presence only. Cryptographic validation against a trust list is not performed.',
            'Error Level Analysis is weak on PNGs and on heavily re-compressed images.',
            'A verdict is evidence, not proof. Treat it as one input to a human decision.',
          ].map((line) => (
            <li key={line} className="flex gap-2.5">
              <span style={{ color: 'var(--warning)' }} aria-hidden="true">•</span>{line}
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  )
}

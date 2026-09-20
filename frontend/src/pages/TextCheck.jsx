import { useState } from 'react'

import { api } from '../api.js'
import { Meter } from '../components/charts.jsx'
import { Panel } from '../components/ui.jsx'

/**
 * Plagiarism and AI-generated-text analysis.
 *
 * The two results are presented differently on purpose. Plagiarism overlap is
 * an exact measurement against a supplied reference. AI-text classification
 * combines a trained model with explainable style signals, but remains
 * probabilistic and therefore carries an explicit caveat.
 */

const VERDICT_COLOUR = {
  HIGH: 'var(--critical)',
  MODERATE: 'var(--warning)',
  LOW: 'var(--brand)',
  MINIMAL: 'var(--good)',
  'STRONG INDICATORS': 'var(--critical)',
  'SOME INDICATORS': 'var(--warning)',
  'FEW INDICATORS': 'var(--brand)',
  'MINIMAL INDICATORS': 'var(--good)',
  'STRONG CONCERNS': 'var(--critical)',
  'SOME CONCERNS': 'var(--warning)',
  'FEW CONCERNS': 'var(--brand)',
  'MINIMAL CONCERNS': 'var(--good)',
}

const colourFor = (v) => VERDICT_COLOUR[v] ?? 'var(--ink-muted)'

const MARK_STYLE = {
  plagiarism: {
    colour: 'var(--warning)',
    fill: 'color-mix(in srgb, var(--warning) 26%, transparent)',
  },
  ai: {
    colour: 'var(--brand)',
    fill: 'color-mix(in srgb, var(--brand) 22%, transparent)',
  },
  news: {
    colour: 'var(--signal)',
    fill: 'color-mix(in srgb, var(--signal) 24%, transparent)',
  },
  both: {
    colour: 'var(--critical)',
    fill: 'color-mix(in srgb, var(--critical) 26%, transparent)',
  },
}

// One phrasing per kind, so the hover text reads the same way wherever a mark
// came from.
const MARK_TITLE = {
  plagiarism: (m) => `Plagiarism: ${m.words} consecutive words found in the reference`,
  ai: (m) => `AI indicators (${m.strength}%): ${m.reasons?.join('; ')}`,
  news: (m) => `Credibility signals (${m.strength}%): ${m.reasons?.join('; ')}`,
}

/**
 * Rebuild the submitted text with suspected regions marked.
 *
 * Takes both kinds of region at once and resolves overlaps, because a passage
 * can be flagged by both checks and rendering two overlapping <mark> elements
 * would produce invalid nesting. The text is sliced by character offsets from
 * the backend, so it appears exactly as typed — capitalisation, punctuation
 * and line breaks intact.
 */
function highlight(text, plagiarismSpans = [], aiSpans = [], newsSpans = []) {
  const marks = [
    ...plagiarismSpans.map((s) => ({ ...s, kind: 'plagiarism' })),
    ...aiSpans.map((s) => ({ ...s, kind: 'ai' })),
    ...newsSpans.map((s) => ({ ...s, kind: 'news' })),
  ]
  if (!marks.length) return text

  // Sweep the boundaries so every character gets exactly one classification.
  const edges = new Set([0, text.length])
  marks.forEach((m) => { edges.add(m.start); edges.add(m.end) })
  const points = [...edges].sort((a, b) => a - b)

  const parts = []
  for (let i = 0; i < points.length - 1; i++) {
    const from = points[i]
    const to = points[i + 1]
    if (to <= from) continue

    const covering = marks.filter((m) => m.start <= from && m.end >= to)
    const slice = text.slice(from, to)

    if (!covering.length) {
      parts.push(<span key={from}>{slice}</span>)
      continue
    }

    const kinds = new Set(covering.map((m) => m.kind))
    const kind = kinds.size > 1 ? 'both' : [...kinds][0]
    const style = MARK_STYLE[kind]

    const title = covering.map((m) => MARK_TITLE[m.kind](m)).join('\n')

    parts.push(
      <mark key={from} title={title}
            style={{
              background: style.fill,
              color: 'var(--ink)',
              borderBottom: `2px solid ${style.colour}`,
              borderRadius: 3,
              padding: '1px 2px',
            }}>
        {slice}
      </mark>
    )
  }
  return parts
}

function Legend({ showPlagiarism, showAi, showNews }) {
  // "Overlapping" rather than "Both" now that three kinds can coincide.
  const active = [showPlagiarism, showAi, showNews].filter(Boolean).length
  const items = [
    showPlagiarism && ['plagiarism', 'Matches the reference'],
    showAi && ['ai', 'AI indicators'],
    showNews && ['news', 'Credibility signals'],
    active > 1 && ['both', 'Overlapping'],
  ].filter(Boolean)

  return (
    <div className="flex flex-wrap gap-3 mb-2.5">
      {items.map(([kind, label]) => (
        <span key={kind} className="flex items-center gap-1.5 text-[11px]"
              style={{ color: 'var(--ink-muted)' }}>
          <span className="w-3.5 h-3.5 rounded-sm"
                style={{
                  background: MARK_STYLE[kind].fill,
                  borderBottom: `2px solid ${MARK_STYLE[kind].colour}`,
                }} />
          {label}
        </span>
      ))}
    </div>
  )
}

function Toggle({ checked, onChange, title, body, accent }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="text-left p-4 rounded-xl press transition-all w-full"
      style={{
        background: checked ? `color-mix(in srgb, ${accent} 12%, transparent)` : 'var(--surface-2)',
        border: `1px solid ${checked ? `color-mix(in srgb, ${accent} 45%, transparent)` : 'var(--border)'}`,
      }}
    >
      <div className="flex items-center gap-2.5 mb-1.5">
        <span className="w-4 h-4 rounded flex items-center justify-center text-[10px] font-bold shrink-0"
              style={{
                background: checked ? accent : 'transparent',
                border: `1px solid ${checked ? accent : 'var(--border-bright)'}`,
                color: 'var(--on-accent)',
              }}
              aria-hidden="true">
          {checked ? '✓' : ''}
        </span>
        <span className="font-semibold text-[13px]" style={{ color: checked ? 'var(--ink)' : 'var(--ink-2)' }}>
          {title}
        </span>
      </div>
      <p className="text-[11.5px] leading-relaxed ml-[26px]" style={{ color: 'var(--ink-muted)' }}>
        {body}
      </p>
    </button>
  )
}

function ScoreBlock({ label, percent, verdict, caption }) {
  const colour = colourFor(verdict)
  return (
    <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
      <div>
        <div className="text-xs mb-1.5" style={{ color: 'var(--ink-muted)' }}>{label}</div>
        <div className="text-5xl leading-none figure" style={{ color: colour }}>{percent}%</div>
      </div>
      <div className="pb-1">
        <div className="text-sm font-semibold" style={{ color: colour }}>{verdict}</div>
        <div className="text-[11.5px] mt-0.5" style={{ color: 'var(--ink-muted)' }}>{caption}</div>
      </div>
    </div>
  )
}

export default function TextCheck() {
  const [text, setText] = useState('')
  const [reference, setReference] = useState('')
  const [wantPlagiarism, setWantPlagiarism] = useState(true)
  const [wantAi, setWantAi] = useState(true)
  const [wantNews, setWantNews] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  const canRun = words > 0 && (wantPlagiarism || wantAi || wantNews) && !busy

  const run = async () => {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const body = new FormData()
      body.append('text', text)
      body.append('reference', reference)
      body.append('check_plagiarism', wantPlagiarism)
      body.append('check_ai', wantAi)
      body.append('check_news', wantNews)

      setResult(await api.analyzeText(body))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const plag = result?.plagiarism
  const ai = result?.ai_text
  const news = result?.news_credibility

  // Regions to mark. Both checks report character offsets into the submitted
  // text, so they can be rendered in one pass over the same string.
  const plagSpans = plag?.available ? (plag.matched_spans ?? []) : []
  const aiSpans = ai?.available ? (ai.flagged_sentences ?? []) : []
  const newsSpans = news?.available ? (news.flagged_sentences ?? []) : []

  return (
    <div className="space-y-5 max-w-4xl">
      <Panel index={0} title="Text Analysis">
        <p className="text-[13px] -mt-1 mb-5 leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
          Paste text to check. Plagiarism is measured exactly against a reference you supply;
          AI detection combines a trained text classifier with explainable style indicators.
          News credibility looks at how a piece is written — it cannot check whether a claim is true.
        </p>

        {/* The three filters */}
        <div className="grid sm:grid-cols-3 gap-3 mb-5">
          <Toggle
            checked={wantPlagiarism}
            onChange={setWantPlagiarism}
            accent="var(--warning)"
            title="Plagiarism"
            body="Exact 5-word-sequence overlap against a reference text. Needs something to compare with — it does not search the web."
          />
          <Toggle
            checked={wantAi}
            onChange={setWantAi}
            accent="var(--brand)"
            title="AI-generated content"
            body="A trained human-vs-AI classifier supported by sentence variation, vocabulary, repetition and phrasing signals."
          />
          <Toggle
            checked={wantNews}
            onChange={setWantNews}
            accent="var(--signal)"
            title="Fake news signals"
            body="Missing attribution, sensationalist phrasing, loaded language and unfalsifiable claims. Describes the writing, not the truth of it."
          />
        </div>

        <label className="block mb-4">
          <span className="text-[12px] mb-1.5 flex items-center justify-between" style={{ color: 'var(--ink-2)' }}>
            <span>Text to check</span>
            <span className="tnum" style={{ color: words < 40 && words > 0 ? 'var(--warning)' : 'var(--ink-muted)' }}>
              {words} words{words > 0 && words < 40 ? ' — 40+ needed for AI indicators' : ''}
            </span>
          </span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={9}
            placeholder="Paste the text you want to analyse…"
            className="w-full px-4 py-3 rounded-xl text-[13px] outline-none resize-y leading-relaxed"
            style={{
              background: 'rgba(3,7,18,.5)',
              border: '1px solid var(--border)',
              color: 'var(--ink)',
            }}
          />
        </label>

        {wantPlagiarism && (
          <label className="block mb-5">
            <span className="text-[12px] mb-1.5 block" style={{ color: 'var(--ink-2)' }}>
              Reference text to compare against
            </span>
            <textarea
              value={reference}
              onChange={(e) => setReference(e.target.value)}
              rows={5}
              placeholder="Paste the suspected source — an article, a classmate's essay, documentation…"
              className="w-full px-4 py-3 rounded-xl text-[13px] outline-none resize-y leading-relaxed"
              style={{
                background: 'rgba(3,7,18,.5)',
                border: '1px solid var(--border)',
                color: 'var(--ink)',
              }}
            />
          </label>
        )}

        <button
          onClick={run}
          disabled={!canRun}
          className="px-7 py-2.5 rounded-full text-sm font-semibold press disabled:opacity-40"
          style={{
            background: 'linear-gradient(135deg, var(--brand), var(--brand-2))',
            color: 'var(--on-accent)',
            boxShadow: canRun ? '0 0 28px -10px rgba(0,240,255,.8)' : 'none',
            cursor: canRun ? 'pointer' : 'not-allowed',
          }}
        >
          {busy ? 'Analysing…' : 'Analyse text'}
        </button>

        {error && (
          <div className="mt-4 px-4 py-3 rounded-lg text-[13px]"
               style={{
                 background: 'color-mix(in srgb, var(--critical) 12%, transparent)',
                 border: '1px solid color-mix(in srgb, var(--critical) 34%, transparent)',
                 color: 'var(--ink-2)',
               }}>
            <strong style={{ color: 'var(--critical)' }}>✕ </strong>{error}
          </div>
        )}
      </Panel>

      {/* ------------------------------------------------------ plagiarism --- */}
      {plag && (
        <Panel index={1} title="Plagiarism">
          {!plag.available ? (
            <p className="text-[13px]" style={{ color: 'var(--ink-muted)' }}>{plag.reason}</p>
          ) : (
            <>
              <ScoreBlock
                label="Overlap with reference"
                percent={plag.overlap_percent}
                verdict={plag.verdict}
                caption={`${plag.matched_ngrams} of ${plag.total_ngrams} ${plag.ngram_size}-word sequences matched`}
              />

              <div className="mt-4">
                <Meter value={plag.overlap_percent / 100} color={colourFor(plag.verdict)} />
              </div>

              {plag.matched_spans?.length > 0 && (
                <p className="text-[12px] mt-4" style={{ color: 'var(--ink-muted)' }}>
                  {plag.flagged_words} of {plag.total_words} words match the reference —
                  marked below.
                </p>
              )}

              <p className="text-[11.5px] mt-4 pt-3 border-t leading-relaxed"
                 style={{ color: 'var(--ink-muted)', borderColor: 'var(--border)' }}>
                {plag.note}
              </p>
            </>
          )}
        </Panel>
      )}

      {/* -------------------------------------------------------- AI text --- */}
      {ai && (
        <Panel index={2} title="AI-Generated Content">
          {!ai.available ? (
            <p className="text-[13px]" style={{ color: 'var(--ink-muted)' }}>{ai.reason}</p>
          ) : (
            <>
              <ScoreBlock
                label="AI-generated probability"
                percent={ai.ai_likelihood_percent}
                verdict={ai.verdict}
                caption={`${ai.word_count} words · ${ai.sentence_count} sentences${ai.model_test_accuracy_percent ? ` · model test accuracy ${ai.model_test_accuracy_percent}%` : ''}`}
              />

              <div className="mt-4">
                <Meter value={ai.ai_likelihood_percent / 100} color={colourFor(ai.verdict)} />
              </div>

              <div className="mt-5">
                <div className="text-[12px] mb-2.5" style={{ color: 'var(--ink-2)' }}>
                  Evidence used for this score
                </div>
                <ul className="space-y-2.5">
                  {ai.signals.map((s, i) => (
                    <li key={s.name} className="fade-in" style={{ '--i': i }}>
                      <div className="flex items-center gap-3 text-[12.5px]">
                        <span className="flex-1">{s.name}</span>
                        <span className="w-24">
                          <Meter value={s.strength / 100} color="var(--brand)" height={4} />
                        </span>
                        <span className="tnum w-12 text-right" style={{ color: 'var(--ink-2)' }}>
                          {s.strength.toFixed(0)}%
                        </span>
                        <span className="tnum w-10 text-right text-[11px]" style={{ color: 'var(--ink-muted)' }}>
                          ×{s.weight}%
                        </span>
                      </div>
                      <div className="text-[11px] ml-0.5 mt-0.5" style={{ color: 'var(--ink-muted)' }}>
                        {s.detail}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>

              {ai.flagged_sentences?.length > 0 && (
                <p className="text-[12px] mt-4" style={{ color: 'var(--ink-muted)' }}>
                  {ai.flagged_sentences.length} passage
                  {ai.flagged_sentences.length === 1 ? '' : 's'} carry the strongest
                  indicators ({ai.flagged_word_count} of {ai.total_word_count} words) —
                  marked below.
                </p>
              )}

              <div className="mt-5 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                <p className="text-[12px] leading-relaxed"
                   style={{ color: 'var(--warning)' }}>
                  ⚠ {ai.note}
                </p>
              </div>
            </>
          )}
        </Panel>
      )}

      {/* -------------------------------------------- news credibility --- */}
      {news && (
        <Panel index={3} title="Fake News Signals">
          {!news.available ? (
            <p className="text-[13px]" style={{ color: 'var(--ink-muted)' }}>{news.reason}</p>
          ) : (
            <>
              <ScoreBlock
                label="Credibility concern"
                percent={news.concern_percent}
                verdict={news.verdict}
                caption={`${news.word_count} words, ${news.sentence_count} sentences`}
              />

              <div className="mt-4">
                <Meter value={news.concern_percent / 100} color={colourFor(news.verdict)} />
              </div>

              <div className="mt-5">
                <div className="text-[12px] mb-2.5" style={{ color: 'var(--ink-2)' }}>
                  What the score is made of
                </div>
                <ul className="space-y-2.5">
                  {news.signals.map((sig, i) => (
                    <li key={sig.name} className="fade-in" style={{ '--i': i }}>
                      <div className="flex items-center gap-3 text-[12.5px]">
                        <span className="flex-1">{sig.name}</span>
                        <span className="w-24">
                          <Meter value={sig.strength / 100} color="var(--signal)" height={4} />
                        </span>
                        <span className="tnum w-12 text-right" style={{ color: 'var(--ink-2)' }}>
                          {sig.strength.toFixed(0)}%
                        </span>
                        <span className="tnum w-10 text-right text-[11px]" style={{ color: 'var(--ink-muted)' }}>
                          ×{sig.weight}%
                        </span>
                      </div>
                      <div className="text-[11px] ml-0.5 mt-0.5" style={{ color: 'var(--ink-muted)' }}>
                        {sig.detail}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>

              {news.attribution_found?.length > 0 && (
                <p className="text-[12px] mt-4" style={{ color: 'var(--ink-muted)' }}>
                  Attribution found: {news.attribution_found.join(', ')}
                </p>
              )}

              {news.flagged_sentences?.length > 0 && (
                <p className="text-[12px] mt-2" style={{ color: 'var(--ink-muted)' }}>
                  {news.flagged_sentences.length} passage
                  {news.flagged_sentences.length === 1 ? '' : 's'} carry the strongest
                  signals ({news.flagged_word_count} of {news.total_word_count} words) —
                  marked below.
                </p>
              )}

              <div className="mt-5 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                <p className="text-[12px] leading-relaxed" style={{ color: 'var(--warning)' }}>
                  ⚠ {news.note}
                </p>
              </div>
            </>
          )}
        </Panel>
      )}

      {/* ------------------------------------------------- marked-up text --- */}
      {(plagSpans.length > 0 || aiSpans.length > 0 || newsSpans.length > 0) && (
        <Panel index={4} title="Suspected Areas">
          <p className="text-[13px] -mt-1 mb-3" style={{ color: 'var(--ink-muted)' }}>
            Your text with every flagged region marked. Hover any highlight to see why.
          </p>

          <Legend showPlagiarism={plagSpans.length > 0} showAi={aiSpans.length > 0}
                  showNews={newsSpans.length > 0} />

          <div className="text-[13px] leading-[1.95] px-4 py-3.5 rounded-xl whitespace-pre-wrap"
               style={{
                 background: 'rgba(3,7,18,.45)',
                 border: '1px solid var(--border)',
                 color: 'var(--ink-2)',
               }}>
            {highlight(text, plagSpans, aiSpans, newsSpans)}
          </div>

          {aiSpans.length > 0 && (
            <div className="mt-5">
              <div className="text-[12px] mb-2" style={{ color: 'var(--ink-2)' }}>
                Why each passage was flagged
              </div>
              <ul className="space-y-2">
                {aiSpans.map((s, i) => (
                  <li key={i} className="text-[12px] px-3 py-2 rounded-lg fade-in"
                      style={{
                        '--i': i,
                        background: 'color-mix(in srgb, var(--brand) 8%, transparent)',
                        borderLeft: '2px solid var(--brand)',
                      }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="tnum font-semibold" style={{ color: 'var(--brand)' }}>
                        {s.strength}%
                      </span>
                      <span style={{ color: 'var(--ink-muted)' }}>· {s.words} words</span>
                    </div>
                    <ul className="list-disc ml-4 space-y-0.5" style={{ color: 'var(--ink-2)' }}>
                      {s.reasons.map((r, j) => <li key={j}>{r}</li>)}
                    </ul>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Panel>
      )}
    </div>
  )
}

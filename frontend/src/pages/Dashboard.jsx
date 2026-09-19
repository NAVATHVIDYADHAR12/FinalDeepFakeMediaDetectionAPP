import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { api, timeAgo } from '../api.js'
import { Donut } from '../components/charts.jsx'
import { StandaloneNotice, EmptyState, ModelsMissing, Panel, Spinner, StatTile, VerdictBadge } from '../components/ui.jsx'
import UploadZone from '../components/UploadZone.jsx'

export default function Dashboard({ health }) {
  const [stats, setStats] = useState(null)
  const [scans, setScans] = useState(null)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  const refresh = () => {
    Promise.all([api.stats(), api.scans({ limit: 6 })])
      .then(([s, r]) => { setStats(s); setScans(r.scans); setError(null) })
      .catch((e) => setError(e.message))
  }

  useEffect(refresh, [])

  if (error) {
    return (
      <EmptyState icon="⚠" title="Could not load the dashboard"
                  body={error} />
    )
  }
  if (!stats) return <Spinner label="Loading dashboard…" />

  const trend = stats.trend ?? []
  const totals = trend.map((d) => d.total)
  const fakes = trend.map((d) => d.fake)
  const cleans = trend.map((d) => d.total - d.fake)

  const segments = [
    { label: 'Authentic',  value: stats.authentic,  color: 'var(--good)' },
    { label: 'Suspicious', value: stats.suspicious, color: 'var(--warning)' },
    { label: 'Fake / Manipulated', value: stats.fake, color: 'var(--critical)' },
  ]

  const last = scans?.[0]

  return (
    <div className="space-y-5">
      {health?.engine === 'browser' ? <StandaloneNotice />
        : health && !health.models_loaded ? <ModelsMissing /> : null}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)]">
        <Panel title="Start a New Scan">
          <p className="text-[13px] mb-4 -mt-1" style={{ color: 'var(--ink-muted)' }}>
            Upload an image or video to analyse its authenticity.
          </p>
          <UploadZone compact disabled={Boolean(health) && !health.models_loaded && health.engine !== 'browser'}
                      onComplete={(report) => navigate(`/report/${report.scan_id}`)} />
        </Panel>

        <div className="grid grid-cols-2 gap-4 content-start">
          <StatTile index={0} label="Total Scans" value={stats.total_scans}
                    icon="◎" color="var(--brand)" trend={totals}
                    sub={stats.avg_processing_ms ? `${stats.avg_processing_ms.toFixed(0)} ms avg` : null} />
          <StatTile index={1} label="Authentic" value={stats.authentic}
                    icon="✓" color="var(--good)" trend={cleans}
                    sub={`${stats.authentic_pct}%`} />
          <StatTile index={2} label="Suspicious" value={stats.suspicious}
                    icon="!" color="var(--warning)"
                    sub={`${stats.suspicious_pct}%`} />
          <StatTile index={3} label="Fake / Manipulated" value={stats.fake}
                    icon="✕" color="var(--critical)" trend={fakes}
                    sub={`${stats.fake_pct}%`} />
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel index={4} title="Authenticity Distribution">
          {stats.total_scans === 0 ? (
            <EmptyState icon="◔" title="No scans yet"
                        body="Analyse a file and the distribution will appear here." />
          ) : (
            <>
              <Donut segments={segments}
                     centerValue={`${stats.authentic_pct}%`}
                     centerLabel="Authentic" />
              <p className="text-[11px] mt-4 pt-3 border-t" style={{ color: 'var(--ink-muted)', borderColor: 'var(--border)' }}>
                {`Based on ${stats.verified_scans.toLocaleString()} verified scan${stats.verified_scans === 1 ? '' : 's'}.`}
                {stats.unverified_count > 0 && ` ${stats.unverified_count.toLocaleString()} historical scan${stats.unverified_count === 1 ? ' was' : 's were'} not verified and excluded.`}
              </p>
            </>
          )}
        </Panel>

        <Panel index={5} title="Recent Scans"
               action={<Link to="/history" className="text-xs" style={{ color: 'var(--brand)' }}>View All</Link>}>
          {!scans?.length ? (
            <EmptyState icon="≡" title="Nothing scanned yet"
                        body="Your recent analyses will be listed here." />
          ) : (
            <ul className="divide-y" style={{ borderColor: 'var(--border)' }}>
              {scans.map((s) => (
                <li key={s.scan_id}>
                  <Link to={`/report/${s.scan_id}`}
                        className="flex items-center gap-3 py-2.5 px-2 -mx-2 rounded-lg text-sm row-hover">
                    <span className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 text-xs"
                          style={{ background: 'var(--surface-2)' }} aria-hidden="true">
                      {s.media_type === 'video' ? '▶' : '▣'}
                    </span>
                    <span className="flex-1 truncate">{s.filename}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded shrink-0"
                          style={{ background: 'var(--surface-2)', color: 'var(--ink-muted)' }}>
                      {s.media_type}
                    </span>
                    <VerdictBadge verdict={s.verdict} />
                    <span className="text-xs shrink-0 w-[70px] text-right"
                          style={{ color: 'var(--ink-muted)' }}>
                      {timeAgo(s.created_at)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      {last && (
        <Panel index={6} title="Last Scan Result"
               action={<Link to={`/report/${last.scan_id}`} className="text-xs" style={{ color: 'var(--brand)' }}>
                 View Full Report →
               </Link>}>
          <div className="flex flex-wrap items-center gap-x-10 gap-y-4">
            <div>
              <div className="text-xs mb-1" style={{ color: 'var(--ink-muted)' }}>File</div>
              <div className="font-medium truncate max-w-[280px]">{last.filename}</div>
            </div>
            <div>
              <div className="text-xs mb-1" style={{ color: 'var(--ink-muted)' }}>Authenticity Score</div>
              <div className="text-2xl figure"
                   style={{ color: last.verdict === 'AUTHENTIC' ? 'var(--good)' : last.verdict === 'FAKE' ? 'var(--critical)' : 'var(--warning)' }}>
                {last.authenticity_score != null ? `${last.authenticity_score}%` : "\u2014"}
              </div>
            </div>
            <div>
              <div className="text-xs mb-1" style={{ color: 'var(--ink-muted)' }}>Model certainty</div>
              <div className="text-2xl figure">{last.confidence != null ? `${(last.confidence * 100).toFixed(0)}%` : "\u2014"}</div>
            </div>
            <div>
              <div className="text-xs mb-1" style={{ color: 'var(--ink-muted)' }}>Risk Level</div>
              <div className="text-2xl figure">{last.risk_level ?? '—'}</div>
            </div>
            <div>
              <div className="text-xs mb-1.5" style={{ color: 'var(--ink-muted)' }}>Verdict</div>
              <VerdictBadge verdict={last.verdict} size="lg" />
            </div>
          </div>
        </Panel>
      )}
    </div>
  )
}

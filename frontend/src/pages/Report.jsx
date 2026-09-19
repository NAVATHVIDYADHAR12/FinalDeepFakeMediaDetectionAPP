import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api, formatBytes, verdictMeta } from '../api.js'
import { FrameTimeline, Meter } from '../components/charts.jsx'
import { EmptyState, Findings, KeyValue, Panel, Spinner, Timeline, VerdictBadge } from '../components/ui.jsx'

/** Big headline figure. The number is the deliverable, so it gets the space. */
function ScoreHero({ report }) {
  const meta = verdictMeta(report.verdict)
  // A report from the browser engine has no classifier behind it, so the score
  // is genuinely absent. An em-dash is shown rather than a fabricated number.
  const scored = report.fake_probability !== null && report.fake_probability !== undefined

  return (
    <div className="flex flex-wrap items-center gap-x-12 gap-y-5">
      <div>
        <div className="text-xs mb-1.5" style={{ color: 'var(--ink-muted)' }}>Model-estimated authenticity</div>
        <div className="text-5xl leading-none figure" style={{ color: meta.color }}>
          {scored ? `${report.authenticity_score}%` : '\u2014'}
        </div>
        <div className="text-xs mt-2" style={{ color: meta.color }}>
          {!scored ? 'No classifier available'
            : report.verdict === 'AUTHENTIC' ? 'No strong manipulation signal detected'
            : 'Likely AI-generated or edited'}
        </div>
      </div>
      <div>
        <div className="text-xs mb-1.5" style={{ color: 'var(--ink-muted)' }}>
          {report.confidence_kind === 'threshold_calibrated_model_score' ? 'AI likelihood'
            : report.confidence_kind === 'uncalibrated_model_certainty' ? 'Model certainty' : 'Confidence'}
        </div>
        <div className="text-4xl leading-none figure">
          {scored ? `${(report.confidence * 100).toFixed(0)}%` : '\u2014'}
        </div>
        <div className="text-xs mt-2" style={{ color: 'var(--ink-muted)' }}>
          {!scored ? 'Not measured'
            : report.confidence_kind === 'threshold_calibrated_model_score' ? 'AI-generated threshold: 65%'
            : report.confidence_kind === 'uncalibrated_model_certainty' ? 'Not calibrated across image sources'
            : report.confidence > 0.7 ? 'Very high'
            : report.confidence > 0.4 ? 'Moderate' : 'Low \u2014 treat with care'}
        </div>
      </div>
      <div>
        <div className="text-xs mb-1.5" style={{ color: 'var(--ink-muted)' }}>Risk Level</div>
        <div className="text-4xl leading-none figure">{report.risk_level ?? '\u2014'}</div>
      </div>
      <div>
        <div className="text-xs mb-2" style={{ color: 'var(--ink-muted)' }}>Verdict</div>
        <VerdictBadge verdict={report.verdict} size="lg" />
      </div>
    </div>
  )
}

/** Per-face panel with the CAM heatmap toggle. */
function FaceCard({ face }) {
  const [view, setView] = useState('heatmap')
  const meta = verdictMeta(face.verdict)
  const image = view === 'heatmap' ? (face.heatmap_preview ?? face.crop_preview) : face.crop_preview

  return (
    <div className="rounded-xl p-4" style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium">
          {face.is_full_frame ? 'Full frame (no face found)' : `Face ${face.face_id}`}
        </span>
        <VerdictBadge verdict={face.verdict} />
      </div>

      <div className="flex gap-4 flex-wrap">
        {image && (
          <div>
            <img src={image} alt={`Face ${face.face_id} ${view}`}
                 className="w-[168px] h-[168px] object-cover rounded-lg"
                 style={{ border: '1px solid var(--border)' }} />
            {face.heatmap_preview && (
              <div className="flex gap-1 mt-2" role="tablist">
                {['heatmap', 'original'].map((mode) => (
                  <button key={mode} role="tab" aria-selected={view === mode}
                          onClick={() => setView(mode)}
                          className="flex-1 text-[11px] py-1 rounded capitalize"
                          style={{
                            background: view === mode ? 'var(--brand-2)' : 'var(--surface-3)',
                            color: view === mode ? 'var(--ink)' : 'var(--ink-2)',
                          }}>
                    {mode}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="flex-1 min-w-[220px] space-y-3">
          {face.fake_probability != null ? (
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span style={{ color: 'var(--ink-muted)' }}>Fake probability</span>
                <span className="tnum font-semibold" style={{ color: meta.color }}>
                  {(face.fake_probability * 100).toFixed(1)}%
                </span>
              </div>
              <Meter value={face.fake_probability} color={meta.color} />
            </div>
          ) : (
            <p className="text-xs" style={{ color: 'var(--ink-muted)' }}>
              No classifier available &mdash; forensic checks only.
            </p>
          )}

          <KeyValue rows={[
            ['Model agreement', face.model_agreement != null ? `${(face.model_agreement * 100).toFixed(0)}%` : '\u2014'],
            ['Detection score', face.detection_score != null ? face.detection_score.toFixed(3) : '—'],
            ['Identity vector', face.has_embedding ? 'extracted' : 'unavailable'],
          ]} />

          <div className="pt-2 border-t" style={{ borderColor: 'var(--border)' }}
               hidden={!(face.models ?? []).length}>
            <div className="text-[11px] mb-2" style={{ color: 'var(--ink-muted)' }}>Per-model verdict</div>
            <ul className="space-y-1.5">
              {(face.models ?? []).map((m) => (
                <li key={m.arch} className="flex items-center gap-2 text-xs">
                  <span className="flex-1 truncate">{m.name}</span>
                  <span className="w-16"><Meter value={m.fake_probability} color={verdictMeta(m.verdict).color} height={4} /></span>
                  <span className="tnum w-11 text-right">{(m.fake_probability * 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Report() {
  const { id } = useParams()
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setReport(null)
    api.scan(id).then(setReport).catch((e) => setError(e.message))
  }, [id])

  if (error) return <EmptyState icon="⚠" title="Report not found" body={error} />
  if (!report) return <Spinner label="Loading report…" />

  const isVideo = report.media_type === 'video'

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3 flex-wrap">
        <Link to="/history" className="text-sm" style={{ color: 'var(--brand)' }}>← History</Link>
        <h1 className="text-xl font-bold truncate headline">{report.filename}</h1>
        <span className="text-xs px-2 py-0.5 rounded"
              style={{ background: 'var(--surface-2)', color: 'var(--ink-muted)' }}>
          {report.scan_id}
        </span>
      </div>

      {report.engine === 'browser' && (
        <div className="panel p-4 rise"
             style={{ borderColor: 'color-mix(in srgb, var(--warning) 38%, transparent)' }}>
          <div className="flex items-start gap-3">
            <span style={{ color: 'var(--warning)' }} aria-hidden="true">&#9888;</span>
            <p className="text-[13px]" style={{ color: 'var(--ink-2)' }}>{report.engine_note}</p>
          </div>
        </div>
      )}

      <Panel index={0}>
        <ScoreHero report={report} />
        {report.generation_analysis && (
          <p className="text-xs mt-5 pt-4 border-t" style={{ color: 'var(--ink-muted)', borderColor: 'var(--border)' }}>
            {report.decision_reliability === 'LIMITED'
              ? 'Final verdict withheld because input quality can make the model unstable. The estimate is shown for inspection only.'
              : 'AI detection is probabilistic, not proof. Realistic generation, resizing, screenshots and heavy compression can change the score; use provenance and source context alongside this result.'}
          </p>
        )}
      </Panel>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <Panel index={1} title={isVideo ? 'Per-Frame Analysis' : 'Face Analysis'}>
          {isVideo ? (
            <>
              <FrameTimeline frames={report.frame_scores} />
              {report.most_suspicious_frame && (
                <div className="mt-5 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
                  <div className="text-xs mb-2.5" style={{ color: 'var(--ink-muted)' }}>
                    Most suspicious frame — #{report.most_suspicious_frame.frame}
                    {report.most_suspicious_frame.timestamp_sec != null && ` at ${report.most_suspicious_frame.timestamp_sec}s`}
                  </div>
                  <div className="flex gap-3 flex-wrap">
                    <figure>
                      <img src={report.most_suspicious_frame.preview} alt="Most suspicious frame"
                           className="w-[168px] h-[168px] object-cover rounded-lg"
                           style={{ border: '1px solid var(--border)' }} />
                      <figcaption className="text-[11px] mt-1.5" style={{ color: 'var(--ink-muted)' }}>Original</figcaption>
                    </figure>
                    {report.most_suspicious_frame.heatmap_preview && (
                      <figure>
                        <img src={report.most_suspicious_frame.heatmap_preview} alt="Activation heatmap"
                             className="w-[168px] h-[168px] object-cover rounded-lg"
                             style={{ border: '1px solid var(--border)' }} />
                        <figcaption className="text-[11px] mt-1.5" style={{ color: 'var(--ink-muted)' }}>
                          Model attention
                        </figcaption>
                      </figure>
                    )}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="space-y-4">
              {report.faces.map((f) => <FaceCard key={f.face_id} face={f} />)}
            </div>
          )}
        </Panel>

        <div className="space-y-5">
          <Panel index={2} title="Key Findings"><Findings items={report.findings} /></Panel>

          <Panel index={3} title="File Details">
            <KeyValue rows={[
              ['Type', report.media_type],
              ['Dimensions', report.dimensions],
              ['Size', formatBytes(report.file_size_bytes)],
              ...(isVideo ? [
                ['Duration', `${report.duration_sec}s`],
                ['Frame rate', `${report.fps} fps`],
                ['Frames analysed', `${report.frames_analyzed} of ${report.total_frames}`],
                ['People tracked', report.people_detected],
              ] : [
                ['Faces detected', report.faces_detected],
              ]),
              ['Processing time', `${(report.processing_ms ?? 0).toFixed(0)} ms`],
            ]} />
          </Panel>
        </div>
      </div>

      {isVideo && report.tracks?.length > 0 && (
        <Panel index={4} title="Per-Person Tracking">
          <p className="text-[13px] -mt-1 mb-3" style={{ color: 'var(--ink-muted)' }}>
            Each person is followed across sampled frames by face embedding. A genuine face
            scores consistently; a swapped one flickers as the generator struggles with pose
            and occlusion — that variance is the temporal signal.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs" style={{ color: 'var(--ink-muted)' }}>
                  <th className="py-2 pr-4 font-medium">Person</th>
                  <th className="py-2 pr-4 font-medium">Frames</th>
                  <th className="py-2 pr-4 font-medium">Mean score</th>
                  <th className="py-2 pr-4 font-medium">Peak</th>
                  <th className="py-2 pr-4 font-medium">Variance (σ)</th>
                  <th className="py-2 font-medium">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {report.tracks.map((t) => (
                  <tr key={t.track_id} className="border-t row-hover" style={{ borderColor: 'var(--border)' }}>
                    <td className="py-2.5 pr-4">Person {t.track_id}</td>
                    <td className="py-2.5 pr-4 tnum">{t.frames_seen}</td>
                    <td className="py-2.5 pr-4 tnum">{(t.mean_fake_probability * 100).toFixed(1)}%</td>
                    <td className="py-2.5 pr-4 tnum">{(t.max_fake_probability * 100).toFixed(1)}%</td>
                    <td className="py-2.5 pr-4 tnum">
                      {t.score_std.toFixed(3)}
                      {t.temporally_inconsistent && (
                        <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded"
                              style={{ background: 'color-mix(in srgb, var(--critical) 18%, transparent)', color: 'var(--critical)' }}>
                          unstable
                        </span>
                      )}
                    </td>
                    <td className="py-2.5"><VerdictBadge verdict={t.verdict} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {report.models?.length > 0 && (
        <Panel index={5} title="Model Comparison">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs" style={{ color: 'var(--ink-muted)' }}>
                  <th className="py-2 pr-4 font-medium">Model / Engine</th>
                  <th className="py-2 pr-4 font-medium">Result</th>
                  <th className="py-2 pr-4 font-medium w-[180px]">Fake probability</th>
                  <th className="py-2 font-medium">Test accuracy</th>
                </tr>
              </thead>
              <tbody>
                {report.models.map((m) => (
                  <tr key={m.arch} className="border-t row-hover" style={{ borderColor: 'var(--border)' }}>
                    <td className="py-2.5 pr-4">{m.name}</td>
                    <td className="py-2.5 pr-4"><VerdictBadge verdict={m.verdict} /></td>
                    <td className="py-2.5 pr-4">
                      <div className="flex items-center gap-2.5">
                        <span className="flex-1"><Meter value={m.fake_probability} color={verdictMeta(m.verdict).color} /></span>
                        <span className="tnum w-11 text-right text-xs">{(m.fake_probability * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="py-2.5 tnum">
                      {m.test_accuracy != null ? `${(m.test_accuracy * 100).toFixed(1)}%` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <div className="grid gap-5 lg:grid-cols-3">
        <Panel index={6} title="Metadata">
          <KeyValue rows={[
            ['EXIF present', report.metadata.has_exif ? 'yes' : 'no'],
            ['Camera', report.metadata.camera_make
              ? `${report.metadata.camera_make} ${report.metadata.camera_model ?? ''}`.trim() : '—'],
            ['Software', report.metadata.software ?? '—'],
            ['Captured', report.metadata.datetime ?? '—'],
            ['GPS', report.metadata.has_gps ? 'present' : 'none'],
            ['Stripped', report.metadata.metadata_stripped ? 'yes' : 'no'],
          ]} />
        </Panel>

        <Panel index={7} title="Provenance (C2PA)">
          <div className="flex items-center gap-2 mb-2.5">
            <span aria-hidden="true" style={{ color: report.c2pa.signature_found ? 'var(--good)' : 'var(--ink-muted)' }}>
              {report.c2pa.signature_found ? '✓' : '○'}
            </span>
            <span className="font-medium text-sm">
              {report.c2pa.signature_found ? 'Credential present' : 'Not found'}
            </span>
          </div>
          <p className="text-[12px] leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
            {report.c2pa.note}
          </p>
        </Panel>

        <Panel index={8} title="Error Level Analysis">
          {report.ela?.error ? (
            <p className="text-[12px]" style={{ color: 'var(--ink-muted)' }}>Unavailable for this file.</p>
          ) : report.ela ? (
            <>
              <KeyValue rows={[
                ['Mean error', report.ela.mean_error?.toFixed(4)],
                ['Block spread', report.ela.block_spread?.toFixed(4)],
              ]} />
              <p className="text-[12px] mt-2.5 leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
                {report.ela.note}
              </p>
            </>
          ) : (
            <p className="text-[12px]" style={{ color: 'var(--ink-muted)' }}>Not run for video.</p>
          )}
        </Panel>
      </div>

      <Panel index={9} title="Evidence Timeline">
        <Timeline steps={report.timeline} />
      </Panel>
    </div>
  )
}

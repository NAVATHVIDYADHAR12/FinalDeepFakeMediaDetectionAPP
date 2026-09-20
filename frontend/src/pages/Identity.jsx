import { useEffect, useRef, useState } from 'react'

import { api, verdictMeta } from '../api.js'
import { Meter } from '../components/charts.jsx'
import { EmptyState, Panel, Spinner, VerdictBadge } from '../components/ui.jsx'

/**
 * Face recognition: enrol known people, then identify faces in a new image and
 * report each one's deepfake verdict alongside the identity. The pairing is the
 * point - "this claims to be X, and the face is manipulated" is the useful
 * statement, not either half on its own.
 */
export default function Identity() {
  const [identities, setIdentities] = useState(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [matches, setMatches] = useState(null)
  const [cameraMode, setCameraMode] = useState(null)
  const [cameraReady, setCameraReady] = useState(false)
  const [facingMode, setFacingMode] = useState('user')

  const enrollRef = useRef(null)
  const matchRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)

  const load = () => api.identities().then((r) => setIdentities(r.identities)).catch((e) => setError(e.message))

  // The braces matter. `load` returns a promise, and passing it directly as
  // the effect body made React treat that promise as the cleanup function -
  // it called it on unmount and threw "destroy is not a function", which took
  // the whole page down to a blank screen. Awaiting is still wanted at the
  // other call sites, so `load` keeps returning the promise and only the
  // effect discards it.
  useEffect(() => { load() }, [])
  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
  }, [])

  useEffect(() => {
    if (cameraMode && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.play().catch(() => {})
    }
  }, [cameraMode])

  const enroll = async (file) => {
    if (!name.trim()) { setError('Enter a name before choosing a photo.'); return }
    setBusy(true); setError(null)
    try {
      await api.enroll(name.trim(), file)
      setName('')
      await load()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const identify = async (file) => {
    setBusy(true); setError(null); setMatches(null)
    try {
      setMatches(await api.matchIdentity(file))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const remove = async (n) => { await api.deleteIdentity(n); load() }

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setCameraReady(false)
  }

  const closeCamera = () => {
    stopCamera()
    setCameraMode(null)
  }

  const openCamera = async (mode, requestedFacing = facingMode) => {
    if (mode === 'enroll' && !name.trim()) {
      setError('Enter a name before taking an enrolment photo.')
      return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera capture is not supported by this browser. Use HTTPS or choose a photo instead.')
      return
    }

    stopCamera()
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: requestedFacing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })
      streamRef.current = stream
      setFacingMode(requestedFacing)
      setCameraMode(mode)
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play().catch(() => {})
        }
      })
    } catch (e) {
      const denied = e?.name === 'NotAllowedError' || e?.name === 'SecurityError'
      setError(denied
        ? 'Camera permission was denied. Allow camera access in your browser, then try again.'
        : `Could not open the camera${e?.message ? `: ${e.message}` : '.'}`)
    }
  }

  const switchCamera = () => {
    const next = facingMode === 'user' ? 'environment' : 'user'
    openCamera(cameraMode, next)
  }

  const capture = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      setError('The camera is still starting. Wait a moment and try again.')
      return
    }
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
    const mode = cameraMode
    canvas.toBlob((blob) => {
      if (!blob) {
        setError('Could not capture the camera frame. Please try again.')
        return
      }
      const file = new File([blob], `camera-${Date.now()}.jpg`, { type: 'image/jpeg' })
      closeCamera()
      if (mode === 'enroll') enroll(file)
      else identify(file)
    }, 'image/jpeg', 0.92)
  }

  return (
    <div className="space-y-5">
      {error && (
        <div className="px-4 py-3 rounded-lg text-[13px]"
             style={{
               background: 'color-mix(in srgb, var(--critical) 12%, transparent)',
               border: '1px solid color-mix(in srgb, var(--critical) 34%, transparent)',
             }}>
          <strong style={{ color: 'var(--critical)' }}>✕ </strong>{error}
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Enrol a Person">
          <p className="text-[13px] -mt-1 mb-4 leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
            Stores a 128-dimension face embedding, not the photo itself. Enrolling the same
            person again averages the vectors, which makes matching more robust across pose
            and lighting.
          </p>
          <div className="flex gap-2 flex-wrap">
            <input value={name} onChange={(e) => setName(e.target.value)}
                   placeholder="Person's name" aria-label="Person's name"
                   className="flex-1 min-w-[160px] px-3 py-2 rounded-lg text-sm outline-none"
                   style={{ background: 'var(--surface-2)', border: '1px solid var(--border)', color: 'var(--ink)' }} />
            <button onClick={() => enrollRef.current?.click()} disabled={busy || !name.trim()}
                    className="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-45"
                    style={{ background: 'linear-gradient(135deg, var(--brand), var(--brand-2))', color: 'var(--on-accent)' }}>
              Choose photo
            </button>
            <button onClick={() => openCamera('enroll')} disabled={busy || !name.trim()}
                    className="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-45"
                    style={{ background: 'var(--surface-3)', border: '1px solid var(--border-bright)' }}>
              Use camera
            </button>
            <input ref={enrollRef} type="file" accept="image/*" className="hidden"
                   onChange={(e) => e.target.files?.[0] && enroll(e.target.files[0])} />
          </div>
        </Panel>

        <Panel title="Identify Faces">
          <p className="text-[13px] -mt-1 mb-4 leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
            Matches every face in an image against the enrolled gallery and reports its
            deepfake verdict at the same time.
          </p>
          <button onClick={() => matchRef.current?.click()} disabled={busy}
                  className="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-45"
                  style={{ background: 'var(--surface-3)', border: '1px solid var(--border-bright)' }}>
            Upload image to identify
          </button>
          <button onClick={() => openCamera('identify')} disabled={busy}
                  className="ml-2 px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-45"
                  style={{ background: 'linear-gradient(135deg, var(--brand), var(--brand-2))', color: 'var(--on-accent)' }}>
            Use camera
          </button>
          <input ref={matchRef} type="file" accept="image/*" className="hidden"
                 onChange={(e) => e.target.files?.[0] && identify(e.target.files[0])} />
        </Panel>
      </div>

      {busy && <Spinner label="Working…" />}

      {matches && (
        <Panel title="Identification Result">
          <p className="text-[13px] mb-4 -mt-1" style={{ color: 'var(--ink-muted)' }}>
            {matches.faces_detected} face(s) detected · gallery holds {matches.gallery_size} identity(ies)
          </p>
          {!matches.matches.length ? (
            <EmptyState icon="☺" title="No faces found in that image" />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs" style={{ color: 'var(--ink-muted)' }}>
                    <th className="py-2 pr-4 font-medium">Face</th>
                    <th className="py-2 pr-4 font-medium">Identity</th>
                    <th className="py-2 pr-4 font-medium w-[170px]">Similarity</th>
                    <th className="py-2 font-medium">Deepfake verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {matches.matches.map((m) => (
                    <tr key={m.face_id} className="border-t row-hover" style={{ borderColor: 'var(--border)' }}>
                      <td className="py-3 pr-4">Face {m.face_id}</td>
                      <td className="py-3 pr-4 font-medium"
                          style={{ color: m.matched ? 'var(--ink)' : 'var(--ink-muted)' }}>
                        {m.matched ? m.identity : 'Unknown'}
                      </td>
                      <td className="py-3 pr-4">
                        <div className="flex items-center gap-2.5">
                          <span className="flex-1">
                            <Meter value={Math.max(0, m.similarity)}
                                   color={m.matched ? 'var(--good)' : 'var(--ink-muted)'} />
                          </span>
                          <span className="tnum text-xs w-11 text-right">{m.similarity.toFixed(3)}</span>
                        </div>
                      </td>
                      <td className="py-3">
                        {m.verdict ? (
                          <div className="flex items-center gap-2.5">
                            <VerdictBadge verdict={m.verdict} />
                            <span className="tnum text-xs" style={{ color: verdictMeta(m.verdict).color }}>
                              {(m.fake_probability * 100).toFixed(0)}%
                            </span>
                          </div>
                        ) : <span style={{ color: 'var(--ink-muted)' }}>—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      )}

      <Panel title="Enrolled Identities">
        {!identities ? <Spinner />
          : !identities.length ? (
            <EmptyState icon="☺" title="Gallery is empty"
                        body="Enrol someone above to start matching faces against known people." />
          ) : (
            <ul className="divide-y" style={{ borderColor: 'var(--border)' }}>
              {identities.map((i) => (
                <li key={i.name} className="flex items-center gap-3 py-2.5 text-sm">
                  <span className="w-8 h-8 rounded-full flex items-center justify-center shrink-0"
                        style={{ background: 'var(--surface-3)' }} aria-hidden="true">☺</span>
                  <span className="flex-1 font-medium">{i.name}</span>
                  <span className="text-xs" style={{ color: 'var(--ink-muted)' }}>
                    {i.sample_count} sample{i.sample_count === 1 ? '' : 's'}
                  </span>
                  <button onClick={() => remove(i.name)} aria-label={`Remove ${i.name}`}
                          className="text-xs px-2 py-1 rounded" style={{ color: 'var(--ink-muted)' }}>✕</button>
                </li>
              ))}
            </ul>
          )}
      </Panel>

      {cameraMode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
             style={{ background: 'rgba(1,5,14,.86)', backdropFilter: 'blur(8px)' }}
             role="dialog" aria-modal="true" aria-label="Camera capture">
          <div className="w-full max-w-2xl rounded-2xl p-4 sm:p-5"
               style={{ background: 'var(--surface-1)', border: '1px solid var(--border-bright)' }}>
            <div className="flex items-center justify-between gap-3 mb-3">
              <div>
                <h2 className="font-semibold">{cameraMode === 'enroll' ? 'Take enrolment photo' : 'Capture face to identify'}</h2>
                <p className="text-xs mt-0.5" style={{ color: 'var(--ink-muted)' }}>
                  Centre the face, use even lighting, and keep the camera steady.
                </p>
              </div>
              <button onClick={closeCamera} className="px-3 py-2 rounded-lg text-sm"
                      aria-label="Close camera" style={{ border: '1px solid var(--border)' }}>✕</button>
            </div>

            <div className="relative overflow-hidden rounded-xl aspect-video"
                 style={{ background: '#020611', border: '1px solid var(--border)' }}>
              <video ref={videoRef} autoPlay muted playsInline
                     onLoadedMetadata={() => setCameraReady(true)}
                     className="w-full h-full object-cover"
                     style={{ transform: facingMode === 'user' ? 'scaleX(-1)' : 'none' }} />
              {!cameraReady && (
                <div className="absolute inset-0 flex items-center justify-center text-sm"
                     style={{ color: 'var(--ink-muted)' }}>Starting camera…</div>
              )}
            </div>
            <canvas ref={canvasRef} className="hidden" aria-hidden="true" />

            <div className="flex flex-wrap justify-center gap-3 mt-4">
              <button onClick={switchCamera} disabled={!cameraReady}
                      className="px-4 py-2 rounded-lg text-sm disabled:opacity-45"
                      style={{ background: 'var(--surface-3)', border: '1px solid var(--border)' }}>
                ↻ Switch camera
              </button>
              <button onClick={capture} disabled={!cameraReady || busy}
                      className="px-6 py-2 rounded-lg text-sm font-semibold disabled:opacity-45"
                      style={{ background: 'linear-gradient(135deg, var(--brand), var(--brand-2))', color: 'var(--on-accent)' }}>
                ◉ Capture photo
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Thin wrapper over the backend REST API.
 *
 * Paths are relative by default: in development Vite proxies /api to the
 * FastAPI server on :8000, and in production FastAPI serves this build itself,
 * so the same paths work in both cases.
 *
 * When the frontend is hosted separately from the backend — a static deploy on
 * Vercel, for instance — set VITE_API_BASE at build time to the backend's
 * origin (e.g. https://omniguard-api.onrender.com) and every call is redirected
 * there. Empty by default, which keeps the same-origin behaviour.
 */
import * as engine from './engine/index.js'

const ENV_API_BASE = import.meta.env?.VITE_API_BASE ?? ''
const VERCEL_HOST = typeof window !== 'undefined'
  && /(^|\.)final-deep-fake-media-detection(?:-app)?(?:-[a-z0-9]+)?\.vercel\.app$/i
    .test(window.location.hostname)
const HOSTED_API_FALLBACK = VERCEL_HOST
  ? 'https://omniguard-ai-backend.onrender.com'
  : ''

export const API_BASE = (ENV_API_BASE || HOSTED_API_FALLBACK).replace(/\/$/, '')
export const REMOTE_BACKEND_CONFIGURED = Boolean(API_BASE)

/** Prefix a path with the configured API origin. */
export const apiUrl = (path) => `${API_BASE}${path}`

async function request(path, options = {}) {
  const res = await fetch(apiUrl(path), {
    credentials: API_BASE ? 'include' : 'same-origin',
    ...options,
  })

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      // response had no JSON body; keep the status line
    }
    const error = new Error(detail)
    error.status = res.status
    throw error
  }
  return res.json()
}

const upload = (path, formData) => request(path, { method: 'POST', body: formData })

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/** Wait for a sleeping/restarting Render instance before sending a large file.
 * GET retries are safe; the scan POST itself is still sent exactly once. */
async function waitForRemoteBackend(maxWaitMs = 90_000) {
  if (!REMOTE_BACKEND_CONFIGURED) return
  const deadline = Date.now() + maxWaitMs
  let lastError = null
  while (Date.now() < deadline) {
    try {
      const res = await fetch(apiUrl('/api/health'), { credentials: 'include' })
      if (res.ok) return
      lastError = new Error(`${res.status} ${res.statusText}`)
    } catch (error) {
      lastError = error
    }
    await delay(3_000)
  }
  throw new Error(
    `Detection service did not become ready within 90 seconds${lastError?.message ? `: ${lastError.message}` : ''}`
  )
}

const remoteUpload = async (path, formData) => {
  await waitForRemoteBackend()
  return upload(path, formData)
}

/* ---------------------------------------------------------------------------
   Backend detection and standalone fallback.

   Locally the Python service is there and does everything. On a static host it
   is not, and rather than showing a dead interface the app falls back to the
   browser engine: real EXIF, Error Level Analysis, C2PA and history, with the
   neural verdict honestly reported as unavailable.

   The probe result is cached, so this costs one request per page load rather
   than one per call.
--------------------------------------------------------------------------- */

let backendProbe = null

export function resetBackendProbe() { backendProbe = null }

async function backendAvailable() {
  if (backendProbe) return backendProbe

  backendProbe = (async () => {
    try {
      // A short timeout: a sleeping free-tier host should not stall the UI for
      // 30 seconds before the fallback kicks in.
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 4000)
      const res = await fetch(apiUrl('/api/health'), {
        signal: controller.signal,
        credentials: API_BASE ? 'include' : 'same-origin',
      })
      clearTimeout(timer)
      return res.ok
    } catch {
      return false
    }
  })()

  return backendProbe
}

/** Use the server when it is there, the browser engine when it is not. */
async function viaBackendOr(serverCall, localCall) {
  // A configured remote API is authoritative. In particular, Render's free
  // instances can take 50+ seconds to wake after inactivity. Probing them with
  // the short local-development timeout below used to cache a false "offline"
  // result and silently run the browser-only forensic fallback for the rest of
  // the page session. That produced a real metadata report but no neural
  // verdict even though the classifiers were deployed and healthy.
  //
  // Call the configured service directly instead. The browser keeps the
  // request open while a sleeping instance wakes, and a genuine connection
  // failure is surfaced to the user rather than disguised as an unverified
  // local scan.
  if (REMOTE_BACKEND_CONFIGURED) return serverCall()

  if (await backendAvailable()) {
    try {
      return await serverCall()
    } catch (err) {
      // A transport failure mid-session means the server went away; fall
      // through rather than surfacing a network error. A real HTTP status is
      // a genuine answer and is passed on untouched.
      if (err.status) throw err
      backendProbe = Promise.resolve(false)
    }
  }
  return localCall()
}

export const api = {
  health: () => viaBackendOr(
    () => request('/api/health'),
    async () => engine.health(),
  ),

  systemInfo: () => viaBackendOr(
    () => request('/api/system/info'),
    async () => engine.systemInfo(),
  ),

  models: () => viaBackendOr(
    () => request('/api/models'),
    async () => ({ ready: false, engine: 'browser', models: [] }),
  ),

  stats: () => viaBackendOr(
    () => request('/api/stats'),
    () => engine.store.stats(),
  ),

  scans: (opts = {}) => viaBackendOr(
    () => {
      const { limit = 20, offset = 0, verdict } = opts
      const q = new URLSearchParams({ limit, offset })
      if (verdict) q.set('verdict', verdict)
      return request(`/api/scans?${q}`)
    },
    async () => ({ scans: await engine.store.recentScans(opts) }),
  ),

  scan: (id) => viaBackendOr(
    () => request(`/api/scan/${id}`),
    async () => {
      const report = await engine.store.getScan(id)
      if (!report) {
        const err = new Error(`No scan with id ${id}`)
        err.status = 404
        throw err
      }
      return report
    },
  ),

  deleteScan: (id) => viaBackendOr(
    () => request(`/api/scan/${id}`, { method: 'DELETE' }),
    async () => ({ deleted: await engine.store.deleteScan(id) }),
  ),

  /** Routes to the image or video analyser based on file extension. */
  analyze: (file) => viaBackendOr(
    () => {
      const fd = new FormData()
      fd.append('file', file)
      return remoteUpload('/api/scan', fd)
    },
    () => engine.analyze(file),
  ),

  identities: () => viaBackendOr(
    () => request('/api/identities'),
    async () => ({ identities: [], engine: 'browser' }),
  ),

  enroll: (name, file, notes = '') => viaBackendOr(
    () => {
      const fd = new FormData()
      fd.append('name', name)
      fd.append('file', file)
      if (notes) fd.append('notes', notes)
      return remoteUpload('/api/identity/enroll', fd)
    },
    async () => {
      // Face recognition needs the SFace model; there is nothing honest to
      // return without it.
      const err = new Error(
        'Face recognition needs the local service — it uses a face-embedding '
        + 'model that is not part of this build.'
      )
      err.status = 503
      throw err
    },
  ),

  matchIdentity: (file) => viaBackendOr(
    () => {
      const fd = new FormData()
      fd.append('file', file)
      return remoteUpload('/api/identity/match', fd)
    },
    async () => {
      const err = new Error(
        'Face recognition needs the local service — it uses a face-embedding '
        + 'model that is not part of this build.'
      )
      err.status = 503
      throw err
    },
  ),

  deleteIdentity: (name) => viaBackendOr(
    () => request(`/api/identity/${encodeURIComponent(name)}`, { method: 'DELETE' }),
    async () => ({ deleted: name }),
  ),
}

/** Verdict presentation. Icon + label always accompany the color, so meaning
 *  never rests on hue alone. */
export const VERDICT = {
  AUTHENTIC:  { label: 'Authentic',          color: 'var(--good)',     icon: '✓', tone: 'good' },
  SUSPICIOUS: { label: 'Suspicious',         color: 'var(--warning)',  icon: '!', tone: 'warning' },
  FAKE:       { label: 'Fake / Manipulated', color: 'var(--critical)', icon: '✕', tone: 'critical' },
  // Forensics ran, but no classifier was available to give a verdict. Shown in
  // a neutral tone: it is an absence of judgement, not a judgement.
  UNVERIFIED: { label: 'Not verified',       color: 'var(--ink-muted)', icon: '?', tone: 'muted' },
}

export const verdictMeta = (v) => VERDICT[v] ?? {
  label: v ?? 'Unknown', color: 'var(--ink-muted)', icon: '?', tone: 'muted',
}

export const formatBytes = (n) => {
  if (!n && n !== 0) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

export const timeAgo = (iso) => {
  if (!iso) return '—'
  const then = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`)
  const secs = Math.max(0, (Date.now() - then.getTime()) / 1000)
  if (secs < 60) return 'just now'
  if (secs < 3600) return `${Math.floor(secs / 60)} min ago`
  if (secs < 86400) return `${Math.floor(secs / 3600)} hr ago`
  return `${Math.floor(secs / 86400)} d ago`
}

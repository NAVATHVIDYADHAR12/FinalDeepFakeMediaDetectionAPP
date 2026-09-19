/**
 * Standalone analysis engine.
 *
 * Runs entirely in the browser so the hosted build is a working tool rather
 * than a brochure: real EXIF parsing, real Error Level Analysis, real C2PA
 * detection, real history in IndexedDB.
 *
 * What it deliberately does NOT do is produce a deepfake verdict. That needs
 * the trained CNN ensemble — ~50 MB of ONNX that only exists after the
 * training notebook has run. Rather than guess a number, every report from
 * this engine is marked UNVERIFIED and says exactly what is missing.
 *
 * A pretrained third-party detector was evaluated as a substitute and
 * rejected: measured against six genuine photographs it scored 0.44–0.77 with
 * no separation and called four of them fake. A confidently wrong verdict is
 * worse than an absent one.
 */

import { checkC2PA, errorLevelAnalysis, readExif } from './forensics.js'
import * as store from './store.js'

const uid = () =>
  'SCN' + Math.random().toString(36).slice(2, 8).toUpperCase()

const nowISO = () => new Date().toISOString().replace(/\.\d+Z$/, '+00:00')

const IMAGE_EXT = ['jpg', 'jpeg', 'jfif', 'png', 'bmp', 'webp', 'tiff']
const VIDEO_EXT = ['mp4', 'mov', 'avi', 'mkv', 'webm']

/** Downscaled data URL, so stored reports stay a sensible size. */
function previewOf(bitmap, max = 480) {
  const scale = Math.min(1, max / Math.max(bitmap.width, bitmap.height))
  const c = document.createElement('canvas')
  c.width = Math.round(bitmap.width * scale)
  c.height = Math.round(bitmap.height * scale)
  c.getContext('2d').drawImage(bitmap, 0, 0, c.width, c.height)
  return c.toDataURL('image/jpeg', 0.82)
}

function buildFindings({ metadata, c2pa, ela }) {
  const findings = [{
    severity: 'medium',
    text: 'No neural verdict — the deepfake classifiers are not loaded in this build',
  }]

  if (metadata.metadata_stripped) {
    findings.push({ severity: 'medium', text: 'Metadata missing or removed' })
  } else if (metadata.camera_make) {
    findings.push({
      severity: 'info',
      text: `Camera metadata present (${metadata.camera_make} ${metadata.camera_model ?? ''})`.trim(),
    })
  }

  if (metadata.software) {
    findings.push({ severity: 'medium', text: `Processed by editing software: ${metadata.software}` })
  }
  if (metadata.has_gps) {
    findings.push({ severity: 'info', text: 'GPS coordinates present in metadata' })
  }

  findings.push(c2pa.signature_found
    ? { severity: 'info', text: 'C2PA content credential present' }
    : { severity: 'medium', text: 'No C2PA signature found' })

  if (ela?.suspicious_regions) {
    findings.push({
      severity: 'high',
      text: 'Error Level Analysis shows inconsistent compression regions',
    })
  } else if (ela && !ela.error) {
    findings.push({ severity: 'info', text: 'Compression response is uniform across the frame' })
  }

  return findings
}

/** Analyse an image entirely in the browser. */
export async function analyzeImage(file) {
  const started = performance.now()
  const timeline = []
  const mark = (stage) =>
    timeline.push({ stage, elapsed_ms: Number((performance.now() - started).toFixed(1)) })

  const buffer = await file.arrayBuffer()
  mark('File loaded')

  let bitmap
  try {
    bitmap = await createImageBitmap(new Blob([buffer], { type: file.type || 'image/jpeg' }))
  } catch {
    throw new Error(`Could not decode image: ${file.name}`)
  }
  mark('Image decoded')

  const metadata = readExif(buffer)
  metadata.dimensions = `${bitmap.width}x${bitmap.height}`
  metadata.format = (file.name.split('.').pop() || '').toUpperCase()
  mark('Metadata extraction')

  const c2pa = checkC2PA(buffer)
  mark('Provenance check')

  let ela
  try {
    ela = await errorLevelAnalysis(bitmap)
  } catch (err) {
    ela = { error: String(err?.message ?? err) }
  }
  mark('Error level analysis')

  const preview = previewOf(bitmap)
  const findings = buildFindings({ metadata, c2pa, ela })
  mark('Report generated')

  const report = {
    scan_id: uid(),
    created_at: nowISO(),
    media_type: 'image',
    filename: file.name,
    file_size_bytes: file.size,
    dimensions: `${bitmap.width}x${bitmap.height}`,
    width: bitmap.width,
    height: bitmap.height,

    // No classifier: these stay null rather than being invented.
    fake_probability: null,
    authenticity_score: null,
    confidence: null,
    verdict: 'UNVERIFIED',
    risk_level: 'UNKNOWN',

    engine: 'browser',
    engine_note:
      'Analysed in your browser. Metadata, compression and provenance checks are real; '
      + 'the neural deepfake verdict requires the trained models, which are not part of this build.',

    faces_detected: 0,
    faces: [{
      face_id: 1,
      is_full_frame: true,
      bbox: [0, 0, bitmap.width, bitmap.height],
      detection_score: null,
      fake_probability: null,
      verdict: 'UNVERIFIED',
      confidence: null,
      model_agreement: null,
      has_embedding: false,
      crop_preview: preview,
      models: [],
    }],
    models: [],

    metadata,
    c2pa,
    ela,
    findings,
    timeline,
    processing_ms: Number((performance.now() - started).toFixed(1)),
  }

  bitmap.close?.()
  await store.saveScan(report)
  return report
}

/**
 * Video: metadata and provenance only.
 *
 * Frames could be sampled through a <video> element, but with no classifier to
 * score them it would be work without a result, so it is not pretended.
 */
export async function analyzeVideo(file) {
  const started = performance.now()
  const buffer = await file.arrayBuffer()

  const c2pa = checkC2PA(buffer)
  const metadata = {
    has_exif: false, metadata_stripped: true,
    camera_make: null, camera_model: null, software: null,
    datetime: null, has_gps: false, fields: {},
  }

  // Read intrinsic properties from a detached video element.
  const meta = await new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const v = document.createElement('video')
    v.preload = 'metadata'
    v.onloadedmetadata = () => {
      resolve({ duration: v.duration, width: v.videoWidth, height: v.videoHeight })
      URL.revokeObjectURL(url)
    }
    v.onerror = () => { resolve({ duration: 0, width: 0, height: 0 }); URL.revokeObjectURL(url) }
    v.src = url
  })

  const report = {
    scan_id: uid(),
    created_at: nowISO(),
    media_type: 'video',
    filename: file.name,
    file_size_bytes: file.size,
    dimensions: `${meta.width}x${meta.height}`,
    duration_sec: Number((meta.duration || 0).toFixed(2)),
    fps: null,
    total_frames: null,
    frames_analyzed: 0,
    frames_with_faces: 0,

    fake_probability: null,
    authenticity_score: null,
    confidence: null,
    verdict: 'UNVERIFIED',
    risk_level: 'UNKNOWN',

    engine: 'browser',
    engine_note:
      'Video frame analysis needs the trained models and the local Python service. '
      + 'Only container-level checks ran here.',

    people_detected: 0,
    tracks: [],
    frame_scores: [],
    most_suspicious_frame: null,
    metadata,
    c2pa,
    ela: null,
    findings: [
      { severity: 'medium', text: 'No neural verdict — video analysis requires the local service' },
      c2pa.signature_found
        ? { severity: 'info', text: 'C2PA content credential present' }
        : { severity: 'medium', text: 'No C2PA signature found' },
    ],
    timeline: [{ stage: 'Container inspected', elapsed_ms: Number((performance.now() - started).toFixed(1)) }],
    processing_ms: Number((performance.now() - started).toFixed(1)),
  }

  await store.saveScan(report)
  return report
}

export async function analyze(file) {
  const ext = (file.name.split('.').pop() || '').toLowerCase()
  if (IMAGE_EXT.includes(ext)) return analyzeImage(file)
  if (VIDEO_EXT.includes(ext)) return analyzeVideo(file)
  throw new Error(
    `Unsupported file type ".${ext}". Images: ${IMAGE_EXT.join(', ')} · Videos: ${VIDEO_EXT.join(', ')}`
  )
}

/** Mirrors /api/system/info so the System page renders unchanged. */
export function systemInfo() {
  return {
    detector: {
      ready: false,
      model_count: 0,
      models: [],
      trained_on: null,
      test_set_size: null,
      ensemble_metrics: {},
    },
    ai_detector: {
      ready: false,
      model_count: 0,
      models: [],
      trained_on: null,
      test_set_size: null,
      ensemble_metrics: {},
    },
    face_analyzer_ready: false,
    engine: 'browser',
    thresholds: { suspicious: 0.4, fake: 0.65 },
    limits: { max_upload_mb: 200, video_max_frames: 32 },
    supported: {
      image: IMAGE_EXT.map((e) => `.${e}`),
      video: VIDEO_EXT.map((e) => `.${e}`),
    },
  }
}

export const health = () => ({
  status: 'ok',
  engine: 'browser',
  models_loaded: false,
  model_count: 0,
  face_analyzer: false,
})

export { store }

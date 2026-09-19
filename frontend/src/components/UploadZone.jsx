import { useRef, useState } from 'react'
import { api, formatBytes } from '../api.js'

const IMAGE_EXT = ['.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.webp', '.tiff']
const VIDEO_EXT = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

/**
 * Drag-and-drop upload with client-side extension checking.
 *
 * The type check happens here as well as on the server so an unsupported file
 * fails instantly instead of after a slow upload.
 */
export default function UploadZone({ onComplete, compact = false, disabled = false }) {
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [file, setFile] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const submit = async (picked) => {
    setError(null)

    const ext = `.${picked.name.split('.').pop().toLowerCase()}`
    if (![...IMAGE_EXT, ...VIDEO_EXT].includes(ext)) {
      setError(`Unsupported file type "${ext}". Supported: ${[...IMAGE_EXT, ...VIDEO_EXT].join(', ')}`)
      return
    }

    setFile(picked)
    setBusy(true)
    try {
      onComplete(await api.analyze(picked))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    if (disabled) return
    const picked = e.dataTransfer.files?.[0]
    if (picked) submit(picked)
  }

  const isVideo = file && VIDEO_EXT.includes(`.${file.name.split('.').pop().toLowerCase()}`)

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !disabled && !busy && inputRef.current?.click()}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && !busy && inputRef.current?.click()}
        aria-disabled={disabled}
        className={`rounded-xl border-2 border-dashed flex flex-col items-center justify-center text-center transition-colors ${compact ? 'py-9 px-5' : 'py-16 px-6'}`}
        style={{
          borderColor: dragging ? 'var(--brand)' : 'var(--border-bright)',
          background: dragging ? 'color-mix(in srgb, var(--brand) 10%, transparent)' : 'var(--surface-2)',
          cursor: disabled ? 'not-allowed' : busy ? 'progress' : 'pointer',
          opacity: disabled ? 0.55 : 1,
        }}
      >
        {busy ? (
          <>
            <span className="spin w-7 h-7 rounded-full border-2 border-t-transparent mb-3"
                  style={{ borderColor: 'var(--brand)', borderTopColor: 'transparent' }} />
            <p className="font-medium text-sm">Analysing {file?.name}…</p>
            <p className="text-xs mt-1" style={{ color: 'var(--ink-muted)' }}>
              {isVideo ? 'Sampling frames and tracking faces — this can take a minute.' : 'Running the model ensemble.'}
            </p>
          </>
        ) : (
          <>
            <span className="text-3xl mb-2.5" aria-hidden="true" style={{ color: 'var(--brand)' }}>⇪</span>
            <p className="font-medium">
              {disabled ? 'Detection unavailable — no models loaded' : 'Drag & drop your file here'}
            </p>
            <p className="text-xs mt-1.5 max-w-sm" style={{ color: 'var(--ink-muted)' }}>
              Images: JPG, JPEG, JFIF, PNG, WEBP, BMP, TIFF · Videos: MP4, MOV, AVI, MKV, WEBM
            </p>
            {!disabled && (
              <span className="mt-4 px-5 py-2 rounded-lg text-sm font-medium"
                    style={{ background: 'linear-gradient(135deg, var(--brand), var(--brand-2))', color: 'var(--on-accent)' }}>
                Choose File
              </span>
            )}
          </>
        )}

        <input ref={inputRef} type="file" className="hidden"
               accept={[...IMAGE_EXT, ...VIDEO_EXT].join(',')}
               onChange={(e) => e.target.files?.[0] && submit(e.target.files[0])} />
      </div>

      {file && !busy && !error && (
        <p className="text-xs mt-2.5" style={{ color: 'var(--ink-muted)' }}>
          {file.name} · {formatBytes(file.size)}
        </p>
      )}

      {error && (
        <div className="mt-3 px-3.5 py-2.5 rounded-lg text-[13px]"
             style={{
               background: 'color-mix(in srgb, var(--critical) 12%, transparent)',
               border: '1px solid color-mix(in srgb, var(--critical) 34%, transparent)',
               color: 'var(--ink-2)',
             }}>
          <strong style={{ color: 'var(--critical)' }}>✕ Scan failed — </strong>{error}
        </div>
      )}
    </div>
  )
}

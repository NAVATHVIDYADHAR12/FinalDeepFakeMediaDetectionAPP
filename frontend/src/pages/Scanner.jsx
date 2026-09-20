import { useNavigate, useSearchParams } from 'react-router-dom'
import { StandaloneNotice, ServiceUnavailableNotice, ModelsMissing, Panel } from '../components/ui.jsx'
import UploadZone from '../components/UploadZone.jsx'

const COPY = {
  image: {
    title: 'Image Verification',
    body: 'Detects face swaps, GAN-generated faces and edited regions. Every face is located, cropped and scored independently, then the model shows which pixels drove its decision.',
  },
  video: {
    title: 'Video Verification',
    body: 'Samples frames across the clip, tracks each person by face embedding, and scores them over time. A genuine face scores steadily; a manipulated one flickers.',
  },
  all: {
    title: 'Media Scanner',
    body: 'Upload any supported image or video. The file is routed to the right pipeline automatically.',
  },
}

export default function Scanner({ health }) {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const copy = COPY[params.get('type')] ?? COPY.all

  return (
    <div className="max-w-3xl mx-auto space-y-5">
      {health?.service_unavailable ? <ServiceUnavailableNotice detail={health.error} />
        : health?.engine === 'browser' ? <StandaloneNotice />
        : health && !health.models_loaded ? <ModelsMissing /> : null}

      <Panel title={copy.title}>
        <p className="text-[13px] -mt-1 mb-5 leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
          {copy.body}
        </p>
        <UploadZone disabled={Boolean(health) && !health.models_loaded && health.engine !== 'browser'}
                    onComplete={(report) => navigate(`/report/${report.scan_id}`)} />
      </Panel>

      <Panel title="What gets checked">
        <ul className="grid sm:grid-cols-2 gap-x-6 gap-y-2.5 text-[13px]" style={{ color: 'var(--ink-2)' }}>
          {[
            'Face detection and per-face scoring',
            'CNN ensemble vote across trained models',
            'Class-activation heatmap (why it decided that)',
            'EXIF metadata and editing-software traces',
            'C2PA content-credential presence',
            'Error Level Analysis for spliced regions',
            'Identity embedding extraction',
            'Temporal consistency (video only)',
          ].map((item) => (
            <li key={item} className="flex gap-2">
              <span style={{ color: 'var(--brand)' }} aria-hidden="true">▸</span>{item}
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  )
}

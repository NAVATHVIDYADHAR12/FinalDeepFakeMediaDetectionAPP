import { Fragment, useLayoutEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

import HeroCanvas from '../components/HeroCanvas.jsx'
import Reveal from '../components/Reveal.jsx'
import Footer from '../components/Footer.jsx'
import Marquee from '../components/Marquee.jsx'
import TiltCard from '../components/TiltCard.jsx'
import { useScroller } from '../ScrollContext.js'
import { prefersReducedMotion, useCountUp, useInView } from '../hooks.js'

gsap.registerPlugin(ScrollTrigger)

/* Section ids are the anchor targets for the nav bar's Features dropdown. */
export const SECTIONS = [
  { id: 'detect',   label: 'Detection Engine' },
  { id: 'explain',  label: 'Explainability' },
  { id: 'identity', label: 'Face Recognition' },
  { id: 'forensics', label: 'Forensic Signals' },
  { id: 'pipeline', label: 'How It Works' },
  { id: 'honesty',  label: 'What It Cannot Do' },
]

/* Ticker copy. Accented items are the claims worth catching mid-scroll. */
const TICKER = [
  { text: 'DEEPFAKE DETECTION', accent: true },
  { text: 'Face swap · Face reenactment · GAN-generated faces' },
  { text: 'THREE-MODEL CNN ENSEMBLE', accent: true },
  { text: 'EfficientNet-B0 · XceptionNet · MobileNetV3' },
  { text: 'TRAINED ON FACEFORENSICS++', accent: true },
  { text: '190,000 face crops · four forgery methods' },
  { text: 'CLASS ACTIVATION HEATMAPS', accent: true },
  { text: 'See exactly which pixels drove the verdict' },
  { text: 'PER-PERSON VIDEO TRACKING', accent: true },
  { text: 'Temporal consistency across sampled frames' },
  { text: 'FACE RECOGNITION', accent: true },
  { text: '128-dimension embeddings · enrolled identity gallery' },
  { text: 'FORENSIC CORROBORATION', accent: true },
  { text: 'EXIF metadata · Error Level Analysis · C2PA provenance' },
  { text: 'RUNS ENTIRELY ON YOUR MACHINE', accent: true },
  { text: 'No upload · no account required · no cloud calls' },
  { text: '~200 MILLISECONDS PER IMAGE', accent: true },
  { text: 'ONNX Runtime on CPU · no GPU needed' },
  { text: 'A VERDICT IS EVIDENCE, NOT PROOF', accent: true },
]

/* Three rows for the band above the footer. Kept separate so the stack never
   shows the same phrase twice across rows at the same moment. */
const BAND_ROWS = [
  [
    { text: 'DEEPFAKE DETECTION', accent: true }, { text: 'Face swap' },
    { text: 'FACE REENACTMENT', accent: true }, { text: 'GAN-generated faces' },
    { text: 'NEURALTEXTURES', accent: true }, { text: 'Face2Face' },
    { text: 'FACESWAP', accent: true }, { text: 'Synthetic identities' },
    { text: 'AI-GENERATED MEDIA', accent: true }, { text: 'Manipulated video' },
  ],
  [
    { text: 'EFFICIENTNET-B0', accent: true }, { text: 'XceptionNet' },
    { text: 'MOBILENETV3', accent: true }, { text: 'Model ensemble' },
    { text: 'ONNX RUNTIME', accent: true }, { text: 'CPU inference' },
    { text: 'CLASS ACTIVATION MAPS', accent: true }, { text: 'Explainable verdicts' },
    { text: 'FACEFORENSICS++', accent: true }, { text: '190,000 training images' },
  ],
  [
    { text: 'EXIF METADATA', accent: true }, { text: 'Error Level Analysis' },
    { text: 'C2PA PROVENANCE', accent: true }, { text: 'Compression forensics' },
    { text: 'FACE EMBEDDINGS', accent: true }, { text: 'Identity matching' },
    { text: 'TEMPORAL CONSISTENCY', accent: true }, { text: 'Per-person tracking' },
    { text: 'RUNS LOCALLY', accent: true }, { text: 'No cloud, no account' },
  ],
]

const CAPABILITIES = [
  {
    icon: '◈', accent: 'var(--brand)',
    title: 'CNN Ensemble',
    body: 'Three independently fine-tuned networks — EfficientNet-B0, XceptionNet and MobileNetV3 — vote on every face. They fail on different images, so the ensemble beats any single model.',
  },
  {
    icon: '◉', accent: 'var(--good)',
    title: 'Per-Face Analysis',
    body: 'Every face is located, cropped with a 20% margin and scored on its own. One forged face in a group photo makes the whole image manipulated.',
  },
  {
    icon: '▤', accent: 'var(--warning)',
    title: 'Activation Heatmaps',
    body: 'A class activation map shows exactly which pixels pushed the score toward "fake" — computed from the same forward pass as the prediction, not an approximation.',
  },
  {
    icon: '⧗', accent: 'var(--brand)',
    title: 'Temporal Analysis',
    body: 'A real face scores consistently across video frames. A swapped one flickers as the generator struggles with pose and occlusion. That variance is itself the evidence.',
  },
  {
    icon: '☺', accent: 'var(--good)',
    title: 'Identity Matching',
    body: '128-dimension face embeddings match people against an enrolled gallery, so the report can say "this claims to be X, and the face is manipulated".',
  },
  {
    icon: '⚿', accent: 'var(--warning)',
    title: 'Forensic Corroboration',
    body: 'EXIF metadata, Error Level Analysis and C2PA provenance run independently of the neural network — cheap checks that catch what the model cannot see.',
  },
]

const PIPELINE = [
  { n: '01', title: 'Upload',      body: 'Drop an image or video. The file is routed to the right pipeline automatically.' },
  { n: '02', title: 'Detect',      body: 'YuNet locates every face and five landmarks in milliseconds, on CPU.' },
  { n: '03', title: 'Classify',    body: 'Each crop passes through the model ensemble; the votes are averaged.' },
  { n: '04', title: 'Explain',     body: 'A heatmap is computed showing which regions drove the decision.' },
  { n: '05', title: 'Corroborate', body: 'Metadata, compression and provenance checks run alongside the model.' },
  { n: '06', title: 'Report',      body: 'A verdict, a confidence, and the evidence behind both.' },
]

const STATS = [
  { value: 190, suffix: 'k', label: 'Training images', detail: 'FaceForensics++ crops' },
  { value: 3, suffix: '', label: 'Neural networks', detail: 'One calibrated ensemble' },
  { value: 200, prefix: '~', suffix: 'ms', label: 'Per image, on CPU', detail: 'No GPU required' },
  { value: 0, suffix: '', label: 'Cloud calls', detail: 'Your media stays private' },
]

function AnimatedStat({ stat, index }) {
  const [ref, inView] = useInView({ threshold: 0.45 })
  const count = useCountUp(stat.value, { duration: 1150 + index * 150, start: inView })
  return (
    <div ref={ref} className="stat-prism panel p-5 text-center lift" style={{ '--stat-i': index }}>
      <div className="stat-prism__glow" aria-hidden="true" />
      <div className="figure text-3xl mb-1 relative" style={{ color: 'var(--brand)' }}>
        {stat.prefix}{Math.round(count)}{stat.suffix}
      </div>
      <div className="text-[12px] relative" style={{ color: 'var(--ink-2)' }}>{stat.label}</div>
      <div className="text-[9px] tracking-[.12em] uppercase mt-2 relative"
           style={{ color: 'var(--ink-muted)' }}>{stat.detail}</div>
    </div>
  )
}

/**
 * Hero copy over the video.
 *
 * One animation only: an entrance timeline that runs on load. The words rise
 * out of their masks, the subheading follows, then each button reveals on its
 * own, then the scroll cue. After that the copy simply stays put over the
 * footage for the whole hero — it is not animated away again.
 *
 * Sizes are capped and spacing tightened so the block fits inside one viewport
 * height. The earlier version overflowed on laptop screens, which is why the
 * lower half of the copy was cut off.
 */
function HeroHeadline({ scroller, triggerRef }) {
  const innerRef = useRef(null)

  useLayoutEffect(() => {
    const node = innerRef.current
    if (!node || prefersReducedMotion()) return

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ delay: 0.25 })

      tl.from('.hero-word > span', {
        yPercent: 115, opacity: 0, duration: 0.95,
        ease: 'power4.out', stagger: 0.08,
      })
        // Buttons arrive one at a time rather than as a pair.
        .from('.hero-cta > *', {
          y: 26, opacity: 0, scale: 0.94, duration: 0.65,
          ease: 'back.out(1.6)', stagger: 0.22,
        }, '-=0.30')
        .from('.hero-scroll', {
          y: 12, opacity: 0, duration: 0.7, ease: 'power2.out',
        }, '-=0.15')
    }, node)

    return () => ctx.revert()
  }, [])

  // The strapline is deliberately NOT part of the load timeline. It reveals
  // near the end of the frame scrub, so it lands as the footage finishes
  // rather than competing with the headline for attention at second zero.
  useLayoutEffect(() => {
    const node = innerRef.current
    const trigger = triggerRef?.current
    if (!node || !trigger || prefersReducedMotion()) return

    const ctx = gsap.context(() => {
      gsap.set('.hero-sub', { opacity: 0, y: 26 })
      gsap.to('.hero-sub', {
        opacity: 1, y: 0, duration: 1.0, ease: 'power3.out',
        scrollTrigger: {
          trigger,
          scroller: scroller || undefined,
          // ~65% through the hero's scroll distance: the footage is nearly
          // finished, but there is still runway before the section leaves.
          start: '65% top',
          toggleActions: 'play none none reverse',
        },
      })
    }, node)

    return () => ctx.revert()
  }, [scroller, triggerRef])

  const line1 = ['See', 'beyond']
  const line2 = ['the', 'real.']

  return (
    <div className="relative z-10 w-full">
      <div ref={innerRef} className="text-center px-6 max-w-4xl mx-auto">
        <h1 className="font-display font-bold leading-[1.02] tracking-tight"
            style={{ fontSize: 'clamp(2.1rem, 6.2vw, 4.4rem)' }}>
          {[line1, line2].map((line, li) => (
            <span key={li} className="block">
              {line.map((word, wi) => (
                // Each word is a masked box and the inner span slides up out of
                // it. The padding keeps the mask off descenders and the glow.
                //
                // The space is a real text node BETWEEN the masks. Putting it
                // inside failed twice over: an inline-block trims its own
                // trailing whitespace, and overflow:hidden clips whatever
                // survives — which is why the words ran together as
                // "Seebeyond". A margin would fix the look but leave no space
                // in the DOM, so the heading would still be read aloud and
                // copied as one word.
                <Fragment key={wi}>
                  <span className="hero-word inline-block overflow-hidden align-bottom"
                        style={{ paddingBottom: '0.14em', marginBottom: '-0.14em' }}>
                    <span className={`inline-block ${li === 1 ? 'headline-animated' : ''}`}>
                      {word}
                    </span>
                  </span>
                  {wi < line.length - 1 && ' '}
                </Fragment>
              ))}
            </span>
          ))}
        </h1>

        <p className="hero-sub mt-5 text-[14px] sm:text-base max-w-lg mx-auto leading-relaxed"
           style={{ color: 'var(--ink-2)' }}>
          Deepfake and AI-generated media detection that shows its working —
          real trained models, running locally, explaining every verdict.
        </p>

        <div className="hero-cta mt-7 flex flex-wrap gap-3 justify-center">
          <Link to="/scan" className="px-6 py-2.5 rounded-full text-sm font-semibold press"
                style={{
                  background: 'linear-gradient(135deg, var(--brand), var(--brand-2))',
                  color: 'var(--on-accent)',
                  boxShadow: '0 0 34px -8px rgba(0,240,255,.75)',
                }}>
            Analyse a file
          </Link>
          <Link to="/dashboard" className="px-6 py-2.5 rounded-full text-sm font-semibold press"
                style={{
                  background: 'rgba(226,232,240,.07)',
                  border: '1px solid rgba(226,232,240,.2)',
                  color: 'var(--ink)',
                  backdropFilter: 'blur(10px)',
                }}>
            Open dashboard
          </Link>
        </div>

        <div className="hero-scroll mt-8 flex flex-col items-center gap-1.5"
             style={{ color: 'var(--ink-muted)' }}>
          <span className="text-[10px] tracking-[0.3em] font-display">SCROLL</span>
          <span className="w-px h-7 block"
                style={{ background: 'linear-gradient(to bottom, var(--brand), transparent)' }} />
        </div>
      </div>
    </div>
  )
}


function SectionHeading({ eyebrow, title, body, align = 'left' }) {
  return (
    <Reveal from={align === 'right' ? 'right' : 'left'}
            className={align === 'right' ? 'text-right ml-auto' : 'text-left'}>
      <div className="max-w-xl">
        <div className="text-[11px] tracking-[0.24em] font-display mb-3"
             style={{ color: 'var(--brand)' }}>
          {eyebrow}
        </div>
        <h2 className="font-display font-bold leading-tight mb-4"
            style={{ fontSize: 'clamp(1.7rem, 3.6vw, 2.7rem)' }}>
          {title}
        </h2>
        <p className="text-[15px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>
          {body}
        </p>
      </div>
    </Reveal>
  )
}

/** Animated schematic: a face being scanned, drawn in SVG. */
function ScanVisual() {
  return (
    <svg viewBox="0 0 320 220" className="w-full h-auto" role="img"
         aria-label="Schematic of a face being scanned and scored">
      <defs>
        <linearGradient id="scanline" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="var(--brand)" stopOpacity="0" />
          <stop offset="50%" stopColor="var(--brand)" stopOpacity="1" />
          <stop offset="100%" stopColor="var(--brand)" stopOpacity="0" />
        </linearGradient>
      </defs>

      <rect x="0.5" y="0.5" width="319" height="219" rx="12"
            fill="rgba(11,19,43,.55)" stroke="rgba(226,232,240,.10)" />

      {/* face bounding box + landmarks */}
      <rect x="108" y="46" width="104" height="128" rx="8"
            fill="none" stroke="var(--brand)" strokeWidth="1.4" strokeDasharray="5 4"
            opacity="0.85">
        <animate attributeName="stroke-dashoffset" from="0" to="18" dur="1.6s" repeatCount="indefinite" />
      </rect>
      {[[132, 92], [188, 92], [160, 118], [138, 146], [182, 146]].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="3" fill="var(--brand)">
          <animate attributeName="opacity" values="0.35;1;0.35" dur="2s"
                   begin={`${i * 0.18}s`} repeatCount="indefinite" />
        </circle>
      ))}

      {/* corner brackets */}
      {[[108, 46, 1, 1], [212, 46, -1, 1], [108, 174, 1, -1], [212, 174, -1, -1]].map(([x, y, sx, sy], i) => (
        <path key={i} d={`M ${x} ${y + 16 * sy} L ${x} ${y} L ${x + 16 * sx} ${y}`}
              fill="none" stroke="var(--good)" strokeWidth="2" strokeLinecap="round" />
      ))}

      {/* sweeping scan line */}
      <rect x="108" width="104" height="2" fill="url(#scanline)">
        <animate attributeName="y" values="46;172;46" dur="3.4s" repeatCount="indefinite" />
      </rect>

      {/* verdict readout */}
      <text x="238" y="70" fill="var(--ink-muted)" fontSize="8" fontFamily="monospace">MODELS</text>
      {['EfficientNet', 'Xception', 'MobileNet'].map((m, i) => (
        <g key={m}>
          <text x="238" y={86 + i * 20} fill="var(--ink-2)" fontSize="7.5" fontFamily="monospace">{m}</text>
          <rect x="238" y={90 + i * 20} width="60" height="3" rx="1.5" fill="rgba(226,232,240,.12)" />
          <rect x="238" y={90 + i * 20} width="0" height="3" rx="1.5" fill="var(--critical)">
            <animate attributeName="width" values="0;54;54" dur="2.6s"
                     begin={`${i * 0.3}s`} repeatCount="indefinite" />
          </rect>
        </g>
      ))}
      <text x="238" y="166" fill="var(--critical)" fontSize="11" fontWeight="bold"
            fontFamily="monospace">FAKE</text>

      <text x="22" y="70" fill="var(--ink-muted)" fontSize="8" fontFamily="monospace">INPUT</text>
      {[0, 1, 2].map((i) => (
        <rect key={i} x="22" y={82 + i * 26} width="62" height="18" rx="3"
              fill="rgba(226,232,240,.05)" stroke="rgba(226,232,240,.10)">
          <animate attributeName="opacity" values="0.3;1;0.3" dur="2.4s"
                   begin={`${i * 0.4}s`} repeatCount="indefinite" />
        </rect>
      ))}
    </svg>
  )
}

/** Animated schematic: the heatmap explaining a verdict. */
function HeatmapVisual() {
  const cells = []
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      // hot in the middle-lower region, where blending artefacts concentrate
      const d = Math.hypot(x - 3.5, y - 4.5) / 5.5
      cells.push({ x, y, heat: Math.max(0, 1 - d) })
    }
  }

  return (
    <svg viewBox="0 0 320 220" className="w-full h-auto" role="img"
         aria-label="Schematic of a class activation heatmap over a face">
      <rect x="0.5" y="0.5" width="319" height="219" rx="12"
            fill="rgba(11,19,43,.55)" stroke="rgba(226,232,240,.10)" />

      <g transform="translate(96, 30)">
        {cells.map((c, i) => (
          <rect key={i} x={c.x * 16} y={c.y * 16} width="15" height="15" rx="2"
                fill={c.heat > 0.62 ? 'var(--critical)'
                    : c.heat > 0.42 ? 'var(--warning)'
                    : c.heat > 0.22 ? 'var(--brand)' : 'rgba(0,240,255,.10)'}
                opacity={0.18 + c.heat * 0.72}>
            <animate attributeName="opacity"
                     values={`${0.1 + c.heat * 0.3};${0.2 + c.heat * 0.75};${0.1 + c.heat * 0.3}`}
                     dur="3s" begin={`${(c.x + c.y) * 0.06}s`} repeatCount="indefinite" />
          </rect>
        ))}
      </g>

      <text x="22" y="44" fill="var(--ink-muted)" fontSize="8" fontFamily="monospace">CAM</text>
      <text x="22" y="60" fill="var(--ink-2)" fontSize="7" fontFamily="monospace">Σ W·features</text>

      <g transform="translate(22, 150)">
        {[['var(--critical)', 'high'], ['var(--warning)', 'mid'], ['var(--brand)', 'low']].map(([c, l], i) => (
          <g key={l} transform={`translate(0, ${i * 16})`}>
            <rect width="9" height="9" rx="2" fill={c} />
            <text x="14" y="8" fill="var(--ink-muted)" fontSize="7" fontFamily="monospace">{l}</text>
          </g>
        ))}
      </g>
    </svg>
  )
}

export default function Landing() {
  const scroller = useScroller()
  const [, setHeroReady] = useState(false)
  const heroRef = useRef(null)

  // ScrollTrigger measures on creation; the hero is 200vh, so positions shift
  // once it mounts. A refresh after layout settles keeps every trigger honest.
  useLayoutEffect(() => {
    const t = setTimeout(() => ScrollTrigger.refresh(), 350)
    return () => clearTimeout(t)
  }, [scroller])

  return (
    <div className="-mx-6">
      {/* ---------------------------------------------------------- hero --- */}
      {/* Two viewport heights: the first is the sticky stage, the second is the
          scroll distance that drives the frame scrub. */}
      <section ref={heroRef} className="relative" style={{ height: '200vh' }}>
        <div className="sticky top-0 h-screen overflow-hidden flex items-center justify-center">
          <HeroCanvas onReady={() => setHeroReady(true)} />
          <HeroHeadline scroller={scroller} triggerRef={heroRef} />
        </div>
      </section>

      {/* ------------------------------------------------------- ticker --- */}
      <section aria-label="What OmniGuard does"
               className="border-y relative"
               style={{
                 borderColor: 'var(--border)',
                 background: 'rgba(11,19,43,.45)',
                 backdropFilter: 'blur(10px)',
               }}>
        <div className="px-6"><Marquee items={TICKER} /></div>
      </section>

      {/* --------------------------------------------------------- stats --- */}
      <section className="px-6 py-16 relative">
        <Reveal stagger={0.22} from="up"
                className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
          {STATS.map((s) => (
            <AnimatedStat key={s.label} stat={s} index={STATS.indexOf(s)} />
          ))}
        </Reveal>
      </section>

      {/* ------------------------------------------------------ detection --- */}
      <section id="detect" className="px-6 py-20 scroll-mt-28">
        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-14 items-center">
          <SectionHeading
            eyebrow="DETECTION ENGINE"
            title="Three networks, one verdict"
            body="Every face is scored by three independently trained convolutional networks. Because different architectures fail on different images, averaging their votes is measurably stronger than trusting any one of them — and when they disagree, the report says so instead of hiding it behind a confident number."
          />
          <Reveal from="right" delay={0.1}>
            <TiltCard className="panel p-4 ensemble-stage">
              <div className="ensemble-stage__orbit" aria-hidden="true">
                <span>E</span><span>X</span><span>M</span>
              </div>
              <ScanVisual />
              <div className="ensemble-stage__caption">
                <span className="glow-pulse" /> Consensus engine · live inference
              </div>
            </TiltCard>
          </Reveal>
        </div>
      </section>

      {/* ---------------------------------------------------- capabilities --- */}
      <section className="px-6 py-20">
        <div className="max-w-6xl mx-auto">
          <Reveal from="up" className="text-center mb-14">
            <div className="text-[11px] tracking-[0.24em] font-display mb-3"
                 style={{ color: 'var(--brand)' }}>CAPABILITIES</div>
            <h2 className="font-display font-bold headline"
                style={{ fontSize: 'clamp(1.8rem, 4vw, 3rem)' }}>
              Everything the report is built from
            </h2>
          </Reveal>

          <Reveal stagger={0.22} from="scale"
                  className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {CAPABILITIES.map((c, i) => (
              <TiltCard key={c.title} className="h-full report-card-wrap">
              <article className="panel p-6 h-full report-capability" style={{ '--card-accent': c.accent }}>
                <div className="report-capability__grid" aria-hidden="true" />
                {/* SOS beacon: currentColor drives the glow, so each icon
                    signals in its own accent. --i offsets the phase so the
                    grid blinks in sequence rather than in unison. */}
                <div className="w-11 h-11 rounded-xl flex items-center justify-center text-lg mb-4 sos-glow"
                     style={{
                       background: `color-mix(in srgb, ${c.accent} 16%, transparent)`,
                       color: c.accent,
                       '--i': i,
                     }}
                     aria-hidden="true">{c.icon}</div>
                <h3 className="font-display font-semibold text-[16px] mb-2">{c.title}</h3>
                <p className="text-[13.5px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>
                  {c.body}
                </p>
                <div className="report-capability__index figure" aria-hidden="true">0{i + 1}</div>
              </article>
              </TiltCard>
            ))}
          </Reveal>
        </div>
      </section>

      {/* --------------------------------------------------- explainability --- */}
      <section id="explain" className="px-6 py-20 scroll-mt-28">
        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-14 items-center">
          <Reveal from="left" className="order-2 lg:order-1">
            <TiltCard className="panel p-4"><HeatmapVisual /></TiltCard>
          </Reveal>
          <div className="order-1 lg:order-2">
            <SectionHeading
              align="right"
              eyebrow="EXPLAINABILITY"
              title="It shows you why"
              body='A verdict you cannot interrogate is not evidence. Every result carries a class activation map — the exact spatial breakdown of the "fake" score, computed from the same forward pass as the prediction. Red is where the network objected. On a real face swap, that heat lands on the jawline and hairline, precisely where blending is hardest.'
            />
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------- identity --- */}
      <section id="identity" className="px-6 py-20 scroll-mt-28">
        <div className="max-w-6xl mx-auto">
          <SectionHeading
            eyebrow="FACE RECOGNITION"
            title="Who, and whether it is really them"
            body="Faces are reduced to 128-dimension embeddings and matched against a gallery you enrol. Identity alone is not that useful, and manipulation alone is not that useful — together they let the report make the statement that actually matters: this claims to be a specific person, and the face has been altered."
          />
          <Reveal stagger={0.22} from="up" className="grid sm:grid-cols-3 gap-5 mt-12">
            {[
              ['Enrol', 'Only the 128-number vector is stored. The photograph is discarded immediately.'],
              ['Match', 'Cosine similarity against every enrolled identity, at OpenCV’s recommended threshold.'],
              ['Track', 'The same embedding follows a person across video frames, giving each their own timeline.'],
            ].map(([t, b]) => (
              <div key={t} className="panel p-6 lift">
                <h3 className="font-display font-semibold mb-2" style={{ color: 'var(--good)' }}>{t}</h3>
                <p className="text-[13.5px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>{b}</p>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      {/* ------------------------------------------------------- forensics --- */}
      <section id="forensics" className="px-6 py-20 scroll-mt-28">
        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-14 items-start">
          <SectionHeading
            eyebrow="FORENSIC SIGNALS"
            title="Checks the model cannot make"
            body="Neural networks are not the only evidence. Three classical checks run alongside them, entirely independently — which is what makes them worth having."
          />
          <Reveal stagger={0.22} from="right" className="space-y-4">
            {[
              ['EXIF metadata', 'Cameras write rich capture data. Generative models write none, and most editors destroy it.'],
              ['Error Level Analysis', 'Re-compresses the image and measures which regions move. A spliced area has a different compression history.'],
              ['C2PA provenance', 'Looks for a signed content credential. Presence is meaningful; absence mostly is not, and the report says so.'],
            ].map(([t, b]) => (
              <div key={t} className="panel p-5 lift">
                <h3 className="font-display font-semibold text-[15px] mb-1.5">{t}</h3>
                <p className="text-[13px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>{b}</p>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      {/* -------------------------------------------------------- pipeline --- */}
      <section id="pipeline" className="px-6 py-20 scroll-mt-28">
        <div className="max-w-6xl mx-auto">
          <Reveal from="up" className="text-center mb-14">
            <div className="text-[11px] tracking-[0.24em] font-display mb-3"
                 style={{ color: 'var(--brand)' }}>HOW IT WORKS</div>
            <h2 className="font-display font-bold headline"
                style={{ fontSize: 'clamp(1.8rem, 4vw, 3rem)' }}>
              Six steps, about two hundred milliseconds
            </h2>
          </Reveal>

          <Reveal stagger={0.2} from="left"
                  className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {PIPELINE.map((s) => (
              <div key={s.n} className="panel p-6 lift relative overflow-hidden">
                <div className="figure absolute -top-2 right-3 text-[56px] leading-none select-none"
                     style={{ color: 'rgba(0,240,255,.07)' }} aria-hidden="true">{s.n}</div>
                <h3 className="font-display font-semibold mb-2 relative">{s.title}</h3>
                <p className="text-[13.5px] leading-relaxed relative" style={{ color: 'var(--ink-2)' }}>
                  {s.body}
                </p>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      {/* --------------------------------------------------------- honesty --- */}
      <section id="honesty" className="px-6 py-20 scroll-mt-28">
        <div className="max-w-4xl mx-auto">
          <SectionHeading
            eyebrow="LIMITATIONS"
            title="What it cannot do"
            body="A detector that claims to catch everything is lying, and a judge will find the gap faster than you will. These are stated up front, in the product and in the report."
          />
          <Reveal stagger={0.2} from="up" className="mt-10 space-y-3">
            {[
              'It detects face-based forgery. It is not a general "was this made by AI" detector — a synthetic landscape is outside its training distribution.',
              'Images with no detectable face fall back to whole-frame analysis, which is markedly less reliable. The report says so when that happens.',
              'Accuracy is measured on a held-out split of the same dataset. Cross-dataset generalisation is the harder benchmark and is not claimed.',
              'C2PA is checked for presence only, not cryptographically validated against a trust list.',
              'A verdict is evidence, not proof. It belongs as one input to a human decision, never as a replacement for one.',
            ].map((t) => (
              <div key={t} className="flex gap-3 panel p-4">
                <span aria-hidden="true" style={{ color: 'var(--warning)' }}>▲</span>
                <p className="text-[13.5px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>{t}</p>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      {/* ------------------------------------------------------------- cta --- */}
      <section className="px-6 py-24">
        <Reveal from="scale" className="max-w-3xl mx-auto text-center">
          <div className="panel p-12 relative overflow-hidden">
            <div className="absolute inset-0 pointer-events-none" aria-hidden="true"
                 style={{ background: 'radial-gradient(ellipse 70% 100% at 50% 0%, rgba(0,240,255,.14), transparent 70%)' }} />
            <h2 className="font-display font-bold mb-4 relative headline-animated"
                style={{ fontSize: 'clamp(1.7rem, 4vw, 2.6rem)' }}>
              Verify something now
            </h2>
            <p className="text-[15px] mb-8 relative" style={{ color: 'var(--ink-2)' }}>
              Drop in an image or a video clip. Everything runs on this machine — no upload,
              no account, no cloud.
            </p>
            {/* The spark runs continuously, hovered or not. */}
            <Link to="/scan"
                  className="inline-flex items-center px-8 py-3.5 rounded-full text-sm font-semibold press relative spark-sweep"
                  style={{
                    background: 'linear-gradient(135deg, var(--brand), var(--brand-2))',
                    color: 'var(--on-accent)',
                    boxShadow: '0 0 40px -10px rgba(0,240,255,.8)',
                  }}>
              <span>Open the scanner</span>
            </Link>
          </div>
        </Reveal>
      </section>

      {/* --------------------------------------------------- closing band --- */}
      {/* Three rows moving at once. Alternating directions and a 5-second phase
          offset per row mean the stack never reads as one solid block sliding
          past. No controls here — this is texture, not something to operate. */}
      <section aria-label="What OmniGuard covers"
               className="border-y overflow-hidden"
               style={{
                 borderColor: 'var(--border)',
                 background: 'linear-gradient(180deg, rgba(11,19,43,.28), rgba(3,7,18,.6))',
               }}>
        <div className="py-4 space-y-1">
          {BAND_ROWS.map((row, i) => (
            <Marquee
              key={i}
              items={row}
              controls={false}
              speed={38 + i * 7}
              initialDirection={i % 2 === 0 ? -1 : 1}
              phase={i * 5}
              className="px-2 opacity-90"
            />
          ))}
        </div>
      </section>

      <Footer />
    </div>
  )
}

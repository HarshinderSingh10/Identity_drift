import { useId, type ReactNode } from 'react'

export interface BarDatum {
  label: string
  value: number
  /** Optional second line under the category label, e.g. sample size. */
  sublabel?: string
  /** Text printed at the bar end; defaults to the formatted value. */
  display?: string
}

interface ChartBase {
  title: string
  /** Full-sentence description for assistive technology. */
  description: string
  data: BarDatum[]
  /** Axis maximum (must be >= largest value). */
  max: number
  tickStep: number
  axisLabel: string
  formatTick?: (n: number) => string
}

const W = 640
const fmtDefault = (n: number) => String(n)

/** Keeps only finite values and clamps negatives to 0 so a bad datum cannot produce NaN geometry. */
const safeValue = (v: number) => (Number.isFinite(v) && v > 0 ? v : 0)

/** Axis ticks; returns [] for an invalid max/step instead of looping forever. */
function makeTicks(max: number, tickStep: number): number[] {
  const ticks: number[] = []
  if (!(Number.isFinite(max) && max > 0 && Number.isFinite(tickStep) && tickStep > 0)) return ticks
  for (let t = 0; t <= max + 1e-9 && ticks.length < 200; t += tickStep) ticks.push(Number(t.toFixed(6)))
  return ticks
}

function EmptyChart({ message }: { message: string }) {
  return <p className="py-8 text-center text-sm text-ink-muted">{message}</p>
}

/** Vertical bar chart drawn as responsive SVG. Values are plotted exactly as given. */
export function BarChart({ title, description, data, max, tickStep, axisLabel, formatTick = fmtDefault }: ChartBase) {
  const id = useId()
  const H = 360
  const m = { top: 24, right: 16, bottom: 64, left: 56 }
  const iw = W - m.left - m.right
  const ih = H - m.top - m.bottom
  if (data.length === 0 || !(Number.isFinite(max) && max > 0)) return <EmptyChart message="No data to display." />
  const ticks = makeTicks(max, tickStep)
  const slot = iw / data.length
  const bw = Math.min(64, slot * 0.6)
  const y = (v: number) => m.top + ih - (Math.min(safeValue(v), max) / max) * ih

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby={`${id}-t ${id}-d`}
      className="h-auto w-full"
    >
      <title id={`${id}-t`}>{title}</title>
      <desc id={`${id}-d`}>{description}</desc>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={m.left} x2={W - m.right} y1={y(t)} y2={y(t)} stroke="var(--color-line)" />
          <text x={m.left - 8} y={y(t) + 4} textAnchor="end" fontSize="12" fill="var(--color-ink-muted)">
            {formatTick(t)}
          </text>
        </g>
      ))}
      <text
        transform={`translate(14 ${m.top + ih / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize="12"
        fill="var(--color-ink-muted)"
      >
        {axisLabel}
      </text>
      {data.map((d, i) => {
        const cx = m.left + slot * i + slot / 2
        const top = y(d.value)
        return (
          <g key={`${d.label}-${i}`}>
            <rect x={cx - bw / 2} y={top} width={bw} height={m.top + ih - top} fill="var(--color-accent)" rx="2" />
            <text x={cx} y={top - 6} textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--color-ink)">
              {d.display ?? String(d.value)}
            </text>
            <text x={cx} y={m.top + ih + 18} textAnchor="middle" fontSize="12" fill="var(--color-ink)">
              {d.label}
            </text>
            {d.sublabel && (
              <text x={cx} y={m.top + ih + 33} textAnchor="middle" fontSize="11" fill="var(--color-ink-muted)">
                {d.sublabel}
              </text>
            )}
          </g>
        )
      })}
      <line x1={m.left} x2={m.left} y1={m.top} y2={m.top + ih} stroke="var(--color-ink-muted)" />
      <line x1={m.left} x2={W - m.right} y1={m.top + ih} y2={m.top + ih} stroke="var(--color-ink-muted)" />
    </svg>
  )
}

/** Horizontal bar chart, suited to long category labels. */
export function HBarChart({ title, description, data, max, tickStep, axisLabel, formatTick = fmtDefault }: ChartBase) {
  const id = useId()
  const rowH = 30
  const m = { top: 12, right: 56, bottom: 48, left: 120 }
  const iw = W - m.left - m.right
  const ih = rowH * data.length
  const H = m.top + ih + m.bottom
  if (data.length === 0 || !(Number.isFinite(max) && max > 0)) return <EmptyChart message="No data to display." />
  const ticks = makeTicks(max, tickStep)
  const x = (v: number) => m.left + (Math.min(safeValue(v), max) / max) * iw

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby={`${id}-t ${id}-d`}
      className="h-auto w-full"
    >
      <title id={`${id}-t`}>{title}</title>
      <desc id={`${id}-d`}>{description}</desc>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={m.top} y2={m.top + ih} stroke="var(--color-line)" />
          <text x={x(t)} y={m.top + ih + 16} textAnchor="middle" fontSize="12" fill="var(--color-ink-muted)">
            {formatTick(t)}
          </text>
        </g>
      ))}
      <text x={m.left + iw / 2} y={H - 8} textAnchor="middle" fontSize="12" fill="var(--color-ink-muted)">
        {axisLabel}
      </text>
      {data.map((d, i) => {
        const cy = m.top + rowH * i + rowH / 2
        return (
          <g key={`${d.label}-${i}`}>
            <text x={m.left - 8} y={cy + 4} textAnchor="end" fontSize="12" fill="var(--color-ink)">
              {d.label}
            </text>
            <rect x={m.left} y={cy - 9} width={x(d.value) - m.left} height={18} fill="var(--color-accent)" rx="2" />
            <text x={x(d.value) + 6} y={cy + 4} fontSize="12" fontWeight="600" fill="var(--color-ink)">
              {d.display ?? String(d.value)}
            </text>
          </g>
        )
      })}
      <line x1={m.left} x2={m.left} y1={m.top} y2={m.top + ih} stroke="var(--color-ink-muted)" />
    </svg>
  )
}

interface ChartFigureProps {
  caption: string
  source: string
  children: ReactNode
}

/** Wraps a chart with a caption and provenance note. */
export function ChartFigure({ caption, source, children }: ChartFigureProps) {
  return (
    <figure className="min-w-0">
      <div className="rounded-card border border-line bg-page p-4 shadow-card">{children}</div>
      <figcaption className="mt-2 text-caption text-ink-muted">
        <span className="font-medium text-ink">{caption}</span> {source}
      </figcaption>
    </figure>
  )
}

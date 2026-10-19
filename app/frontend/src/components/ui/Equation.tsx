import type { ReactNode } from 'react'

/** Italic math variable with optional subscript / superscript. */
export function V({ children, sub, sup }: { children: ReactNode; sub?: ReactNode; sup?: ReactNode }) {
  return (
    <span className="font-serif italic">
      {children}
      {sub !== undefined && <sub className="text-[0.7em] not-italic">{sub}</sub>}
      {sup !== undefined && <sup className="text-[0.7em] not-italic">{sup}</sup>}
    </span>
  )
}

/** Stacked fraction built from plain HTML (no math library required). */
export function Frac({ num, den }: { num: ReactNode; den: ReactNode }) {
  return (
    <span className="mx-1 inline-flex flex-col items-center align-middle">
      <span className="border-b border-ink px-1 pb-0.5">{num}</span>
      <span className="px-1 pt-0.5">{den}</span>
    </span>
  )
}

interface EquationProps {
  /** Equation number as printed in the paper, if any. */
  number?: number
  /** Plain-text reading for assistive technology. */
  label: string
  children: ReactNode
}

/** Display equation: scrolls horizontally on narrow screens, announced via aria-label. */
export function Equation({ number, label, children }: EquationProps) {
  return (
    <div className="flex items-center gap-4 rounded-card border border-line bg-page p-4 shadow-card">
      <div
        role="math"
        aria-label={label}
        tabIndex={0}
        className="min-w-0 flex-1 overflow-x-auto py-1 text-center font-serif text-lg text-ink"
      >
        {children}
      </div>
      {number !== undefined && <span className="shrink-0 text-sm text-ink-muted">({number})</span>}
    </div>
  )
}

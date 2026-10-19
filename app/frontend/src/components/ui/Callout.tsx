import type { ReactNode } from 'react'

interface CalloutProps {
  title: string
  variant?: 'note' | 'caution'
  children: ReactNode
}

/**
 * Highlights an important interpretive caveat.
 * "note" = neutral information; "caution" = limits on how a result may be read.
 */
export function Callout({ title, variant = 'note', children }: CalloutProps) {
  const styles =
    variant === 'caution'
      ? 'border-ink-muted bg-surface'
      : 'border-accent bg-accent-soft'
  return (
    <aside className={`rounded-card border border-l-4 p-5 ${styles}`}>
      <h3 className="text-subheading text-ink">{title}</h3>
      <div className="mt-2 space-y-2 text-ink-muted">{children}</div>
    </aside>
  )
}

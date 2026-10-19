import type { ReactNode } from 'react'
import { Container } from './Container.tsx'

interface SectionProps {
  id?: string
  title?: string
  description?: string
  /** Use the subtle background to alternate sections. */
  tone?: 'plain' | 'subtle'
  children: ReactNode
}

/** A page section with an optional heading, consistent vertical rhythm and width. */
export function Section({ id, title, description, tone = 'plain', children }: SectionProps) {
  const headingId = id && title ? `${id}-heading` : undefined
  return (
    <section
      id={id}
      aria-labelledby={headingId}
      className={`scroll-mt-16 border-t border-line py-12 md:py-16 ${tone === 'subtle' ? 'bg-surface' : 'bg-page'}`}
    >
      <Container>
        {title && (
          <div className="mb-8 max-w-3xl">
            <h2 id={headingId} className="text-heading text-ink">
              {title}
            </h2>
            {description && <p className="mt-2 text-ink-muted">{description}</p>}
          </div>
        )}
        {children}
      </Container>
    </section>
  )
}

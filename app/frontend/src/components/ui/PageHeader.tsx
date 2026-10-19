import type { ReactNode } from 'react'
import { Container } from './Container.tsx'

interface PageHeaderProps {
  eyebrow?: string
  title: string
  description?: ReactNode
}

/** Standard page title block. Renders the single <h1> for each page. */
export function PageHeader({ eyebrow, title, description }: PageHeaderProps) {
  return (
    <header className="bg-surface py-12 md:py-16">
      <Container>
        <div className="max-w-3xl">
          {eyebrow && <p className="mb-3 text-label uppercase text-accent">{eyebrow}</p>}
          <h1 className="text-title text-ink md:text-display">{title}</h1>
          {description && <div className="mt-4 text-lead text-ink-muted">{description}</div>}
        </div>
      </Container>
    </header>
  )
}

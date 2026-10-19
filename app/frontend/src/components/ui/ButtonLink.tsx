import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

interface ButtonLinkProps {
  to: string
  variant?: 'primary' | 'secondary'
  children: ReactNode
}

const base =
  'inline-flex items-center justify-center rounded-control px-4 py-2.5 text-sm font-semibold transition-colors'

const variants = {
  primary: 'bg-accent text-white hover:bg-accent-strong',
  secondary: 'border border-line bg-page text-ink hover:bg-surface',
} as const

/** Router-aware link styled as a button. */
export function ButtonLink({ to, variant = 'primary', children }: ButtonLinkProps) {
  return (
    <Link to={to} className={`${base} ${variants[variant]}`}>
      {children}
    </Link>
  )
}

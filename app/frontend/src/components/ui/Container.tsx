import type { ElementType, ReactNode } from 'react'

interface ContainerProps {
  children: ReactNode
  as?: ElementType
  className?: string
}

/** Centres content to the page width with consistent horizontal gutters. */
export function Container({ children, as: Tag = 'div', className = '' }: ContainerProps) {
  return (
    <Tag className={`mx-auto w-full max-w-page px-4 sm:px-6 lg:px-8 ${className}`.trim()}>
      {children}
    </Tag>
  )
}

import { useEffect } from 'react'
import { siteConfig } from '../config/site'

/** Sets the document title for the current page, e.g. "Results · Identity Drift". */
export function usePageTitle(title?: string): void {
  useEffect(() => {
    document.title = title ? `${title} · ${siteConfig.name}` : siteConfig.projectTitle
  }, [title])
}

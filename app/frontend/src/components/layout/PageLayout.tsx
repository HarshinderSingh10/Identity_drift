import { useEffect, useRef } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Footer } from './Footer.tsx'
import { Navbar } from './Navbar.tsx'

/**
 * Shared shell for every route: skip link, navbar, main content, footer.
 * Used as a React Router layout route, so pages render through <Outlet />.
 */
export function PageLayout() {
  const { pathname, hash } = useLocation()
  const mainRef = useRef<HTMLElement>(null)
  const previousPath = useRef(pathname)

  // On client-side navigation: scroll to the #hash target if there is one,
  // otherwise reset scroll and move focus to <main> so keyboard and
  // screen-reader users start at the new page content.
  // Comparing paths (rather than a "first render" flag) keeps the initial page
  // load untouched, even when React StrictMode re-runs effects in development.
  useEffect(() => {
    const pathChanged = previousPath.current !== pathname
    previousPath.current = pathname
    if (hash) {
      const target = document.getElementById(decodeURIComponent(hash.slice(1)))
      if (target) {
        target.scrollIntoView()
        return
      }
    }
    if (!pathChanged) return
    window.scrollTo(0, 0)
    mainRef.current?.focus({ preventScroll: true })
  }, [pathname, hash])

  return (
    <div className="flex min-h-screen flex-col bg-page text-ink">
      <a
        href="#main-content"
        className="sr-only rounded-control bg-accent px-4 py-2 text-sm font-semibold text-white focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50"
      >
        Skip to main content
      </a>
      <Navbar />
      <main id="main-content" ref={mainRef} tabIndex={-1} className="flex-1 focus:outline-none">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}

import { useEffect, useId, useRef, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { navItems, siteConfig } from '../../config/site.ts'
import { Container } from '../ui/Container.tsx'

const desktopLink = ({ isActive }: { isActive: boolean }) =>
  `rounded-control px-3 py-2 text-sm font-medium transition-colors ${
    isActive ? 'bg-accent-soft text-accent' : 'text-ink-muted hover:bg-surface hover:text-ink'
  }`

const mobileLink = ({ isActive }: { isActive: boolean }) =>
  `block rounded-control px-3 py-3 text-base font-medium transition-colors ${
    isActive ? 'bg-accent-soft text-accent' : 'text-ink hover:bg-surface'
  }`

/** Shown only while no real repository URL has been supplied in site config. */
function GithubPlaceholder({ className }: { className: string }) {
  return (
    <span
      className={`${className} cursor-default text-ink-muted`}
      title="Repository link coming soon"
    >
      GitHub <span className="text-caption">(coming soon)</span>
    </span>
  )
}

export function Navbar() {
  const { pathname } = useLocation()
  const menuId = useId()
  const toggleRef = useRef<HTMLButtonElement>(null)

  // The menu is "open" only on the route where it was opened, so navigating
  // (including with the browser back button) always closes it.
  const [openAt, setOpenAt] = useState<string | null>(null)
  const isOpen = openAt === pathname

  useEffect(() => {
    if (!isOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpenAt(null)
        toggleRef.current?.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [isOpen])

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-page/95 backdrop-blur-sm">
      <Container>
        <div className="flex h-16 items-center justify-between gap-4">
          <Link
            to="/"
            className="flex items-center gap-2 rounded-control text-ink"
            aria-label={`${siteConfig.projectTitle}, home`}
          >
            <span
              aria-hidden="true"
              className="flex h-8 w-8 items-center justify-center rounded-control bg-accent text-sm font-bold text-white"
            >
              ID
            </span>
            <span className="text-subheading font-semibold tracking-tight">{siteConfig.name}</span>
          </Link>

          <nav aria-label="Primary" className="hidden lg:block">
            <ul className="flex items-center gap-1">
              {navItems.map((item) => (
                <li key={item.to}>
                  <NavLink to={item.to} end={item.end} className={desktopLink}>
                    {item.label}
                  </NavLink>
                </li>
              ))}
              <li>
                {siteConfig.githubUrl ? (
                  <a
                    href={siteConfig.githubUrl}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="rounded-control px-3 py-2 text-sm font-medium text-ink-muted transition-colors hover:bg-surface hover:text-ink"
                  >
                    GitHub
                    <span className="sr-only"> (opens in a new tab)</span>
                  </a>
                ) : (
                  <GithubPlaceholder className="inline-block px-3 py-2 text-sm font-medium" />
                )}
              </li>
            </ul>
          </nav>

          <button
            ref={toggleRef}
            type="button"
            className="inline-flex h-10 w-10 items-center justify-center rounded-control border border-line text-ink hover:bg-surface lg:hidden"
            aria-expanded={isOpen}
            aria-controls={menuId}
            aria-label={isOpen ? 'Close navigation menu' : 'Open navigation menu'}
            onClick={() => setOpenAt(isOpen ? null : pathname)}
          >
            <svg
              aria-hidden="true"
              width="20"
              height="20"
              viewBox="0 0 20 20"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
            >
              {isOpen ? (
                <path d="M4 4l12 12M16 4L4 16" />
              ) : (
                <path d="M3 5h14M3 10h14M3 15h14" />
              )}
            </svg>
          </button>
        </div>
      </Container>

      {isOpen && (
        <nav id={menuId} aria-label="Primary mobile" className="border-t border-line bg-page lg:hidden">
          <Container>
            <ul className="space-y-1 py-3">
              {navItems.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    className={mobileLink}
                    onClick={() => setOpenAt(null)}
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
              <li>
                {siteConfig.githubUrl ? (
                  <a
                    href={siteConfig.githubUrl}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="block rounded-control px-3 py-3 text-base font-medium text-ink hover:bg-surface"
                  >
                    GitHub
                    <span className="sr-only"> (opens in a new tab)</span>
                  </a>
                ) : (
                  <GithubPlaceholder className="block px-3 py-3 text-base font-medium" />
                )}
              </li>
            </ul>
          </Container>
        </nav>
      )}
    </header>
  )
}

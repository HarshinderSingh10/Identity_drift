import { Link } from 'react-router-dom'
import { navItems, researchDisclaimer, siteConfig, teamMembers } from '../../config/site.ts'
import { Container } from '../ui/Container.tsx'

export function Footer() {
  const year = new Date().getFullYear()

  return (
    <footer className="border-t border-line bg-surface">
      <Container>
        <div className="grid gap-10 py-12 md:grid-cols-2 lg:grid-cols-4">
          <div className="lg:col-span-2">
            <p className="text-subheading text-ink">{siteConfig.projectTitle}</p>
            <p className="mt-3 max-w-xl text-sm text-ink-muted">{siteConfig.description}</p>
            <p className="mt-3 max-w-xl text-sm text-ink-muted">
              Adaptive re-enrollment is discussed in the paper as a direction for future work; it has not
              been experimentally validated.
            </p>
          </div>

          <nav aria-label="Footer">
            <h2 className="text-label uppercase text-ink-muted">Explore</h2>
            <ul className="mt-3 space-y-2 text-sm">
              {navItems.map((item) => (
                <li key={item.to}>
                  <Link
                    to={item.to}
                    className="rounded-control text-ink hover:text-accent hover:underline"
                  >
                    {item.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          <div>
            <h2 className="text-label uppercase text-ink-muted">Research team</h2>
            <ul className="mt-3 space-y-2 text-sm text-ink">
              {teamMembers.map((member) => (
                <li key={member.name}>{member.name}</li>
              ))}
            </ul>
            <p className="mt-4 text-sm text-ink-muted">
              {siteConfig.department}
              <br />
              {siteConfig.institute}
            </p>
          </div>
        </div>

        <div className="space-y-3 border-t border-line py-6 text-caption text-ink-muted">
          <p>
            <span className="font-medium text-ink">Research disclaimer.</span> {researchDisclaimer}
          </p>
          <p>
            &copy; {year} {siteConfig.projectTitle} research team, {siteConfig.institute}.
          </p>
        </div>
      </Container>
    </footer>
  )
}

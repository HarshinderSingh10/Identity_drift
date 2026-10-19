import { paper } from '../data/researchContent.ts'

/**
 * Central, typed site configuration.
 * Facts here are taken from the paper (src/data/researchContent.ts).
 * Student IDs and contact details are deliberately NOT included.
 */

export interface NavItem {
  label: string
  to: string
  /** Match the route exactly (needed for "/" so it is not always active). */
  end?: boolean
}

export interface TeamMember {
  name: string
  department: string
}

export interface SiteConfig {
  /** Short wordmark text. */
  name: string
  /** Website name used in the header, footer and default page title. */
  projectTitle: string
  /** Exact title of the paper (see src/data/researchContent.ts). */
  paperTitle: string
  description: string
  institute: string
  department: string
  /** Set to the real repository URL when available; null renders a placeholder. */
  githubUrl: string | null
}

/*
 * Phase 2: supervisor, degree, session and university were removed because they
 * are not stated in the paper. See docs/CONTENT_VERIFICATION.md (section C).
 */
export const siteConfig: SiteConfig = {
  name: 'Identity Drift',
  projectTitle: 'Identity Drift Detection',
  paperTitle: paper.title,
  description:
    'A research website presenting a study of how stable face embeddings are across longitudinal observations of the same person, and how embedding drift relates to face-verification performance.',
  institute: `${paper.institute}, ${paper.location}`,
  department: paper.department,
  githubUrl: null,
}

/** Authors as listed on the paper. */
export const teamMembers: TeamMember[] = paper.authors.map((name) => ({
  name,
  department: siteConfig.department,
}))

export const navItems: NavItem[] = [
  { label: 'Home', to: '/', end: true },
  { label: 'Research', to: '/research' },
  { label: 'Methodology', to: '/methodology' },
  { label: 'Results', to: '/results' },
  { label: 'Interactive Demo', to: '/demo' },
  { label: 'Team', to: '/team' },
]

export const researchDisclaimer =
  'Results on this site reflect the reported experimental setup (MORPH cohort, fixed ArcFace embeddings, identity-disjoint split, and the stated verification threshold). They should not be interpreted as universal performance guarantees or as causal evidence about aging.'

# Identity Drift Detection: research website

Research website for the paper *Longitudinal Embedding Drift in Facial Representations*. Phase 1: shared layout and design system. Phase 2: the paper's content integrated into all six pages.

**Stack:** React 19, Vite 6, TypeScript, Tailwind CSS v4 (CSS-first config), React Router v7 (`react-router-dom`).

## Commands

```bash
npm install
npm run dev        # local dev server
npm run typecheck  # tsc -b
npm run lint       # eslint .
npm run test       # calculator unit tests (Node's built-in runner; needs Node 22.6+)
npm run build      # tsc -b && vite build
npm run preview    # serve the production build
```

## Structure

```
src/
  components/
    layout/   Navbar, Footer, PageLayout (route layout using <Outlet />)
    ui/       Container, Section, PageHeader, Callout, EvidenceTag, ButtonLink, DataTable, Equation, Cite, SourceNote, PaperFigure
  config/site.ts      Site name, nav items, authors, institute, GitHub URL (null = placeholder)
  data/               researchContent.ts (facts, text, tables, with paper-section source fields), references.ts
  lib/drift.ts        Drift arithmetic for the demo (paper Eq. 1-2)
  components/charts/  Dependency-free SVG bar charts; components/diagrams/ workflow diagram
  hooks/              usePageTitle
  pages/              Home, Research, Methodology, Results, Demo, Team, NotFound
  styles/globals.css  Design tokens (@theme) and base styles
tests/drift.test.ts            Unit tests for src/lib/drift.ts
docs/UPDATE1_VALIDATION_REPORT.md  Update 1 validation and stability report
docs/CONTENT_VERIFICATION.md   Conflicts and items the team must verify
```

## Routes

`/` Home · `/research` · `/methodology` · `/results` · `/demo` · `/team` · any other path shows a 404 page inside the same layout.

## Design tokens

Defined once in `src/styles/globals.css` under `@theme` and used as Tailwind utilities
(`bg-page`, `text-ink`, `text-ink-muted`, `bg-accent`, `bg-accent-soft`, `border-line`, `bg-surface`,
`text-display`, `text-title`, `text-heading`, `text-subheading`, `text-lead`, `text-label`, `text-caption`, `rounded-card`, `shadow-card`).

## Content rules

- Research content comes from the IEEE paper only. Do not copy numbers from the synopsis; see `docs/CONTENT_VERIFICATION.md` (which also lists items the authors must verify).
- Use `EvidenceTag` to separate reported results, interpretations, limitations and future work.
- The demo uses synthetic vectors only and must not claim a live model until one is actually connected.
- Add new facts to `src/data/researchContent.ts` with a `source`; keep numbers as strings exactly as printed.
- Hosting on a static host needs a rewrite of unknown paths to `index.html` (client-side routing).

import { Link } from 'react-router-dom'
import { ButtonLink } from '../components/ui/ButtonLink.tsx'
import { Callout } from '../components/ui/Callout.tsx'
import { Container } from '../components/ui/Container.tsx'
import { EvidenceTag } from '../components/ui/EvidenceTag.tsx'
import { Section } from '../components/ui/Section.tsx'
import { SourceNote } from '../components/ui/SourceNote.tsx'
import {
  abstractSummary,
  ageGapSummary,
  contributions,
  contributionsSummary,
  identityDriftSummary,
  motivation,
  paper,
  performanceSummary,
  studyFacts,
} from '../data/researchContent.ts'
import { usePageTitle } from '../hooks/usePageTitle.ts'

const glance = [
  { term: 'Dataset', detail: 'MORPH, a longitudinal face database' },
  {
    term: 'Embeddings',
    detail: `Fixed ${studyFacts.embeddingDimension}-dimensional ArcFace-based embeddings (${studyFacts.embeddingModel})`,
  },
  { term: 'Protocol', detail: 'Identity-disjoint train / validation / test split' },
  { term: 'Drift measure', detail: '1 − cosine similarity between two embeddings of the same person' },
]

const methodOverview = [
  'Audit MORPH and build an identity-disjoint train / validation / test split.',
  'Detect and align faces, then extract 512-dimensional ArcFace embeddings with InsightFace.',
  'Form same-identity longitudinal pairs and compute drift = 1 − cosine similarity.',
  'Analyze drift per identity and per age-gap group.',
  'Evaluate genuine verification outcomes at a threshold selected on validation and frozen before testing.',
]

// Findings below are copied from the paper's tables; indexes refer to those tables' row order.
const test = performanceSummary.rows[2]
const gap0 = ageGapSummary.rows[0]
const gap20 = ageGapSummary.rows[4]
const findings = [
  {
    value: identityDriftSummary.rows[0][1],
    label: `Identity-level mean drift, averaged over ${studyFacts.identitiesWithTwoOrMore} identities (SD ${identityDriftSummary.rows[1][1]})`,
    source: 'Section V-A',
  },
  {
    value: test[1],
    label: `Test-split ROC-AUC (EER ${test[2]}; at the frozen threshold, FAR ${test[3]} and FRR ${test[4]})`,
    source: 'Table III',
  },
  {
    value: `${gap0[4]} → ${gap20[4]}`,
    label: `Test-split FRR for 0–2 year gaps (${gap0[1]} pairs) versus 20+ year gaps (${gap20[1]} pairs)`,
    source: 'Table IV',
  },
]

const explore = [
  { to: '/research', title: 'Research', text: 'Problem, research question, related work, contributions and limitations.' },
  { to: '/methodology', title: 'Methodology', text: 'Dataset, split, embedding extraction, drift formula and verification protocol.' },
  { to: '/results', title: 'Results', text: 'Reported tables, charts and figures, separated from interpretation.' },
  { to: '/demo', title: 'Interactive Demo', text: 'An illustrative drift calculator using synthetic vectors only.' },
]

export default function Home() {
  usePageTitle()

  return (
    <>
      <section className="bg-surface py-14 md:py-20">
        <Container>
          <div className="grid items-start gap-10 lg:grid-cols-5">
            <div className="lg:col-span-3">
              <p className="mb-3 text-label uppercase text-accent">Research paper</p>
              <h1 className="text-title text-ink md:text-display">{paper.title}</h1>
              <p className="mt-4 text-sm text-ink-muted">
                {paper.authors.join(', ')} · {paper.department}, {paper.institute}, {paper.location}
              </p>
              <p className="mt-5 max-w-2xl text-lead text-ink-muted">{abstractSummary.text}</p>
              <SourceNote source={abstractSummary.source} />
              <ul className="mt-4 flex flex-wrap gap-2" aria-label="Index terms">
                {paper.indexTerms.map((term) => (
                  <li key={term} className="rounded-control border border-line bg-page px-2 py-0.5 text-caption text-ink-muted">
                    {term}
                  </li>
                ))}
              </ul>
              <div className="mt-8 flex flex-wrap gap-3">
                <ButtonLink to="/research">Research</ButtonLink>
                <ButtonLink to="/methodology" variant="secondary">Methodology</ButtonLink>
                <ButtonLink to="/results" variant="secondary">Results</ButtonLink>
                <ButtonLink to="/demo" variant="secondary">Demo</ButtonLink>
              </div>
            </div>

            <div className="rounded-card border border-line bg-page p-6 shadow-card lg:col-span-2">
              <h2 className="text-label uppercase text-ink-muted">Study at a glance</h2>
              <dl className="mt-4 space-y-4">
                {glance.map((item) => (
                  <div key={item.term}>
                    <dt className="text-sm font-semibold text-ink">{item.term}</dt>
                    <dd className="text-sm text-ink-muted">{item.detail}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </Container>
      </section>

      <Section id="motivation" title="Motivation and problem">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          {motivation.map((item) => (
            <p key={item.source + item.text.slice(0, 20)}>{item.text}</p>
          ))}
          <SourceNote source="Section I" />
          <p>
            <Link to="/research" className="font-semibold text-accent hover:underline">
              Read the full research background<span aria-hidden="true"> →</span>
            </Link>
          </p>
        </div>
      </Section>

      <Section id="contributions" tone="subtle" title="Contributions">
        <ol className="max-w-3xl list-decimal space-y-3 pl-6 text-ink-muted marker:font-semibold marker:text-accent">
          {contributions.map((item) => (
            <li key={item.text}>{item.text}</li>
          ))}
        </ol>
        <p className="mt-4 max-w-3xl text-ink-muted">{contributionsSummary.text}</p>
        <SourceNote source="Section I, paragraph 4" />
      </Section>

      <Section id="method" title="Methodology at a glance" description="A short overview; the Methodology page gives the full workflow, formulas and protocol.">
        <ol className="max-w-3xl list-decimal space-y-2 pl-6 text-ink-muted marker:font-semibold marker:text-accent">
          {methodOverview.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
        <p className="mt-4">
          <Link to="/methodology" className="font-semibold text-accent hover:underline">
            Open the methodology<span aria-hidden="true"> →</span>
          </Link>
        </p>
      </Section>

      <Section id="findings" tone="subtle" title="Key reported findings" description="Selected values exactly as reported in the paper (MORPH cohort, observational).">
        <div className="mb-4">
          <EvidenceTag kind="reported" />
        </div>
        <dl className="grid gap-4 sm:grid-cols-3">
          {findings.map((item) => (
            <div key={item.label} className="rounded-card border border-line bg-page p-6 shadow-card">
              <dt className="text-sm text-ink-muted">{item.label}</dt>
              <dd className="mt-1 text-heading text-accent tabular-nums">{item.value}</dd>
              <p className="mt-2 text-caption text-ink-muted">Paper: {item.source}</p>
            </div>
          ))}
        </dl>
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Callout title="How to read these findings" variant="caution">
            <p>
              These are associations in observational data, not proof that aging causes drift. The 20+ year
              group has only 31 test pairs. Adaptive re-enrollment is future work and has not been validated.
            </p>
          </Callout>
          <Callout title="A negative result is reported too">
            <p>
              The first Identity Stability Index formulation matched the current-drift baseline exactly and added
              no independent predictive value.
            </p>
          </Callout>
        </div>
        <p className="mt-6">
          <Link to="/results" className="font-semibold text-accent hover:underline">
            See all results<span aria-hidden="true"> →</span>
          </Link>
        </p>
      </Section>

      <Section id="explore" title="Explore the project">
        <ul className="grid gap-4 sm:grid-cols-2">
          {explore.map((item) => (
            <li key={item.to}>
              <Link
                to={item.to}
                className="block h-full rounded-card border border-line bg-page p-6 shadow-card transition-colors hover:border-accent"
              >
                <span className="text-subheading text-ink">{item.title}</span>
                <span className="mt-2 block text-sm text-ink-muted">{item.text}</span>
                <span className="mt-4 block text-sm font-semibold text-accent">
                  Open {item.title}<span aria-hidden="true"> →</span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </Section>
    </>
  )
}

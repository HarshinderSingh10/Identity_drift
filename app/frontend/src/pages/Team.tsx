import { ButtonLink } from '../components/ui/ButtonLink.tsx'
import { Callout } from '../components/ui/Callout.tsx'
import { PageHeader } from '../components/ui/PageHeader.tsx'
import { Section } from '../components/ui/Section.tsx'
import { references } from '../data/references.ts'
import { paper } from '../data/researchContent.ts'
import { siteConfig, teamMembers } from '../config/site.ts'
import { usePageTitle } from '../hooks/usePageTitle.ts'

export default function Team() {
  usePageTitle('Team')
  const pdfUrl = `${import.meta.env.BASE_URL}${paper.pdfPath}`

  return (
    <>
      <PageHeader
        eyebrow="Team"
        title="Authors, paper and references"
        description="Author and institute details are limited to what is printed on the paper."
      />

      <Section id="authors" title="Authors" description="Names and affiliation as listed on the paper, in the order printed.">
        <ul className="grid gap-4 sm:grid-cols-2">
          {teamMembers.map((member) => (
            <li key={member.name} className="rounded-card border border-line bg-page p-6 shadow-card">
              <p className="text-subheading text-ink">{member.name}</p>
              <p className="mt-1 text-sm text-ink-muted">{member.department}</p>
              <p className="text-sm text-ink-muted">{siteConfig.institute}</p>
            </li>
          ))}
        </ul>
        <p className="mt-4 max-w-3xl text-caption text-ink-muted">
          The paper does not include an acknowledgements section, author biographies or contact details, so none are shown.
        </p>
      </Section>

      <Section id="paper" tone="subtle" title="The paper">
        <div className="max-w-3xl space-y-4">
          <p className="text-subheading text-ink">{paper.title}</p>
          <p className="text-ink-muted">
            {paper.authors.join(', ')}. {paper.department}, {paper.institute}, {paper.location}.
          </p>
          <p className="text-caption text-ink-muted">
            The PDF does not state a venue, publication year or DOI, so no formal citation string is given here.
          </p>
          <a
            href={pdfUrl}
            target="_blank"
            rel="noreferrer noopener"
            className="inline-flex items-center justify-center rounded-control bg-accent px-4 py-2.5 text-sm font-semibold text-white hover:bg-accent-strong"
          >
            Open the paper (PDF)
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        </div>
      </Section>

      <Section id="references" title="References" description="As printed in the paper. Numbers match the in-text citations on the Research and Methodology pages.">
        <ol className="max-w-3xl space-y-4">
          {references.map((ref) => (
            <li key={ref.n} id={`ref-${ref.n}`} className="scroll-mt-24 flex gap-3 text-sm text-ink-muted">
              <span className="shrink-0 font-semibold text-ink">[{ref.n}]</span>
              <span>{ref.text}</span>
            </li>
          ))}
        </ol>
      </Section>

      <Section id="about-site" tone="subtle" title="About this website">
        <div className="max-w-3xl space-y-6">
          <p className="text-ink-muted">
            This site presents the paper’s content. Text labelled “Plain-language explanation (added)” was written for general
            readers and is not part of the paper. The site runs no face-recognition model.
          </p>
          <Callout title="Verification notes" variant="caution">
            <p>
              Inconsistencies and unverified items found while preparing the content are listed in the project’s
              docs/CONTENT_VERIFICATION.md.
            </p>
          </Callout>
          <ButtonLink to="/research" variant="secondary">
            Back to the research overview
          </ButtonLink>
        </div>
      </Section>
    </>
  )
}

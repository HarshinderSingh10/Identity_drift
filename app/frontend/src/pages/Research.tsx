import { Callout } from '../components/ui/Callout.tsx'
import { Cites } from '../components/ui/Cite.tsx'
import { EvidenceTag } from '../components/ui/EvidenceTag.tsx'
import { PageHeader } from '../components/ui/PageHeader.tsx'
import { Section } from '../components/ui/Section.tsx'
import { SourceNote } from '../components/ui/SourceNote.tsx'
import {
  abstract,
  conclusion,
  contributions,
  contributionsSummary,
  discussion,
  explainers,
  futureWork,
  limitations,
  motivation,
  objective,
  paper,
  relatedWork,
  researchGap,
  researchQuestion,
} from '../data/researchContent.ts'
import { usePageTitle } from '../hooks/usePageTitle.ts'

export default function Research() {
  usePageTitle('Research')

  return (
    <>
      <PageHeader
        eyebrow="Research"
        title="Research background and questions"
        description={
          <>
            Background, research question, related work and scope of{' '}
            <cite className="not-italic">&ldquo;{paper.title}&rdquo;</cite>.
          </>
        }
      />

      <Section id="abstract" title="Abstract" description="Verbatim from the paper.">
        <div className="max-w-3xl">
          <p className="text-ink-muted">{abstract.text}</p>
          <SourceNote source={abstract.source} />
        </div>
      </Section>

      <Section id="problem" tone="subtle" title="Problem and motivation">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          {motivation.map((item) => (
            <p key={item.text}>{item.text}</p>
          ))}
          <SourceNote source="Section I" />
        </div>
      </Section>

      <Section id="question" title="Research question and objective">
        <div className="max-w-3xl space-y-6">
          <div>
            <h3 className="text-subheading text-ink">Research question</h3>
            <p className="mt-2 text-ink-muted">{researchQuestion.text}</p>
            <SourceNote source={researchQuestion.source} />
          </div>
          <div>
            <h3 className="text-subheading text-ink">Objective</h3>
            <p className="mt-2 text-ink-muted">{objective.text}</p>
            <SourceNote source={objective.source} />
          </div>
          <div className="space-y-2">
            <EvidenceTag kind="explainer" />
            <p className="text-ink-muted">{explainers.significance}</p>
          </div>
        </div>
      </Section>

      <Section id="related-work" tone="subtle" title="Related work" description="Section II of the paper. Numbers in brackets link to the reference list.">
        <div className="max-w-3xl space-y-8">
          {relatedWork.map((block) => (
            <div key={block.heading}>
              <h3 className="text-subheading text-ink">{block.heading}</h3>
              <p className="mt-2 text-ink-muted">
                {block.text} <Cites numbers={block.cites} />
              </p>
              <SourceNote source={block.source} />
            </div>
          ))}
        </div>
      </Section>

      <Section id="gap" title="Research gap and positioning">
        <div className="max-w-3xl">
          <p className="text-ink-muted">{researchGap.text}</p>
          <SourceNote source={researchGap.source} />
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

      <Section
        id="discussion"
        title="Discussion and conclusion"
        description="How the paper itself interprets the reported results. Each statement is labelled."
      >
        <ul className="max-w-3xl space-y-5">
          {discussion.map((item) => (
            <li key={item.text} className="space-y-2">
              <EvidenceTag kind="interpretation" />
              <p className="text-ink-muted">{item.text}</p>
              <SourceNote source={item.source} />
            </li>
          ))}
          <li className="space-y-2">
            <EvidenceTag kind="interpretation" />
            <p className="text-ink-muted">{conclusion.text}</p>
            <SourceNote source={conclusion.source} />
          </li>
        </ul>
        <div className="mt-8 max-w-3xl">
          <Callout title="Not claimed by this study" variant="caution">
            <p>
              The paper does not claim causation, a validated adaptive re-enrollment system, or that the
              verification threshold is a re-enrollment threshold. The threshold is a verification operating
              boundary only.
            </p>
          </Callout>
        </div>
      </Section>

      <Section id="limitations" tone="subtle" title="Limitations and threats to validity">
        <div className="mb-4">
          <EvidenceTag kind="limitation" />
        </div>
        <ol className="max-w-3xl list-decimal space-y-2 pl-6 text-ink-muted">
          {limitations.map((item) => (
            <li key={item.source}>{item.text}</li>
          ))}
        </ol>
        <SourceNote source="Section VIII" />
      </Section>

      <Section id="future-work" title="Future work">
        <div className="mb-4">
          <EvidenceTag kind="future" />
        </div>
        <p className="mb-3 max-w-3xl text-ink-muted">
          The paper motivates drift-aware re-enrollment research but does not validate it. It proposes that future
          work examine:
        </p>
        <ul className="max-w-3xl list-disc space-y-2 pl-6 text-ink-muted">
          {futureWork.map((item) => (
            <li key={item.text}>{item.text}</li>
          ))}
        </ul>
        <SourceNote source="Section IX" />
      </Section>
    </>
  )
}

import { PipelineDiagram } from '../components/diagrams/PipelineDiagram.tsx'
import { Callout } from '../components/ui/Callout.tsx'
import { DataTable } from '../components/ui/DataTable.tsx'
import { Equation, Frac, V } from '../components/ui/Equation.tsx'
import { EvidenceTag } from '../components/ui/EvidenceTag.tsx'
import { PageHeader } from '../components/ui/PageHeader.tsx'
import { Section } from '../components/ui/Section.tsx'
import { SourceNote } from '../components/ui/SourceNote.tsx'
import { Cites } from '../components/ui/Cite.tsx'
import {
  coverageSummary,
  datasetSummary,
  explainers,
  isiTerms,
  pipelineStages,
  splitSummary,
  studyFacts,
} from '../data/researchContent.ts'
import { usePageTitle } from '../hooks/usePageTitle.ts'

const metrics = [
  { term: 'Drift summaries', detail: 'Identity-level mean, standard deviation and percentiles of drift (Section V-A) and mean drift per age-gap group (Table IV).' },
  { term: 'ROC-AUC, EER', detail: 'Threshold-independent summaries of verification quality (ranking quality and equal error rate), reported for train, validation and test.' },
  { term: 'FAR, FRR, acceptance', detail: 'Threshold-dependent rates, evaluated at the single frozen threshold; FRR and acceptance are also broken out by age gap on the test split.' },
  { term: 'PR-AUC, Brier', detail: 'Used only in the exploratory ISI comparison (Table V).' },
]

export default function Methodology() {
  usePageTitle('Methodology')

  return (
    <>
      <PageHeader
        eyebrow="Methodology"
        title="Experimental methodology"
        description="How the MORPH cohort was audited, split, embedded and analyzed to measure longitudinal drift and its relation to verification."
      />

      <Section id="pipeline" title="Overall workflow" description="Reconstructed from the paper's Fig. 1. The final stage is future work, not part of the experiments.">
        <PipelineDiagram stages={pipelineStages} />
        <p className="mx-auto mt-3 max-w-md text-center text-caption text-ink-muted">
          Figure 1 of the paper. Source: Section III, Fig. 1 (p. 2).
        </p>
      </Section>

      <Section id="datasets" tone="subtle" title="Datasets">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          <p>
            <strong className="text-ink">MORPH</strong> is the primary dataset because it contains repeated observations of the
            same identities across recorded ages, which suits within-identity representation change over time{' '}
            <Cites numbers={[4]} />. The audit found {studyFacts.rawImages} raw images (approximately) and approximately{' '}
            {studyFacts.rawIdentities} identities.
          </p>
          <p>
            The age variable used for ordering and modeling is the <strong className="text-ink">filename-derived age</strong>.
            It is useful for ordering and age-gap analysis but is not a perfect biological-age measurement and should not be
            read as exact physical age or a causal treatment of aging.
          </p>
          <p>
            <strong className="text-ink">AgeDB</strong> <Cites numbers={[5]} /> was audited as a secondary dataset (about 12,240
            images, about 440 identities, four protocol files for 5-, 10-, 20- and 30-year splits). During image-level
            reconstruction, 15,716 records were ambiguous or unresolved, so it was not used as the primary image-level
            validation dataset and is not an evidence source for the paper&rsquo;s claims.
          </p>
          <SourceNote source="Section III-A" />
        </div>
        <div className="mt-8">
          <DataTable
            caption="Dataset summary"
            columns={datasetSummary.columns}
            rows={datasetSummary.rows}
            source={`Source: paper, ${datasetSummary.source}.`}
          />
        </div>
      </Section>

      <Section id="split" title="Identity-disjoint research split">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          <p>
            The original MORPH dataset was not organized for an identity-disjoint longitudinal evaluation. The authors
            constructed a research split at random seed {studyFacts.splitSeed}; no identity appears in more than one of the
            train, validation and test partitions. This reduces identity leakage and gives a more reliable held-out
            evaluation of drift.
          </p>
          <SourceNote source="Section III-B" />
          <div className="space-y-2">
            <EvidenceTag kind="explainer" />
            <p>{explainers.identityDisjoint}</p>
          </div>
        </div>
        <div className="mt-8">
          <DataTable
            caption="Identity-disjoint research split"
            columns={splitSummary.columns}
            rows={splitSummary.rows}
            source={`Source: paper, ${splitSummary.source}.`}
          />
        </div>
      </Section>

      <Section id="preprocessing" tone="subtle" title="Preprocessing and embedding extraction">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          <p>
            Raw images were processed with InsightFace face detection and ArcFace-compatible alignment using the standard{' '}
            {studyFacts.alignmentSize} crop. {studyFacts.processedImages} of {studyFacts.rawImages} images were handled
            successfully; 2 failed (zero faces detected). Detection found {studyFacts.singleFaceCases} single-face cases and{' '}
            {studyFacts.multipleFaceCases} multiple-face cases.
          </p>
          <p>
            Embeddings are {studyFacts.embeddingDimension}-dimensional ArcFace embeddings extracted with the InsightFace{' '}
            <code className="font-mono text-sm">buffalo_l</code> model, followed by L2 normalization.
            Quality checks confirmed {studyFacts.processedImages} successful embeddings, no extraction failures, no missing or
            unexpected embeddings, and that all embeddings were finite, 512-dimensional and approximately unit-normalized.
            The ArcFace model is pretrained and fixed; it was not re-optimized for this study.
          </p>
          <SourceNote source="Section IV-A" />
          <div className="space-y-2">
            <EvidenceTag kind="explainer" />
            <p>{explainers.embedding}</p>
          </div>
        </div>
      </Section>

      <Section id="pairs" title="Longitudinal pair construction">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          <p>
            A pair is two different observations of the same identity. Its age gap is the absolute difference in
            filename-derived age: <V>gap</V> = |<V>age</V><sub>i</sub> − <V>age</V><sub>j</sub>| (the paper states this in
            words; the symbolic form is added here). Preprocessing produced {studyFacts.eligiblePairs} eligible same-identity
            pairs.
          </p>
          <p>
            The analytical reference is the youngest observation by filename-derived age, with filename tie-breaking as a
            deterministic ordering rule. It is an analytical baseline for studying change in a fixed representation space, not
            necessarily a true enrollment image.
          </p>
          <SourceNote source="Section III-C" />
        </div>
        <div className="mt-8">
          <DataTable
            caption="Longitudinal coverage of the MORPH cohort"
            columns={coverageSummary.columns}
            rows={coverageSummary.rows}
            source={`Source: paper, ${coverageSummary.source}.`}
          />
        </div>
      </Section>

      <Section id="drift-definition" tone="subtle" title="Drift definition">
        <div className="max-w-3xl space-y-4">
          <p className="text-ink-muted">
            For normalized embeddings <V sub="i">e</V> and <V sub="j">e</V>, cosine similarity is
          </p>
          <Equation number={1} label="similarity of e i and e j equals e i transpose times e j, divided by the norm of e i times the norm of e j">
            sim(<V sub="i">e</V>, <V sub="j">e</V>) =
            <Frac
              num={<><V sub="i" sup="T">e</V> <V sub="j">e</V></>}
              den={<>‖<V sub="i">e</V>‖ ‖<V sub="j">e</V>‖</>}
            />
          </Equation>
          <p className="text-ink-muted">and embedding drift is defined as</p>
          <Equation number={2} label="drift equals one minus the similarity of e i and e j">
            drift = 1 − sim(<V sub="i">e</V>, <V sub="j">e</V>)
          </Equation>
          <p className="text-ink-muted">
            Low drift indicates a more similar representation; high drift indicates larger representation change.
            Euclidean distance was retained as a secondary consistency metric, but the primary metric is cosine-based drift.
            Drift is a representation-change metric and is not interpreted as recognition error.
          </p>
          <SourceNote source="Section IV-B, Eq. (1)–(2)" />
          <div className="space-y-2">
            <EvidenceTag kind="explainer" />
            <p className="text-ink-muted">{explainers.cosine}</p>
          </div>
        </div>
      </Section>

      <Section id="analysis" title="Identity-level and age-gap analyses">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          <p>
            Drift was summarized per identity across all available same-identity pairs. Observations were also grouped by age
            gap into <strong className="text-ink">0–2, 3–4, 5–9, 10–19 and 20+ years</strong>. These analyses are descriptive
            and observational.
          </p>
          <SourceNote source="Section IV-C" />
        </div>
      </Section>

      <Section id="verification" tone="subtle" title="Verification protocol and threshold">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          <p>
            Verification used genuine longitudinal comparisons and a deterministic within-split impostor set constructed with
            random seed {studyFacts.impostorSeed}. The primary threshold was selected on the validation split using the
            empirical {studyFacts.impostorFarRule} impostor FAR rule and then frozen before testing.
          </p>
          <SourceNote source="Section IV-D" />
        </div>
        <dl className="mt-6 grid max-w-3xl gap-4 sm:grid-cols-2">
          <div className="rounded-card border border-line bg-page p-5 shadow-card">
            <dt className="text-sm text-ink-muted">Similarity threshold</dt>
            <dd className="mt-1 break-all font-mono text-subheading text-ink">{studyFacts.similarityThreshold}</dd>
          </div>
          <div className="rounded-card border border-line bg-page p-5 shadow-card">
            <dt className="text-sm text-ink-muted">Corresponding drift boundary</dt>
            <dd className="mt-1 break-all font-mono text-subheading text-ink">{studyFacts.driftBoundary}</dd>
          </div>
        </dl>
        <div className="mt-6 max-w-3xl space-y-6">
          <Callout title="A verification boundary, not a re-enrollment rule" variant="caution">
            <p>
              This is the operating point chosen for verification. The paper states it should not be interpreted as a
              validated adaptive re-enrollment threshold.
            </p>
          </Callout>
          <div className="space-y-2 text-ink-muted">
            <EvidenceTag kind="explainer" />
            <p>{explainers.frozenThreshold}</p>
            <p>{explainers.farFrr}</p>
          </div>
        </div>
      </Section>

      <Section id="metrics" title="Evaluation metrics">
        <dl className="grid max-w-3xl gap-x-8 gap-y-5 sm:grid-cols-2">
          {metrics.map((item) => (
            <div key={item.term}>
              <dt className="font-semibold text-ink">{item.term}</dt>
              <dd className="text-ink-muted">{item.detail}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 max-w-3xl text-caption text-ink-muted">
          Metric names are from the paper&rsquo;s Tables III–V; the one-line descriptions are added summaries. The paper does not
          define PR-AUC or Brier, and does not state the outcome that Table V predicts (see the verification notes).
        </p>
      </Section>

      <Section id="exploratory" tone="subtle" title="Exploratory analyses">
        <div className="max-w-3xl space-y-4 text-ink-muted">
          <div>
            <h3 className="text-subheading text-ink">Identity Stability Index (ISI)</h3>
            <p className="mt-2">{isiTerms.text}</p>
            <SourceNote source={isiTerms.source} />
          </div>
          <div>
            <h3 className="text-subheading text-ink">Trajectories</h3>
            <p className="mt-2">
              Identity-specific drift trajectories were inspected over repeated observations. Candidate return/persistent
              events were identified with a training-derived exploratory threshold of{' '}
              <span className="font-mono text-sm">{studyFacts.exploratoryTrajectoryThreshold}</span>. These are descriptive
              candidate events only, not evidence of a physiological mechanism or a production rule.
            </p>
            <SourceNote source="Section V-F" />
          </div>
        </div>
      </Section>

      <Section id="caveats" title="Methodological caveats">
        <div className="max-w-3xl space-y-6">
          <Callout title="Age variable" variant="caution">
            <p>
              Age is derived from filenames. It orders observations and defines age gaps, but it is not an exact
              biological-age measurement and is not a causal treatment of aging. Observed drift may also reflect
              illumination, pose, expression, facial hair, image quality, camera/device, compression and appearance
              changes.
            </p>
          </Callout>
          <Callout title="Count differences between tables">
            <p>
              Table I reports 84,133 eligible pairs, while the three per-split pair counts in Table II add up to 84,137.
              The site reproduces both as printed; the paper does not explain the difference of 4. See the verification
              notes in the project documentation.
            </p>
          </Callout>
        </div>
      </Section>
    </>
  )
}

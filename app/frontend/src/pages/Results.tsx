import { BarChart, ChartFigure, HBarChart } from '../components/charts/BarCharts.tsx'
import { Callout } from '../components/ui/Callout.tsx'
import { DataTable } from '../components/ui/DataTable.tsx'
import { EvidenceTag } from '../components/ui/EvidenceTag.tsx'
import { PageHeader } from '../components/ui/PageHeader.tsx'
import { PaperFigure } from '../components/ui/PaperFigure.tsx'
import { Section } from '../components/ui/Section.tsx'
import { SourceNote } from '../components/ui/SourceNote.tsx'
import {
  ageGapChart,
  ageGapSummary,
  ageGapTextMeans,
  driftBinsFinding,
  explainers,
  identityDriftChart,
  identityDriftSummary,
  isiResult,
  isiSummary,
  performanceSummary,
  studyFacts,
  trajectoryFinding,
} from '../data/researchContent.ts'
import { usePageTitle } from '../hooks/usePageTitle.ts'

const dashedPercent = (n: number) => `${n}%`

export default function Results() {
  usePageTitle('Results')

  return (
    <>
      <PageHeader
        eyebrow="Results"
        title="Experimental results"
        description="Values reported in the paper for the MORPH cohort with fixed ArcFace embeddings. Reported values, interpretation and added explanations are labelled separately."
      />

      <Section id="scope" title="Scope of these results">
        <div className="max-w-3xl">
          <Callout title="Read with the reported setup in mind" variant="caution">
            <p>
              All values come from the MORPH cohort with an identity-disjoint split. FAR, FRR and acceptance rates use a
              single frozen verification threshold ({studyFacts.driftBoundary} in drift terms); drift statistics, ROC-AUC and
              EER do not depend on it. The results are observational and should not be read as universal
              performance guarantees or as causal evidence about aging.
            </p>
          </Callout>
        </div>
      </Section>

      <Section
        id="identity-drift"
        tone="subtle"
        title="Identity-level drift"
        description="Summary across 13,119 identities (the paper’s count of identities with at least two observations)."
      >
        <div className="mb-4">
          <EvidenceTag kind="reported" />
        </div>
        <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-2">
          <DataTable
            caption="Identity-level mean drift summary"
            columns={identityDriftSummary.columns}
            rows={identityDriftSummary.rows}
            source={`Source: paper, ${identityDriftSummary.source}.`}
          />
          <ChartFigure
            caption="Identity-level mean drift: summary statistics from the table (standard deviation not plotted)."
            source="Plotted directly from the reported values; the full distribution is shown in the paper’s Fig. 3 below."
          >
            <HBarChart
              title="Identity-level drift summary statistics"
              description="Horizontal bars for minimum 0.0559, 25th percentile 0.1818, median 0.2180, mean 0.2296, 75th percentile 0.2620, 90th percentile 0.3118, 95th percentile 0.3469, 99th percentile 0.4653 and maximum 1.2184."
              data={identityDriftChart.map((d) => ({ label: d.label, value: d.value, display: d.value.toFixed(4) }))}
              max={1.4}
              tickStep={0.2}
              axisLabel="Embedding drift (1 − cosine similarity)"
              formatTick={(n) => n.toFixed(1)}
            />
          </ChartFigure>
        </div>
        <div className="mt-6 max-w-3xl space-y-2">
          <EvidenceTag kind="interpretation" />
          <p className="text-ink-muted">
            Representation change is not negligible even for the same identity, and there is substantial between-identity
            heterogeneity.
          </p>
          <SourceNote source="Section V-A" />
          <p className="text-caption text-ink-muted">
            The 13,119 identities equal the whole-cohort count in Section III-C, so this summary appears to cover all splits,
            not only the held-out test split. The paper does not state this explicitly.
          </p>
        </div>
        <div className="mt-8 max-w-3xl">
          <PaperFigure
            file="fig3-identity-drift-distribution.png"
            alt="Histogram of identity-level mean longitudinal embedding drift across 13,119 identities with mean, median, 95th and 99th percentile markers."
            caption="Fig. 3. Distribution of identity-level mean longitudinal embedding drift across 13,119 identities."
            source="Fig. 3, p. 4"
            width={1400}
            height={967}
          />
          <p className="mt-2 text-caption text-ink-muted">
            The histogram’s bin counts are not given as numbers in the paper, so it is shown as printed rather than redrawn.
          </p>
        </div>
      </Section>

      <Section
        id="age-gap"
        title="Age-gap drift and verification failure"
        description="Held-out test split only (Table IV)."
      >
        <div className="mb-4">
          <EvidenceTag kind="reported" />
        </div>
        <div className="space-y-8">
          <DataTable
            caption="Age-gap drift and verification performance on the test split"
            columns={ageGapSummary.columns}
            rows={ageGapSummary.rows}
            source={`Source: paper, ${ageGapSummary.source}. FRR = false rejection rate.`}
          />
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <ChartFigure
              caption="Mean drift by age gap, test split."
              source="Plotted from the Mean drift column of Table IV."
            >
              <BarChart
                title="Mean drift by longitudinal age gap, test split"
                description="Mean drift is 0.2382 for 0–2 years, 0.2693 for 3–4 years, 0.3363 for 5–9 years, 0.3662 for 10–19 years and 0.6681 for 20+ years."
                data={ageGapChart.map((d) => ({
                  label: d.label,
                  sublabel: `n=${d.pairs.toLocaleString('en-US')}`,
                  value: d.meanDrift,
                  display: d.meanDrift.toFixed(4),
                }))}
                max={0.8}
                tickStep={0.2}
                axisLabel="Mean drift"
                formatTick={(n) => n.toFixed(1)}
              />
            </ChartFigure>
            <ChartFigure
              caption="False rejection rate by age gap, test split (recreates Fig. 5)."
              source="Plotted from the FRR column of Table IV. The 20+ year bar rests on 31 pairs."
            >
              <BarChart
                title="False rejection rate by longitudinal age gap, test split"
                description="FRR is 0.07% for 0–2 years, 0.24% for 3–4 years, 6.39% for 5–9 years, 11.24% for 10–19 years and 61.29% for 20+ years."
                data={ageGapChart.map((d) => ({
                  label: d.label,
                  sublabel: `n=${d.pairs.toLocaleString('en-US')}`,
                  value: d.frrPercent,
                  display: dashedPercent(d.frrPercent),
                }))}
                max={70}
                tickStep={10}
                axisLabel="False rejection rate (%)"
                formatTick={(n) => `${n}`}
              />
            </ChartFigure>
          </div>
          <div className="max-w-3xl">
            <Callout title="Interpret the 20+ year row cautiously" variant="caution">
              <p>
                The 20+ year group contains only 31 test pairs and should not be treated as a general population estimate.
                Higher drift and failure rates at larger age gaps are associations in observational data, not proof of
                causation.
              </p>
            </Callout>
          </div>
        </div>

        <div className="mt-12 grid grid-cols-1 items-start gap-8 lg:grid-cols-2">
          <div className="space-y-4">
            <h3 className="text-subheading text-ink">A second set of age-gap means in the paper’s text</h3>
            <DataTable
              caption="Mean drift by age gap as stated in Section V-B"
              columns={ageGapTextMeans.columns}
              rows={ageGapTextMeans.rows}
              source={`Source: paper, ${ageGapTextMeans.source}. Not the same values as Table IV.`}
            />
            <p className="text-caption text-ink-muted">
              Section V-B describes these as approximate means for “the MORPH pairs analyzed here”. They differ from Table IV
              (test split only) and the paper does not state the population. This site does not merge the two; see the
              verification notes.
            </p>
          </div>
          <PaperFigure
            file="fig4-age-gap-drift.png"
            alt="Box plot of embedding drift for five longitudinal age-gap groups with sample sizes 77,762; 4,676; 922; 660 and 113."
            caption="Fig. 4. Distribution of embedding drift across longitudinal age-gap groups."
            source="Fig. 4, p. 4"
            width={1400}
            height={1052}
          />
        </div>
        <div className="mt-6 max-w-3xl space-y-2">
          <EvidenceTag kind="interpretation" />
          <p className="text-ink-muted">
            Observed drift increased with larger age gaps, and larger age gaps were associated with higher FRR. The effect is
            observational and not uniform across identities.
          </p>
          <SourceNote source="Sections V-B, V-D, VI" />
        </div>
      </Section>

      <Section
        id="verification"
        tone="subtle"
        title="Verification performance"
        description="ROC-AUC and EER do not depend on a threshold. FAR and FRR are evaluated at the threshold selected on the validation split and frozen before testing."
      >
        <div className="mb-4">
          <EvidenceTag kind="reported" />
        </div>
        <DataTable
          caption="Embedding and verification performance"
          columns={performanceSummary.columns}
          rows={performanceSummary.rows}
          source={`Source: paper, ${performanceSummary.source}. ROC-AUC and EER are threshold-independent; FAR and FRR are evaluated at the frozen threshold.`}
        />
        <div className="mt-6 max-w-3xl space-y-4">
          <p className="text-ink-muted">
            The held-out test split achieved ROC-AUC 0.996201, EER 0.003745, FAR 0.008592 and FRR 0.007270; validation and
            train performance were comparably strong.
          </p>
          <SourceNote source="Section V-C" />
          <div className="space-y-2">
            <EvidenceTag kind="explainer" />
            <p className="text-ink-muted">{explainers.driftVersusError}</p>
          </div>
          <Callout title="Overall FRR versus the age-gap table">
            <p>
              Table III’s FRR values are not whole-number fractions of the Table II pair counts (for the test split,
              0.007270 × 12,289 ≈ 89.3), whereas Table IV’s rows imply 53 rejected pairs of 12,289 (about 0.43%). Table III
              therefore does not appear to use the same 12,289 genuine pairs as Table IV. This is unresolved and flagged for
              the authors; see the verification notes.
            </p>
          </Callout>
        </div>
      </Section>

      <Section id="boundary" title="Drift bins and the verification boundary">
        <div className="mb-4">
          <EvidenceTag kind="reported" />
        </div>
        <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-2">
          <div className="space-y-3">
            <p className="text-ink-muted">{driftBinsFinding.text}</p>
            <SourceNote source={driftBinsFinding.source} />
            <p className="text-caption text-ink-muted">
              The paper describes this drift-bin pattern in words and in Fig. 6 only; it does not publish a table of
              per-bin values, so no bin-level numbers or chart are shown here.
            </p>
          </div>
          <PaperFigure
            file="fig6-verification-boundary.png"
            alt="Histogram of genuine test-pair embedding drift with a dashed vertical line at the frozen verification boundary, drift 0.7932."
            caption="Fig. 6. Distribution of genuine test-pair embedding drift and the frozen verification operating boundary at drift = 0.793229669333."
            source="Fig. 6, p. 5"
            width={1400}
            height={988}
          />
        </div>
      </Section>

      <Section id="trajectories" tone="subtle" title="Longitudinal trajectories">
        <div className="mb-4">
          <EvidenceTag kind="reported" />
        </div>
        <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-2">
          <div className="space-y-3">
            <p className="text-ink-muted">{trajectoryFinding.text}</p>
            <SourceNote source={trajectoryFinding.source} />
            <p className="text-caption text-ink-muted">
              The trajectory values behind the figure are not tabulated in the paper, so the figure is shown as printed.
            </p>
          </div>
          <PaperFigure
            file="fig7-trajectories.png"
            alt="Four example identities showing stable, increasing and heterogeneous drift across filename-derived age."
            caption="Fig. 7. Representative longitudinal embedding-drift trajectories. Descriptive; no causal mechanism is implied."
            source="Fig. 7, p. 5"
            width={1400}
            height={1104}
          />
        </div>
      </Section>

      <Section
        id="isi"
        title="Exploratory Identity Stability Index (ISI)"
        description="A negative result for the first formulation."
      >
        <div className="mb-4">
          <EvidenceTag kind="reported" />
        </div>
        <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-2">
          <DataTable
            caption="Exploratory ISI comparison"
            columns={isiSummary.columns}
            rows={isiSummary.rows}
            source={`Source: paper, ${isiSummary.source}.`}
          />
          <div className="space-y-3">
            <p className="text-ink-muted">{isiResult.text}</p>
            <SourceNote source={isiResult.source} />
            <p className="text-caption text-ink-muted">
              Confidence intervals are not available: the intended bootstrap was not executed. The paper does not state which
              outcome Table V predicts.
            </p>
          </div>
        </div>
      </Section>

      <Section id="figures" tone="subtle" title="Other figures in the paper">
        <div className="max-w-3xl">
          <p className="text-ink-muted">
            Fig. 5 (FRR by age gap) is recreated above from Table IV. Fig. 1 (workflow) is reconstructed on the Methodology page.
            Fig. 2, an illustration of comparing one person’s embeddings at several ages, is not reproduced.
          </p>
        </div>
      </Section>
    </>
  )
}

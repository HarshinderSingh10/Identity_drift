import { useState } from 'react'
import { ButtonLink } from '../components/ui/ButtonLink.tsx'
import { Callout } from '../components/ui/Callout.tsx'
import { Equation, Frac, V } from '../components/ui/Equation.tsx'
import { EvidenceTag } from '../components/ui/EvidenceTag.tsx'
import { PageHeader } from '../components/ui/PageHeader.tsx'
import { Section } from '../components/ui/Section.tsx'
import { explainers, studyFacts } from '../data/researchContent.ts'
import { usePageTitle } from '../hooks/usePageTitle.ts'
import { computeDrift, MAX_DIMENSIONS, parseVector, presets } from '../lib/drift.ts'

const fmt = (n: number) => n.toFixed(6)

const inputClass =
  'mt-1 block w-full rounded-control border border-line bg-page p-3 font-mono text-sm text-ink'

export default function Demo() {
  usePageTitle('Interactive Demo')
  const [a, setA] = useState(presets[1].a)
  const [b, setB] = useState(presets[1].b)
  const [note, setNote] = useState(presets[1].note)

  const parsedA = parseVector(a, 'Vector A')
  const parsedB = parseVector(b, 'Vector B')
  const outcome =
    parsedA.ok && parsedB.ok
      ? computeDrift(parsedA.values, parsedB.values)
      : { ok: false as const, error: !parsedA.ok ? parsedA.error : (parsedB as { error: string }).error }

  return (
    <>
      <PageHeader
        eyebrow="Interactive Demo"
        title="Drift-score calculator"
        description="An illustrative calculator for the paper’s drift formula. It works on short synthetic number lists you type in, not on faces."
      />

      <Section id="status" title="What this demo is, and is not">
        <div className="max-w-3xl space-y-6">
          <Callout title="Illustrative only: synthetic values, no model" variant="caution">
            <p>
              Nothing here processes images, extracts embeddings, or runs a face-recognition model. The vectors below are
              made-up numbers. The calculator only applies the arithmetic of Eq. (1) and (2) from the paper. It is separate
              from, and does not reproduce, any experiment on the Results page.
            </p>
          </Callout>
        </div>
      </Section>

      <Section id="formula" tone="subtle" title="Formula used">
        <div className="max-w-3xl space-y-4">
          <Equation number={1} label="similarity of e i and e j equals e i transpose times e j, divided by the norm of e i times the norm of e j">
            sim(<V sub="i">e</V>, <V sub="j">e</V>) =
            <Frac
              num={<><V sub="i" sup="T">e</V> <V sub="j">e</V></>}
              den={<>‖<V sub="i">e</V>‖ ‖<V sub="j">e</V>‖</>}
            />
          </Equation>
          <Equation number={2} label="drift equals one minus the similarity of e i and e j">
            drift = 1 − sim(<V sub="i">e</V>, <V sub="j">e</V>)
          </Equation>
          <p className="text-caption text-ink-muted">Paper: Section IV-B, Eq. (1)–(2).</p>
          <div className="space-y-2 text-ink-muted">
            <EvidenceTag kind="explainer" />
            <p>{explainers.cosine}</p>
            <p>
              For any two nonzero vectors, similarity lies between −1 and 1, so the drift computed here lies between 0 and 2.
              The paper’s real embeddings have {studyFacts.embeddingDimension} dimensions; here you may enter 2 to{' '}
              {MAX_DIMENSIONS} numbers.
            </p>
          </div>
        </div>
      </Section>

      <Section id="calculator" title="Calculator" description="Both lists must have the same length. Separate numbers with commas, spaces or new lines.">
        <div className="max-w-3xl space-y-6">
          <p className="inline-block rounded-control border border-dashed border-ink-muted px-2 py-0.5 text-label uppercase text-ink-muted">
            Illustrative · synthetic values
          </p>

          <fieldset>
            <legend className="text-sm font-semibold text-ink">Example inputs</legend>
            <div className="mt-2 flex flex-wrap gap-2">
              {presets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className="rounded-control border border-line bg-page px-3 py-2 text-sm font-medium text-ink hover:bg-surface"
                  onClick={() => {
                    setA(p.a)
                    setB(p.b)
                    setNote(p.note)
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <p className="mt-2 text-caption text-ink-muted">{note}</p>
          </fieldset>

          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="vec-a" className="text-sm font-semibold text-ink">
                Vector A (e<sub>i</sub>)
              </label>
              <textarea
                id="vec-a"
                rows={3}
                className={inputClass}
                value={a}
                onChange={(e) => {
                  setA(e.target.value)
                  setNote('Custom synthetic input.')
                }}
                spellCheck={false}
              />
            </div>
            <div>
              <label htmlFor="vec-b" className="text-sm font-semibold text-ink">
                Vector B (e<sub>j</sub>)
              </label>
              <textarea
                id="vec-b"
                rows={3}
                className={inputClass}
                value={b}
                onChange={(e) => {
                  setB(e.target.value)
                  setNote('Custom synthetic input.')
                }}
                spellCheck={false}
              />
            </div>
          </div>

          <div aria-live="polite" className="rounded-card border border-line bg-surface p-6">
            <h3 className="text-label uppercase text-ink-muted">Output</h3>
            {outcome.ok ? (
              <dl className="mt-4 grid gap-4 sm:grid-cols-2">
                <div>
                  <dt className="text-sm text-ink-muted">Cosine similarity, sim(e<sub>i</sub>, e<sub>j</sub>)</dt>
                  <dd className="font-mono text-subheading text-ink">{fmt(outcome.result.similarity)}</dd>
                </div>
                <div>
                  <dt className="text-sm text-ink-muted">Drift = 1 − similarity</dt>
                  <dd className="font-mono text-title text-accent">{fmt(outcome.result.drift)}</dd>
                </div>
                <div className="sm:col-span-2 text-sm text-ink-muted">
                  Working: {outcome.result.dimensions} numbers each; dot product {fmt(outcome.result.dotProduct)};
                  ‖e<sub>i</sub>‖ = {fmt(outcome.result.normA)}; ‖e<sub>j</sub>‖ = {fmt(outcome.result.normB)}.
                </div>
              </dl>
            ) : (
              <p role="alert" className="mt-3 text-ink">
                {outcome.error}
              </p>
            )}
          </div>
        </div>
      </Section>

      <Section id="limitations" tone="subtle" title="Limitations of this demo">
        <div className="max-w-3xl space-y-6">
          <ul className="list-disc space-y-2 pl-6 text-ink-muted">
            <li>Inputs are arbitrary numbers you type. They are not embeddings of real faces, and no image is uploaded.</li>
            <li>No trained model is connected, so there is no face detection, alignment or embedding extraction.</li>
            <li>The output is a drift score only. It makes no verification decision and no re-enrollment recommendation.</li>
            <li>The score is a representation-change measure, not a recognition error (Section IV-B).</li>
          </ul>
          <Callout title="About the paper’s threshold">
            <p>
              The paper reports a verification boundary at drift {studyFacts.driftBoundary}, chosen on a validation split of
              real MORPH embeddings and described as not a validated re-enrollment threshold. It is deliberately not applied
              to the toy values on this page.
            </p>
          </Callout>
          <div className="flex flex-wrap gap-3">
            <ButtonLink to="/results">View reported results</ButtonLink>
            <ButtonLink to="/methodology" variant="secondary">
              Read the methodology
            </ButtonLink>
          </div>
        </div>
      </Section>
    </>
  )
}

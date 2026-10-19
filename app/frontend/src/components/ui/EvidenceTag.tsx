export type EvidenceKind = 'reported' | 'interpretation' | 'limitation' | 'future' | 'explainer'

const labels: Record<EvidenceKind, string> = {
  reported: 'Reported result',
  interpretation: 'Interpretation',
  limitation: 'Limitation',
  future: 'Future work, not validated',
  explainer: 'Plain-language explanation (added)',
}

const styles: Record<EvidenceKind, string> = {
  reported: 'border-accent bg-accent-soft text-accent-strong',
  interpretation: 'border-line bg-surface text-ink-muted',
  limitation: 'border-ink-muted bg-page text-ink-muted',
  future: 'border-dashed border-ink-muted bg-page text-ink-muted',
  explainer: 'border-line bg-page text-ink-muted',
}

/**
 * Labels a statement so results, interpretations, limitations and future work
 * are never visually conflated.
 */
export function EvidenceTag({ kind }: { kind: EvidenceKind }) {
  return (
    <span
      className={`inline-block rounded-control border px-2 py-0.5 text-label uppercase ${styles[kind]}`}
    >
      {labels[kind]}
    </span>
  )
}

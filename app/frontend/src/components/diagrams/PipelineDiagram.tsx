import type { PipelineStage } from '../../data/researchContent.ts'

/**
 * Ordered workflow after paper Fig. 1. Rendered as a semantic list so it is
 * readable by assistive technology and wraps cleanly on small screens.
 */
export function PipelineDiagram({ stages }: { stages: PipelineStage[] }) {
  return (
    <ol className="mx-auto flex max-w-md flex-col items-stretch">
      {stages.map((stage, index) => (
        <li key={stage.label} className="flex flex-col items-center">
          <div
            className={`w-full rounded-control border px-4 py-2.5 text-center text-sm font-medium ${
              stage.future
                ? 'border-dashed border-ink-muted bg-page text-ink-muted'
                : 'border-line bg-page text-ink shadow-card'
            }`}
          >
            {stage.label}
            {stage.future && <span className="block text-caption font-normal">Future work, not validated</span>}
          </div>
          {index < stages.length - 1 && (
            <span aria-hidden="true" className="py-1 text-ink-muted">
              ↓
            </span>
          )}
        </li>
      ))}
    </ol>
  )
}

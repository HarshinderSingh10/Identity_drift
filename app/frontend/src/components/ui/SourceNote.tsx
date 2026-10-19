/** Small provenance line so on-page text can be checked against the PDF. */
export function SourceNote({ source }: { source: string }) {
  return <p className="mt-2 text-caption text-ink-muted">Paper: {source}</p>
}

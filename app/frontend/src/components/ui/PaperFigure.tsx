interface PaperFigureProps {
  /** File name under public/figures/. */
  file: string
  alt: string
  caption: string
  source: string
  width: number
  height: number
}

/** A figure reproduced as printed in the paper, with caption and provenance. */
export function PaperFigure({ file, alt, caption, source, width, height }: PaperFigureProps) {
  return (
    <figure className="min-w-0">
      <div className="overflow-hidden rounded-card border border-line bg-page p-2 shadow-card">
        <img
          src={`${import.meta.env.BASE_URL}figures/${file}`}
          alt={alt}
          width={width}
          height={height}
          loading="lazy"
          className="mx-auto h-auto max-w-full"
        />
      </div>
      <figcaption className="mt-2 text-caption text-ink-muted">
        <span className="font-medium text-ink">{caption}</span> Reproduced from the paper ({source}).
      </figcaption>
    </figure>
  )
}

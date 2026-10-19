interface DataTableProps {
  caption: string
  columns: string[]
  rows: string[][]
  /** Where the numbers come from, shown beneath the table. */
  source?: string
}

/** Accessible, horizontally scrollable table. The first column is the row header. */
export function DataTable({ caption, columns, rows, source }: DataTableProps) {
  return (
    <figure className="min-w-0">
      {/* tabIndex lets keyboard users scroll the table on narrow screens. */}
      <div
        className="overflow-x-auto rounded-card border border-line bg-page shadow-card"
        role="region"
        aria-label={caption}
        tabIndex={0}
      >
        <table className="w-full border-collapse text-left text-sm">
          <caption className="sr-only">{caption}</caption>
          <thead className="bg-surface">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  scope="col"
                  className="whitespace-nowrap border-b border-line px-4 py-3 text-label uppercase text-ink-muted"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={`${row[0]}-${rowIndex}`} className="border-b border-line last:border-b-0">
                {row.map((cell, index) =>
                  index === 0 ? (
                    <th key={index} scope="row" className="px-4 py-3 font-medium text-ink">
                      {cell}
                    </th>
                  ) : (
                    <td key={index} className="px-4 py-3 tabular-nums text-ink-muted">
                      {cell}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className="mt-2 text-caption text-ink-muted">
        <span className="font-medium text-ink">{caption}.</span>
        {source ? ` ${source}` : ''}
      </figcaption>
    </figure>
  )
}

import { Link } from 'react-router-dom'

/** Inline numeric citation, e.g. [3], linking to the reference list on the Team page. */
export function Cite({ n }: { n: number }) {
  return (
    <Link
      to={`/team#ref-${n}`}
      className="rounded-control text-accent hover:underline"
      aria-label={`Reference ${n}`}
    >
      [{n}]
    </Link>
  )
}

export function Cites({ numbers }: { numbers: number[] }) {
  return (
    <span className="whitespace-nowrap">
      {numbers.map((n, i) => (
        <span key={n}>
          {i > 0 && ' '}
          <Cite n={n} />
        </span>
      ))}
    </span>
  )
}

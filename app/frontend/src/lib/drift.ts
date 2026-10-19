/**
 * Drift arithmetic used by the demo. Implements the paper's Eq. (1) and (2):
 *   sim(eᵢ, eⱼ) = eᵢᵀeⱼ / (‖eᵢ‖‖eⱼ‖),   drift = 1 − sim(eᵢ, eⱼ).
 * Pure functions, no model and no image processing.
 */

export const MAX_DIMENSIONS = 512

export type ParseResult = { ok: true; values: number[] } | { ok: false; error: string }

export function parseVector(input: string, name: string): ParseResult {
  const tokens = input.split(/[\s,;]+/).filter((t) => t.length > 0)
  if (tokens.length === 0) return { ok: false, error: `${name} is empty.` }
  const values: number[] = []
  for (const token of tokens) {
    const n = Number(token)
    if (!Number.isFinite(n)) return { ok: false, error: `${name} contains “${token}”, which is not a number.` }
    values.push(n)
  }
  if (values.length < 2) return { ok: false, error: `${name} needs at least 2 numbers.` }
  if (values.length > MAX_DIMENSIONS) return { ok: false, error: `${name} has more than ${MAX_DIMENSIONS} numbers.` }
  return { ok: true, values }
}

export const dot = (a: number[], b: number[]): number => a.reduce((sum, v, i) => sum + v * b[i], 0)
export const norm = (a: number[]): number => Math.sqrt(dot(a, a))

export interface DriftResult {
  dimensions: number
  dotProduct: number
  normA: number
  normB: number
  similarity: number
  drift: number
}

export type DriftOutcome = { ok: true; result: DriftResult } | { ok: false; error: string }

export function computeDrift(a: number[], b: number[]): DriftOutcome {
  if (a.length !== b.length) {
    return { ok: false, error: `The vectors have different lengths (${a.length} and ${b.length}); they must match.` }
  }
  if (a.length === 0 || !a.every(Number.isFinite) || !b.every(Number.isFinite)) {
    return { ok: false, error: 'Both vectors must contain only finite numbers.' }
  }
  // Rescale each vector by its largest magnitude before taking norms. Cosine similarity does not depend on a vector's
  // length, and this avoids false "zero vector" results from underflow (e.g. 1e-200) and Infinity from overflow (e.g. 1e200).
  const scaleA = Math.max(...a.map(Math.abs))
  const scaleB = Math.max(...b.map(Math.abs))
  if (scaleA === 0 || scaleB === 0) {
    return { ok: false, error: 'A vector of all zeros has no direction, so cosine similarity is undefined.' }
  }
  const unitA = a.map((v) => v / scaleA)
  const unitB = b.map((v) => v / scaleB)
  const rawSimilarity = dot(unitA, unitB) / (norm(unitA) * norm(unitB))
  const dotProduct = dot(a, b)
  const normA = norm(a)
  const normB = norm(b)
  if (!Number.isFinite(rawSimilarity) || !Number.isFinite(dotProduct) || !Number.isFinite(normA) || !Number.isFinite(normB)) {
    return { ok: false, error: 'These values are too large or too small to compute reliably. Use numbers of a more moderate size.' }
  }
  // Clamp only to absorb floating-point error just outside [-1, 1].
  const similarity = Math.min(1, Math.max(-1, rawSimilarity))
  return {
    ok: true,
    result: { dimensions: a.length, dotProduct, normA, normB, similarity, drift: 1 - similarity },
  }
}

export interface Preset {
  id: string
  label: string
  a: string
  b: string
  note: string
}

/** Synthetic toy vectors. They are NOT embeddings of any real face. */
export const presets: Preset[] = [
  { id: 'same', label: 'Same direction', a: '1, 2, 3, 4', b: '2, 4, 6, 8', note: 'Same direction, different length: similarity 1, drift 0.' },
  { id: 'small', label: 'Small change', a: '0.5, 0.5, 0.5, 0.5', b: '0.55, 0.45, 0.5, 0.5', note: 'A slight rotation away from the first vector.' },
  { id: 'orthogonal', label: 'Unrelated directions', a: '1, 0, 0, 0', b: '0, 1, 0, 0', note: 'Perpendicular vectors: similarity 0, drift 1.' },
  { id: 'opposite', label: 'Opposite directions', a: '1, 0', b: '-1, 0', note: 'Opposite vectors: similarity −1, drift 2 (the largest possible value).' },
]

import assert from 'node:assert/strict'
import { test } from 'node:test'
import { computeDrift, parseVector, presets } from '../src/lib/drift.ts'

const near = (a: number, b: number, e = 1e-9) => Math.abs(a - b) < e
const run = (a: string, b: string) => {
  const x = parseVector(a, 'A')
  if (!x.ok) return x
  const y = parseVector(b, 'B')
  if (!y.ok) return y
  return computeDrift(x.values, y.values)
}

test('every preset computes a finite drift in [0, 2]', () => {
  for (const p of presets) {
    const r = run(p.a, p.b)
    assert.ok(r.ok, p.id)
    if (r.ok) assert.ok(Number.isFinite(r.result.drift) && r.result.drift >= 0 && r.result.drift <= 2, p.id)
  }
})

test('known values: same direction 0, orthogonal 1, opposite 2', () => {
  const same = run('1,2,3,4', '2,4,6,8')
  const orth = run('1,0,0,0', '0,1,0,0')
  const opp = run('1,0', '-1,0')
  assert.ok(same.ok && near(same.result.drift, 0))
  assert.ok(orth.ok && near(orth.result.drift, 1))
  assert.ok(opp.ok && near(opp.result.drift, 2))
})

test('rejects invalid input with clear messages', () => {
  const cases: Array<[string, string, RegExp]> = [
    ['1,2,3', '1,2', /different lengths/],
    ['0,0', '1,2', /zeros/],
    ['', '1,2', /empty/],
    ['1,abc', '1,2', /not a number/],
    ['1', '1', /at least 2/],
    ['1e999,1', '1,2', /not a number/],
    ['NaN,1', '1,2', /not a number/],
    [Array(513).fill('1').join(','), Array(513).fill('1').join(','), /more than 512/],
  ]
  for (const [a, b, re] of cases) {
    const r = run(a, b)
    assert.ok(!r.ok && re.test(r.error), `${a.slice(0, 20)} | ${b.slice(0, 20)}`)
  }
})

test('extreme magnitudes never produce NaN or a false zero-vector error', () => {
  const big = run('1e200,1e200', '1e200,-1e200')
  const tiny = run('1e-200,1e-200', '1e-200,1e-200')
  const huge = run('1e200,1e200', '1e200,1e200')
  for (const r of [big, tiny, huge]) {
    if (r.ok) assert.ok(Number.isFinite(r.result.drift))
    else assert.match(r.error, /too large or too small/)
  }
  assert.ok(tiny.ok && near(tiny.result.drift, 0))
})

test('separators and repeated commas parse', () => {
  assert.ok(run('1;2;3', '3 2 1').ok)
  assert.ok(run('1, 2,, 3', '3, 2, 1').ok)
})

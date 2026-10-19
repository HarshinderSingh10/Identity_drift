# Update 1: validation and stability report

Scope: validation and stability only. No redesign, no new pages or features, no research values changed.

## 0. What could and could not be run in this environment

The sandbox had **no network access** (`registry.npmjs.org` did not resolve), so `npm install` produced no `node_modules`.
Consequently these checks were **not executed** and must be run locally: `npm run build`, `npm run typecheck`, `npm run lint`,
any browser/route/responsive/keyboard test, and a live visual check of the charts.
The **main research repository ZIP was not provided**, only the frontend ZIP, so only the frontend was inspected.

## 1. Commands actually run, and their real results

| Check | Result |
|---|---|
| `npm install` | Did not install anything (no network). Not verified. |
| `npm run build` / `typecheck` / `lint` | **Not run** (dependencies unavailable). |
| Partial `tsc -p tsconfig.app.json --noEmit` with the global TypeScript 6.0.3 | Only environmental errors remained (missing `@types/react`, `react-router-dom`, Vite/CSS typings). No errors outside that class. This is **not** a substitute for a real typecheck, because JSX is untyped without React types. |
| `npm run test` (new, `node:test`) | 5 tests, 5 passed, on Node 22.22.2. |
| Ad-hoc calculator script (21 assertions, same logic) | 21 passed. |
| Duplicate-React-key scan of all data arrays used as keys | No duplicates. |
| Local asset check | `public/paper/IEEE_major_project.pdf` starts with `%PDF-`; 4 figure PNGs exist and their real pixel sizes (1400×967, 1400×1052, 1400×988, 1400×1104) match the `width`/`height` props in `Results.tsx`. |

## 2. Routes (from `src/App.tsx`, by reading code only; not browser-tested)

`/` · `/research` · `/methodology` · `/results` · `/demo` · `/team` · `*` (NotFound), all inside `PageLayout` (skip link, Navbar, `<main>`, Footer).
All internal links (`to=` / `<Link>` / `ButtonLink`, nav and footer items) point to those six paths. Direct navigation and refresh work in `vite dev`/`vite preview`
(SPA fallback). A static host needs a rewrite to `index.html` (already noted in README). **Not browser-verified.**

## 3. Confirmed problems and fixes

| # | Confirmed issue (verified in source/by test) | Fix | File |
|---|---|---|---|
| 1 | Calculator: values that underflow (e.g. `1e-200`) were reported as a "zero vector"; values that overflow (e.g. `1e200`) could yield `NaN`/`Infinity` outputs. | Rescale by max magnitude before computing cosine; return a clear error if any result is non-finite; explicit finite-input guard. Normal inputs give identical results. | `src/lib/drift.ts` |
| 2 | Charts: `tickStep <= 0` or non-finite `max` caused an **infinite loop** in tick generation; empty `data` produced `Infinity`/`NaN` geometry; negative or `NaN` values produced negative bar sizes. | Guarded tick generation (bounded), friendly empty-state message, values clamped to `[0, max]`. Current page data is unaffected. | `src/components/charts/BarCharts.tsx` |
| 3 | Duplicate-key risk: chart groups keyed by label only, table rows keyed by first cell only (no duplicates today, but fragile). | Keys now include the index. | `BarCharts.tsx`, `DataTable.tsx` |
| 4 | Contrast: placeholder "GitHub (coming soon)" used `text-ink-muted/70` (about 3.6:1 on white, below 4.5:1). | Uses `text-ink-muted` (about 7.6:1). | `Navbar.tsx` |
| 5 | No test script existed, although `docs/RESEARCH_ACCURACY_AUDIT.md` refers to "the demo's 13 unit tests"; no test files are in the ZIP. | Added `tests/drift.test.ts` and `npm run test` (no new dependency). | `tests/`, `package.json` |

Not changed (checked and found fine): routing, skip link and focus handling, mobile menu (Escape, `aria-expanded`), form labels on both textareas,
`aria-live` output with `role="alert"` errors, table scroll regions, chart `<title>`/`<desc>`, figure `alt` text, reduced-motion CSS, relative asset paths via `BASE_URL`.

## 4. Calculator checks (logic verified by tests; UI not browser-tested)

Empty, non-numeric, `NaN`, `Infinity`-like, single-number and over-512-length inputs are rejected with messages; length mismatch and all-zero vectors are reported;
mixed separators work; identical vectors give drift of about 0; range is [0, 2]. It remains labelled synthetic and illustrative; no model, upload, backend or database.
Note: there is no dedicated "Reset" button. The four example buttons act as the reset path. Adding one was out of scope.

## 5. Checks that could not be completed

Install/build/typecheck/lint; browser rendering of every route; refresh/direct-URL behaviour on a real server; mobile and desktop width checks;
keyboard-tab walk-through; measured colour contrast beyond the one computed case; live chart rendering; external-URL checks (the site has none besides the optional, currently `null`, GitHub URL).

## 6. Remaining known issues (not changed)

- On very narrow screens the SVG charts scale down uniformly (viewBox 640 wide), so chart text becomes small (about 6 px at 320 px width). Fixing it means a layout change (scrollable chart or per-breakpoint fonts); left for a deliberate decision.
- The calculator shows norms of about 1e-200 as `0.000000` because of fixed 6-decimal formatting; the similarity is still correct.
- `usePageTitle.ts` imports `'../config/site'` without the `.ts` extension, unlike the rest of the project. It works with `moduleResolution: bundler`; left unchanged.

## 7. Research discrepancies for the next update (recorded, not changed)

Carried from `docs/RESEARCH_ACCURACY_AUDIT.md`, none resolvable without the authors' raw outputs:
Table III FRR not an integer fraction of Table II counts (Table IV implies 53/12,289 = 0.43% vs 0.007270); 84,133 vs 84,137 pair totals; 50,013 processed vs 50,015 images;
scope (all pairs vs reference drift, all splits vs test) of the Section V-A/V-B statistics; Table V outcome and PR-AUC/Brier definitions, and ISI identical to baseline (correlation 1.0);
per-drift-bin figures behind Section V-E and the abstract; derivation of the 0.361454629 trajectory threshold, "Stage 8 bootstrap", and reference [7].
New this update: the audit's mention of 13 unit tests could not be matched to any test file in the provided ZIP.

## 8. Run locally

```bash
cd frontend
npm install
npm run typecheck
npm run lint
npm run test
npm run build
npm run dev        # http://localhost:5173
npm run preview    # serves the production build
```

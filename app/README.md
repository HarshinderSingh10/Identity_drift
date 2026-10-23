# Identity Drift Research Showcase

Standalone React + TypeScript + Vite research frontend. It reads no biometric source data in the browser.

## Run

```powershell
cd app
npm install
npm run dev
```

Build the production site with `npm run build`.

`npm run data:build` creates `public/research/research-data.json` by aggregating the Stage 6 identity-level drift file into histogram bins, and copies the approved paper figures/PDF. Raw images, embeddings, and individual identity rows are not published.

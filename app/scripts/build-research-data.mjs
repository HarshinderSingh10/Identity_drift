import { readFile, mkdir, writeFile, copyFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
const root = resolve(import.meta.dirname, '../..'); const out = resolve(import.meta.dirname, '../public/research');
const csv = async p => { const [h,...rows]=(await readFile(resolve(root,p),'utf8')).trim().split(/\r?\n/); const heads=h.split(','); return rows.map(r=>Object.fromEntries(heads.map((x,i)=>[x,r.split(',')[i]]))); };
const numbers = rows => rows.map(r=>Number(r.mean_drift)).filter(Number.isFinite);
const values=numbers(await csv('results/metrics/stage6_identity_stability.csv')); const min=Math.min(...values), max=Math.max(...values), bins=Array.from({length:32},(_,i)=>({start:min+i*(max-min)/32,end:min+(i+1)*(max-min)/32,count:0})); values.forEach(v=>bins[Math.min(31,Math.floor((v-min)/(max-min)*32))].count++);
const take = async p => csv(p);
await mkdir(out,{recursive:true});
const data={
  generatedFrom:'Aggregate transformation of local Stage 6 identity-level data; no identities, images, or embeddings are published.',
  identityHistogram: bins.map(b=>({...b,mid:(b.start+b.end)/2})),
  split:await take('results/metrics/research_split_summary.csv'), ageGapDrift:await take('results/metrics/stage6_age_gap_drift_summary.csv'),
  verification:await take('results/metrics/stage7_verification_summary.csv'), ageGapVerification:(await take('results/metrics/stage7_age_gap_recognition.csv')).filter(x=>x.research_split==='research_test'),
  roc:await take('results/metrics/stage7_test_roc.csv'), isi:await take('results/metrics/stage8_model_performance.csv')
};
await writeFile(resolve(out,'research-data.json'),JSON.stringify(data));
for (const f of ['fig1_pipeline.png','fig2_longitudinal_concept.png','fig3_identity_drift_distribution.png','fig4_age_gap_drift.png','fig5_age_gap_verification.png','fig6_drift_verification_boundary.png','fig7_longitudinal_trajectories.png']) await copyFile(resolve(root,'paper/figures',f),resolve(out,f));
await copyFile(resolve(root,'paper/output/main.pdf'),resolve(out,'paper.pdf'));

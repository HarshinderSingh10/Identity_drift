# Stage 5 — Embedding Extraction

## Objective

Extract fixed InsightFace `buffalo_l` ArcFace representations from successful
Stage 4 aligned MORPH images. This stage does not calculate drift, thresholds,
verification metrics, Stability Index values, or re-enrollment decisions.

## Configuration

- Version: `stage5-v1`
- Model: `InsightFace buffalo_l ArcFace`
- Input: Stage 4 aligned `112x112` BGR images
- Embedding dimension: `512`
- Dtype: `float32`
- Normalization: `L2`
- Execution provider: `CUDAExecutionProvider, CPUExecutionProvider`

## Extraction results

- Stage 4 input images: `50015`
- Successful Stage 4 images: `50013`
- Attempts: `50013`
- Successful embeddings: `50013`
- Failed embeddings: `0`
- Cache hits: `0`
- New embeddings: `50013`
- Elapsed seconds: `757.996`

## Traceability and safeguards

Canonical per-image embeddings are stored under `embeddings/morph/embeddings/`
and mapped through `data/metadata/morph_embedding_metadata.csv`. Failed Stage 4
images are excluded by status rather than deleted from Stage 4 metadata.
Embedding extraction does not fit downstream parameters or use test identities
for tuning.

## Reproducibility

The manifest records package versions, provider configuration, Python version,
timestamp, and Git commit when available. The required smoke test verified
512-dimensional float32 outputs, finite values, L2 norms near one, and repeated
inference reproducibility before the full run.

## Readiness for Stage 6

Embeddings are numerical representations only. Drift analysis remains a
separate Stage 6 operation governed by the frozen `stage3-v1` protocol.

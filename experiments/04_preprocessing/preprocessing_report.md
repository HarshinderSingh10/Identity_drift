# Stage 4 - MORPH Face Preprocessing

## Scope

This stage performed deterministic image loading, face detection, landmark alignment, and processed-image storage. It did not call the ArcFace recognition inference API, calculate embeddings, calculate cosine similarity, calculate drift, run verification, choose thresholds, build a Stability Index, or implement re-enrollment.

## Configuration

- Version: `stage4-v1`
- Detector/model package: `buffalo_l`, detection module only
- Execution provider: `CPUExecutionProvider` (CUDA/cuDNN was unavailable on this host)
- Detector input size: `160x160`
- Alignment: InsightFace `face_align.norm_crop`, `mode="arcface"`
- Output size: `112x112`
- Input/output array convention: OpenCV BGR, `uint8`, pixel range 0-255
- ArcFace recognition normalization: not applied here; model-internal preprocessing remains for Stage 5
- Multiple-face policy: no face is selected; status is `multiple_faces`

## Results

- Total images: 50015
- Successful: 50013
- Failed: 2
- Failure rate: 0.00003999
- Zero-face images: 2
- Single-face images: 50013
- Multiple-face images: 0
- Eligible longitudinal pairs: 84133
- Ineligible longitudinal pairs: 4

## Traceability and eligibility

Every MORPH image has a preprocessing metadata row, including failures. Processed paths preserve the original `Train`, `Validation`, or `Test` relative path under `data/processed/morph/images/`. The original Stage 2 pair file was not modified; pair eligibility is written separately.

## Split integrity

The preprocessing transformation does not reassign identities or images. The Stage 2 research split assignment is copied unchanged, and no preprocessing rule was tuned using test-set statistics.

## Raw-data and embedding safeguards

Raw MORPH, AgeDB images, and AgeDB protocol files were treated as read-only. No new `.npy` or `.npz` embedding files were created, and no embedding or similarity computation was performed.

# Identity Drift Detection and Re-Enrollment System

This project investigates how facial recognition embeddings change across temporal and environmental variations and develops a drift-aware mechanism for determining when an identity should be re-enrolled.

## Main Objectives

- Generate facial embeddings using a pretrained ArcFace-based model.
- Measure embedding drift between images of the same identity.
- Study temporal and environmental sources of embedding variation.
- Develop an Identity Stability Index.
- Develop a data-driven re-enrollment policy.
- Evaluate the proposed approach experimentally.

## Planned Datasets

- AgeDB-30
- CALFW
- CPLFW
- LFW

## Planned Pipeline

```text
Face Image
    |
Face Detection / Alignment
    |
ArcFace Embedding
    |
512-D Embedding
    |
Drift Calculation
    |
Stability Analysis
    |
Re-Enrollment Decision
```

## Planned Metrics

- Cosine similarity
- Embedding drift
- Intra-identity similarity
- Inter-identity similarity
- Identity Stability Index
- Verification metrics such as ROC-AUC, TAR, FAR where appropriate

## Planned Experiments

- Temporal/age variation
- Pose variation
- Lighting variation
- Expression variation
- Image quality variation

## Technology Stack

- Python
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Matplotlib
- PyTorch/ONNX Runtime or the appropriate ArcFace implementation
- InsightFace where appropriate

## Project Rules

- Do not train a face recognition model from scratch.
- Use a pretrained ArcFace-based model.
- The main research contribution is embedding drift analysis and re-enrollment, not model training.
- Keep components modular so they can be tested independently.
- Do not create a GUI yet.
- Do not download datasets or models yet.
- Do not invent experimental results.


# Identity Drift Detection and Adaptive Re-enrollment

> **A research project for measuring temporal changes in face-recognition embeddings and studying their impact on biometric verification and identity stability.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Face%20Recognition-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-ArcFace-green)
![Research](https://img.shields.io/badge/Project-Research-purple)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

---

## 📌 Overview

Face-recognition systems typically assume that a person's facial representation remains sufficiently stable over time. However, facial embeddings can change due to factors such as:

* Age progression
* Pose variation
* Facial expression
* Lighting conditions
* Image quality and compression
* Camera differences
* Long-term temporal changes

This project investigates **Identity Drift** — the measurable change in a person's facial embedding over time.

Instead of treating face recognition as a static problem, this work studies the **trajectory of an individual's embedding across multiple observations** and investigates whether increasing embedding drift is associated with degradation in recognition performance.

The project further introduces an **identity stability analysis** and an **adaptive re-enrollment policy** intended to determine when an existing biometric reference may need to be updated.

---

# 🎯 Research Objectives

The project focuses on five major objectives:

1. **Extract facial embeddings** from longitudinal face images.
2. **Measure identity drift** between observations of the same individual.
3. **Study the relationship between drift and recognition performance.**
4. **Quantify individual identity stability over time.**
5. **Develop a drift-aware re-enrollment policy** for maintaining biometric references.

---

# 🔬 Research Pipeline

```text
                 Longitudinal Face Dataset
                           │
                           ▼
                 Dataset Preparation
                           │
                           ▼
                  Face Preprocessing
                           │
                           ▼
                 Embedding Extraction
                           │
                           ▼
              Temporal Embedding Analysis
                           │
                           ▼
                    Identity Drift
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Recognition Analysis       Stability Analysis
              │                         │
              └────────────┬────────────┘
                           ▼
                Adaptive Re-enrollment
                           │
                           ▼
                 Evaluation & Results
```

---

# 🧠 Core Concepts

## 1. Face Embedding

A face image is converted into a numerical vector representation using a face-recognition model.

The embedding captures facial characteristics in a representation suitable for similarity comparison.

---

## 2. Identity Drift

Identity drift represents the change in facial embedding space across observations of the same individual.

For two normalized embeddings:

```text
Drift = 1 − Cosine Similarity
```

A larger drift value indicates a larger change between the two representations.

The project evaluates both:

* **Consecutive drift** — change between temporally adjacent observations.
* **Reference drift** — change relative to a selected reference embedding.

---

## 3. Temporal Drift

Rather than evaluating only isolated image pairs, the project analyzes longitudinal observations.

```text
Person A

Observation 1 ──► Observation 2 ──► Observation 3 ──► Observation 4
      │                  │                  │                  │
   Embedding          Embedding          Embedding          Embedding
      │                  │                  │                  │
      └──────────────────┴──────────────────┴──────────────────┘
                         │
                  Drift trajectory
```

This allows the project to study whether embedding changes accumulate or fluctuate over time.

---

## 4. Recognition Performance

The project investigates whether increasing identity drift corresponds to changes in verification performance.

The evaluation includes analysis of:

* Genuine similarity
* Impostor similarity
* Verification error
* False rejection rate
* ROC curves
* Acceptance behaviour
* Drift versus verification performance

---

## 5. Identity Stability

A stability analysis is used to characterize how consistently an individual's embedding behaves over time.

The project evaluates:

* Individual drift distributions
* Population-level drift
* Stability scores
* Future recognition error
* Persistence and recovery behaviour
* Pre-failure warning behaviour

---

## 6. Adaptive Re-enrollment

The final stage explores a **drift-aware re-enrollment strategy**.

Instead of periodically replacing biometric references without considering identity behaviour, the system can use observed drift and stability information to determine when an identity may require re-enrollment.

Conceptually:

```text
New Observation
      │
      ▼
Extract Embedding
      │
      ▼
Calculate Drift
      │
      ▼
Evaluate Stability
      │
      ├───────────────┐
      │               │
      ▼               ▼
 Stable Identity   Significant Drift
      │               │
      ▼               ▼
 Keep Reference   Evaluate Re-enrollment
                      │
                      ▼
               Update Reference
```

---

# 📊 Datasets

The research pipeline is designed around longitudinal face datasets and uses datasets including:

* **AgeDB**
* **MORPH**
* Additional datasets/protocols used for verification and evaluation

The datasets themselves are **not included in this repository**.

Dataset preparation and protocol information can be found under:

```text
experiments/
```

The expected data directory structure is:

```text
data/
├── raw/
├── processed/
└── metadata/
```

Dataset files should be downloaded separately and placed according to the project documentation.

---

# 🧪 Experimental Stages

The repository is organized according to the research workflow.

### Stage 2 — Research Dataset

Dataset composition, identity distributions, age gaps and longitudinal coverage are analyzed.

### Stage 3 — Experimental Protocol

Defines the experimental setup and evaluation protocol.

### Stage 4 — Preprocessing

Images are processed and evaluated for preprocessing quality.

### Stage 5 — Embedding Extraction

Face embeddings are extracted from the processed observations.

### Stage 6 — Embedding Drift

Temporal and reference-based identity drift are analyzed.

### Stage 7 — Drift vs Recognition

The relationship between embedding drift and face verification performance is investigated.

### Stage 8 — Identity Stability

Individual stability and future recognition behaviour are analyzed.

---

# 📈 Results

The repository contains generated research visualizations covering:

### Dataset & preprocessing

* Dataset protocol distributions
* Age-span distributions
* Preprocessing success/failure
* Face-count distributions

### Identity drift

* Drift distributions
* Drift versus age gap
* Consecutive drift
* Reference drift
* Longitudinal trajectories
* Drift trajectories and slopes

### Recognition

* Genuine versus impostor similarity
* Drift versus verification error
* False rejection rate versus drift
* Acceptance behaviour
* ROC curves

### Stability

* Identity Stability Index distributions
* Individual versus population drift
* Stability versus recognition margin
* Stability versus future error
* Calibration curves
* Precision-recall curves
* Pre-failure warning analysis

Research figures are available under:

```text
results/plots/
```

---

# 📁 Project Structure

```text
Identity_drift/
│
├── app/
│   └── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── embeddings/
│
├── experiments/
│   ├── 02_research_dataset/
│   ├── 03_experimental_protocol/
│   ├── 04_preprocessing/
│   ├── 05_embedding_extraction/
│   ├── 06_embedding_drift/
│   ├── 07_drift_vs_recognition/
│   └── 08_identity_stability/
│
├── models/
│
├── notebooks/
│
├── paper/
│   ├── figures/
│   ├── main.tex
│   ├── references.bib
│   └── output/
│
├── results/
│   └── plots/
│
├── src/
│   ├── data/
│   ├── drift/
│   ├── embedding/
│   ├── experiments/
│   ├── reenrollment/
│   └── stability/
│
├── tests/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/HarshinderSingh10/Identity_drift.git
cd Identity_drift
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The project provides experiment modules under:

```text
src/experiments/
```

and the main entry point:

```text
main.py
```

Individual research stages can be executed using the corresponding experiment scripts.

For example:

```bash
python main.py
```

Refer to the individual experiment reports and module documentation for stage-specific execution instructions.

---

# 🧪 Testing

The repository includes tests for the major components.

Run:

```bash
pytest
```

Tests cover areas including:

* Dataset loading
* Preprocessing
* Embedding extraction
* Drift metrics
* Drift analysis
* Recognition analysis
* Stability analysis
* Verification experiments
* Identity auditing

---

# 📚 Research Paper

The LaTeX source for the research paper is available under:

```text
paper/
```

The repository includes:

* IEEE-style LaTeX source
* Bibliography
* Research figures
* Generated research paper PDF

**Paper:** `paper/output/main.pdf`

---

# 🔐 Data & Privacy

This repository intentionally does **not** contain:

* Raw face datasets
* Processed face datasets
* Face embeddings
* Model weights
* Private credentials
* Environment variables
* Generated experiment data containing sensitive information

These files are excluded through `.gitignore`.

Users should obtain datasets directly from their respective official sources and comply with their licensing and research-use requirements.

---

# 🚧 Project Status

**Status: Research Prototype**

The project is currently focused on experimental evaluation of identity drift, recognition degradation, identity stability and drift-aware re-enrollment.

Future development may include:

* Improved adaptive re-enrollment strategies
* More longitudinal datasets
* Additional environmental factors
* Cross-model comparison
* Real-time drift monitoring
* Production-oriented biometric monitoring
* Extended security evaluation

---

# 👨‍💻 Authors

**Harshinder Singh**
Department of Computer Science and Engineering
ABES Institute of Technology, Ghaziabad, India

**Gyan Sharma**
Department of Computer Science and Engineering
ABES Institute of Technology, Ghaziabad, India

**Devansh Goel**
Department of Computer Science and Engineering
ABES Institute of Technology, Ghaziabad, India

**Siddhant Nirwal**
Department of Computer Science and Engineering
ABES Institute of Technology, Ghaziabad, India

---

# 📄 License

This project is intended for **academic and research purposes**.

Please review the licenses and usage conditions of any external datasets, pretrained models and third-party libraries before using them.

---


```

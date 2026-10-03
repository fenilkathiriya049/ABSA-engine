# Aspect-Based Sentiment Analysis (ABSA) Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com)
[![spaCy](https://img.shields.io/badge/spaCy-3.7.0-09A3D5.svg)](https://spacy.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A linguistically grounded, end-to-end NLP engine for **Aspect-Based Sentiment Analysis (ABSA)**. Unlike naive document-level or sentence-level sentiment classifiers that blend conflicting polarities into a single score, this system decomposes reviews into fine-grained target entities and classifies polarity conditioned on the target's syntactic context.

The engine addresses two core NLP subtasks formulated in **SemEval-2014 Task 4**:
1. **Aspect Term Extraction (ATE):** Formulated as a token-level sequence labeling task using **Conditional Random Fields (CRF)** with rich morphological, orthographic, and local syntactic context features.
2. **Aspect Sentiment Classification (ASC):** Formulated as a graph-traversal problem over **spaCy Dependency Trees**, isolating shortest-path adjectival modifiers and explicit negation scopes to train a regularized linear classifier.

---

## Architecture Overview

                   [ User Input Text ]
                           │
                           ▼
           [ spaCy Dependency Parsing & POS ]
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
[ Sequence Feature Window ]     [ Syntactic Dependency Graph ]
           │                               │
           ▼                               ▼
[ CRF Sequence Tagger ]         [ Modifier & Negation Traversal ]
           │                               │
           ▼                               │
  Extracted Aspect Spans                   │
  ("crust", "staff")                       │
           │                               │
           └───────────────┬───────────────┘
                           ▼
        [ Feature Vectorizer & ASC Classifier ]
                           │
                           ▼
            { "term": "crust", "sentiment": "positive" }
            { "term": "staff", "sentiment": "negative" }
                           │
                           ▼
             [ FastAPI REST Response / JSON ]

---

## Mathematical & Linguistic Formulations

### 1. Aspect Term Extraction (CRF)
ATE is modeled as sequence tagging where each token $x_t$ is mapped to a label $y_t \in \{\text{B-ASP}, \text{I-ASP}, \text{O}\}$. A linear-chain Conditional Random Field models the conditional distribution:

$$P(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp \left( \sum_{t=1}^T \sum_k \lambda_k f_k(y_{t-1}, y_t, \mathbf{x}, t) \right)$$

Where:
* $f_k(y_{t-1}, y_t, \mathbf{x}, t)$ represents transition and emission feature functions (POS tags, word shapes, prefix/suffix n-grams, context windows).
* $Z(\mathbf{x})$ is the partition function calculated via the Forward-Backward algorithm.
* Regularization is optimized via L-BFGS with elastic net penalties ($c_1=0.1, c_2=0.1$).

### 2. Syntactic Dependency & Negation Traversal (ASC)
Given an aspect token $A$ and sentence dependency tree $G = (V, E)$, modifiers are extracted via directional edge traversal:
* **Adjectival Complements (`acomp`):** Identifies copular predicates where $A \xleftarrow{\text{nsubj}} V_{\text{cop}} \xrightarrow{\text{acomp}} \text{Adj}$ (e.g., *"The crust is thin"*).
* **Attribute Modifiers (`amod`):** Identifies direct adjectival dependents $A \xrightarrow{\text{amod}} \text{Adj}$ (e.g., *"Crispy crust"*).
* **Negation Scope Resolution:** Detects negation tokens $N \in \{\text{"not"}, \text{"never"}, \text{"n't"}, \text{"no"}\}$ attached to the aspect node, its parent governor, or its modifier siblings, inverting the polar feature representation.

---

## Directory Structure

```text
absa-engine/
├── artifacts/                  # Serialized model binaries (.joblib)
├── configs/                    # Centralized hyperparameter and path config
│   └── config.yaml
├── data/
│   ├── raw/                    # SemEval XML datasets
│   └── processed/              # BIO tagged JSONL & ASC CSVs
├── src/
│   ├── preprocessing/          # XML parsing, BIO tag alignment, cleaning
│   │   ├── dataset_loader.py
│   │   └── text_cleaner.py
│   ├── features/               # CRF sequence and syntactic tree extractors
│   │   ├── sequence_features.py
│   │   └── syntactic_extractor.py
│   ├── models/                 # CRF & Logistic Regression training routines
│   │   ├── ate_crf.py
│   │   └── asc_classifier.py
│   ├── evaluation/             # Span-level and polarity evaluation metrics
│   │   └── metrics.py
│   └── pipeline.py             # Unified inference orchestration class
├── api/
│   ├── schemas.py              # Pydantic v2 data contracts
│   └── app.py                  # FastAPI service with lifespan model loading
├── tests/                      # Pytest suite (syntax, pipeline, API)
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Locked project dependencies
└── README.md

Installation & Setup
1. Prerequisites
Python 3.10+ or 3.11

Git

2. Environment Setup
Clone the repository and create an isolated virtual environment:

git clone [https://github.com/](https://github.com/)<your-username>/absa-engine.git
cd absa-engine

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Execution PipelineStep 1: Preprocess DatasetsParse raw SemEval XML datasets into token-aligned sequence files (JSONL) and tabular aspect-polarity pairs (CSV):Bashpython -m src.preprocessing.dataset_loader --raw_path data/raw/Restaurants_Train.xml --output_dir data/processed
Step 2: Train ModelsTrain the CRF sequence labeler and the syntactically-grounded classifier:Bash# 1. Train CRF Sequence Tagger for ATE
python -m src.models.ate_crf

# 2. Train Logistic Regression Classifier for ASC
python -m src.models.asc_classifier
Model binaries are serialized to artifacts/crf_ate_model.joblib and artifacts/logreg_asc_pipeline.joblib.Step 3: Run Benchmark EvaluationsRun span-level F1 evaluation and multi-class polarity performance reports:Bashpython -m src.evaluation.metrics
Step 4: Run TestsRun unit, integration, and HTTP test suites:Bashpython -m pytest tests/ -v
API Deployment & DocumentationLaunch the production ASGI server with Uvicorn:Bashpython -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
Interactive Swagger UI: Visit http://127.0.0.1:8000/docsHealth Check Endpoint: GET http://127.0.0.1:8000/healthInference Example (POST /analyze)Request:Bashcurl -X POST "[http://127.0.0.1:8000/analyze](http://127.0.0.1:8000/analyze)" \
     -H "Content-Type: application/json" \
     -d '{"text": "The crust is thin and crunchy, but the staff is aloof."}'
Response:JSON{
  "text": "The crust is thin and crunchy, but the staff is aloof.",
  "total_aspects": 2,
  "aspects": [
    {
      "term": "crust",
      "sentiment": "positive",
      "confidence": 0.8921,
      "span": [4, 9],
      "syntactic_modifiers": "crunchy thin",
      "negated": false
    },
    {
      "term": "staff",
      "sentiment": "negative",
      "confidence": 0.9104,
      "span": [39, 44],
      "syntactic_modifiers": "aloof",
      "negated": false
    }
  ]
}
Benchmark ResultsEvaluated on the SemEval-2014 Task 4 Restaurant benchmark:SubtaskMetricScoreAspect Term Extraction (ATE)Macro Precision0.8412Macro Recall0.7985Macro F10.8193Aspect Sentiment Classification (ASC)Accuracy0.8240Macro F10.7816Key Highlights & Design DecisionsSyntactic Isolation over Bag-of-Words: Standard n-gram and TF-IDF models conflate polarities in contrastive sentences (e.g., "great food, terrible service"). Traversing dependency trees cleanly isolates modifiers bound to each distinct noun phrase.Deterministic BIO Alignment: Rather than relying on fuzzy string matching that drifts on duplicate terms, token character spans are mapped directly from spaCy token.idx boundaries.Production Lifespan Architecture: Models load once into memory via FastAPI's asynchronous lifespan handler, ensuring low-latency inference (~8–15ms per request on CPU) without per-request deserialization overhead.
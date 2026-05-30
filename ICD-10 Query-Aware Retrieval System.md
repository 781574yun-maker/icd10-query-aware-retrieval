# ICD-10 Query-Aware Retrieval System

A query-aware adaptive retrieval system for **ICD-10-CM medical code recommendation**, combining **TF-IDF**, **BERT semantic retrieval**, and a **Random Forest model selector** to dynamically choose the best retrieval strategy based on query characteristics.

This project was developed through an **industry–academia collaboration** to improve medical code retrieval performance using domain-specific NLP and adaptive model selection.

---

## Project Overview

Medical diagnosis texts are often ambiguous, short, multilingual, and structurally inconsistent, making accurate ICD-10-CM code retrieval challenging.

This project proposes a **query-aware adaptive retrieval framework** that dynamically selects the most suitable retrieval model based on query features.

The system integrates:

- **TF-IDF retrieval**
- **BERT semantic search**
- **Hybrid retrieval (TF-IDF + BERT)**
- **Query feature engineering**
- **Random Forest model selection**
- **Adaptive retrieval strategy**

Instead of using a single retrieval model for all inputs, the system automatically determines which model performs best for a given query.

---

## Key Features

- **BERT Semantic Retrieval**
- **TF-IDF Keyword Retrieval**
- **Hybrid Retrieval (TF-IDF + BERT)**
- **Chinese NLP Processing (jieba & spaCy)**
- **Query-Aware Adaptive Model Selection**
- **Precision@K Evaluation**
- **FastAPI Deployment**
- **ICD-10-CM Medical Code Recommendation**

---

## System Architecture

```text
Medical Query
      ↓
Feature Extraction
      ↓
Random Forest Model Selector
      ↓
┌─────────────┬────────────────┬────────────────┐
│ BERT        │ Hybrid Jieba  │ Hybrid spaCy  │
└─────────────┴────────────────┴────────────────┘
      ↓
Top-K ICD-10 Recommendations
```

---

## project Structure

```text
icd10-query-aware-retrieval/
│
├── src/
│   ├── bert_retrieval.py
│   ├── hybrid_jieba_retrieval.py
│   ├── hybrid_spacy_retrieval.py
│   ├── adaptive_hybrid_retrieval.py
│   ├── query_feature_extractor.py
│   └── train_model_selector.py
│
├── api/
│   ├── bert_api.py
│   ├── hybrid_jieba_api.py
│   ├── hybrid_spacy_api.py
│   └── adaptive_hybrid_api.py
│
├── evaluation/
│   ├── evaluate_bert.py
│   ├── evaluate_hybrid_jieba.py
│   ├── evaluate_hybrid_spacy.py
│   ├── evaluate_adaptive_hybrid.py
│   └── oracle_precision_analysis.py
│
├── data/
│   └── sample_icd_vocab.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Retrieval Models

### 1️⃣ BERT Semantic Retrieval
Uses multilingual sentence embeddings to perform semantic similarity search for ICD code recommendation.

**Model Used**
```text
sentence-transformers/distiluse-base-multilingual-cased-v1
```

---

### 2️⃣ Hybrid Retrieval (jieba + BERT)
Combines:

- **TF-IDF keyword matching**
- **BERT semantic similarity**
- **jieba Chinese tokenization**

Hybrid score:

```text
Hybrid Score = α × TF-IDF + (1 − α) × BERT
```

---

### 3️⃣ Hybrid Retrieval (spaCy + BERT)

Uses:

- **spaCy Chinese tokenization**
- **TF-IDF retrieval**
- **BERT embeddings**

Designed for longer and structurally complex diagnosis queries.

---

### 4️⃣ Adaptive Hybrid Retrieval (Core Contribution)

The system extracts query features and dynamically selects the optimal retrieval model.

Selected models include:

- **BERT**
- **Hybrid Jieba**
- **Hybrid spaCy**

Selection is performed using a **Random Forest classifier** trained on query characteristics.

---

## Query Features Used

The adaptive selector extracts features such as:

- Character length
- Token length
- Number of digits
- Body-part keywords
- Left/right side indicators
- Symptom keywords
- Post-operative indicators
- Punctuation usage
- Multi-diagnosis separators

These features help determine the most suitable retrieval strategy.

---

## Evaluation Metrics

Models were evaluated using:

- **Precision@1**
- **Precision@5**
- **Precision@10**
- **Oracle Precision**
- **Per-query model comparison**

The adaptive retrieval system achieved higher retrieval effectiveness by dynamically selecting the best-performing model.

---

### API Deployment

The system supports **FastAPI deployment**.

### Run Adaptive API

```bash
uvicorn api/adaptive_hybrid_api:app --reload --port 8003
```

Open Swagger UI:

```text
http://127.0.0.1:8003/docs
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/icd10-query-aware-retrieval.git
cd icd10-query-aware-retrieval
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠 Technologies Used

### Programming
- Python

### Machine Learning / NLP
- Scikit-learn
- Sentence Transformers
- BERT
- TF-IDF
- Random Forest

### NLP Libraries
- jieba
- spaCy

### API
- FastAPI

### Data Processing
- Pandas
- NumPy

---

## Data Privacy Notice

Due to confidentiality restrictions, **real insurance claim data and trained model files are not included** in this repository.

Only sample vocabulary data are provided for demonstration purposes.

---

## Academic Context

This project was developed as part of an **industry–academia collaboration project** focused on improving medical code recommendation systems using adaptive NLP retrieval techniques.

---

## Author

**Hua-Yun Zhang**

GitHub:
https://github.com/781574yun-maker

---

## License

This repository is intended for **academic and research purposes**.
# Anime Recommender System

A backend-focused anime recommendation system built with Python, machine learning, and modern data engineering practices.

The project combines structured anime metadata, user ratings, and external metadata enrichment from the Jikan API to build a scalable recommendation pipeline.

---

# Features

## ETL Pipeline

* Kaggle Anime Recommendations Database ingestion
* Metadata cleaning and normalization
* Genre preprocessing
* Resumable Jikan API enrichment
* Automatic checkpointing and recovery
* Retry handling for transient API failures

---

## Feature Engineering

* Merge Kaggle and Jikan datasets

* Generate unified `feature_text`

* Combine:

  * Genres
  * Themes
  * Studios
  * Demographics
  * Synopsis

* Ready for downstream machine learning models

---

## Recommendation Engine

Current recommendation model:

* TF-IDF Vectorization
* Cosine Similarity
* Joblib model persistence

Current recommendation features include:

* Genres
* Themes
* Studios
* Demographics
* Synopsis

---

# Project Structure

```text
anime-recommender/

├── api/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── etl/
│   ├── load_kaggle_ratings.py
│   ├── transform.py
│   ├── fetch_jikan.py
│   └── build_features.py
│
├── models/
│   ├── train.py
│   └── content_based.py
│
├── notebooks/
│
├── sql/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Data Pipeline

```text
Kaggle Dataset
        │
        ▼
Cleaning & Transformation
        │
        ▼
Jikan Metadata Enrichment
        │
        ▼
Feature Engineering
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Cosine Similarity
        │
        ▼
Top-N Recommendations
```

---

# Datasets

## Kaggle

* 12,294 anime
* 7.8 million user ratings

Provides:

* Genres
* Episodes
* Type
* Rating
* Popularity

---

## Jikan API

Provides additional metadata:

* Synopsis
* Themes
* Studios
* Demographics
* Season
* Year
* Score

---

# Current Workflow

1. Load Kaggle dataset
2. Clean metadata
3. Enrich records using Jikan
4. Merge datasets
5. Build feature vectors
6. Train TF-IDF model
7. Compute cosine similarity
8. Save model artifacts
9. Generate recommendations

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* Requests
* Joblib
* FastAPI *(planned)*

---

# Running

Install dependencies

```bash
pip install -r requirements.txt
```

Run ETL

```bash
python etl/load_kaggle_ratings.py
python etl/transform.py
python etl/fetch_jikan.py
python etl/build_features.py
```

Train recommendation model

```bash
python models/train.py
```

Run recommender

```bash
python models/content_based.py
```

---

# Roadmap

## Completed

* Dataset ingestion
* ETL pipeline
* Metadata cleaning
* TF-IDF recommendation engine
* Model persistence
* Resumable Jikan enrichment
* Feature engineering pipeline

## In Progress

* Full Jikan enrichment (12k+ anime)

## Planned

* Collaborative filtering
* Hybrid recommendation engine
* FastAPI REST API
* SQL analytics
* Docker deployment
* Recommendation evaluation metrics

---

# Engineering Highlights

* Modular ETL architecture
* Checkpoint-based data ingestion
* Resume support after interruption
* Model artifact persistence
* Decoupled training and inference
* Feature engineering pipeline
* Backend-oriented project structure

---

Built as a portfolio project demonstrating machine learning engineering, backend development, ETL pipelines, recommendation systems, and production-oriented Python workflows by Aadish Tamaskar
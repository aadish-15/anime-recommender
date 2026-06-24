# Anime Recommender System

A backend-focused anime recommendation system built with Python, machine learning, and data engineering principles.

The project uses anime metadata, user ratings, and external enrichment from the Jikan API to generate anime recommendations through a hybrid recommendation pipeline.

---

## Current Features

### Data Processing

* Kaggle Anime Recommendations Database ingestion
* Anime metadata cleaning and preprocessing
* Genre normalization and feature preparation
* Resumable metadata enrichment pipeline using Jikan API

### Recommendation Engine

* Content-based recommendation system
* TF-IDF vectorization
* Cosine similarity recommendation generation
* Model artifact persistence using Joblib

### Engineering Features

* Modular ETL pipeline
* Checkpointed API enrichment workflow
* Recovery from interrupted enrichment jobs
* Model serialization and loading

---

## Dataset

### Kaggle Anime Recommendations Database

Contains:

* Anime metadata
* User ratings
* Genre information
* Popularity metrics

Current dataset size:

* 12,294 anime entries
* 7.8 million user ratings

### Jikan API Enrichment

Additional metadata:

* Synopsis
* Themes
* Studios
* Demographics
* Season
* Year
* Score

---

## Project Structure

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
│   └── fetch_jikan.py
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

## Recommendation Pipeline

```text
Anime Dataset
      ↓
Data Cleaning
      ↓
Metadata Enrichment (Jikan)
      ↓
Feature Engineering
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity Matrix
      ↓
Top-N Recommendations
```

---

## Current Recommendation Method

The current version uses:

* Genre metadata
* TF-IDF feature extraction
* Cosine similarity

to recommend anime with similar content profiles.

Example:

```python
recommend("Death Note")
```

Output:

```text
Monster
Psycho-Pass
Code Geass
...
```

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Requests
* Joblib
* FastAPI (planned)

---

## Roadmap

### Completed

* Dataset ingestion
* Metadata cleaning
* TF-IDF feature generation
* Content-based recommendation engine
* Model artifact persistence
* Resumable Jikan enrichment pipeline

### In Progress

* Full metadata enrichment
* Feature engineering using synopsis, themes, studios, and demographics

### Planned

* Collaborative filtering
* Hybrid recommendation system
* FastAPI recommendation endpoints
* SQL analytics layer
* Docker deployment

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train recommendation artifacts:

```bash
python models/train.py
```

Run recommendation engine:

```bash
python models/content_based.py
```

Run metadata enrichment:

```bash
python etl/fetch_jikan.py
```

---

## Future Improvements

* Hybrid recommendation architecture
* Similarity search optimization
* Automated retraining workflows
* Recommendation quality evaluation
* API deployment and monitoring

---

## Author

Built as a machine learning and backend engineering portfolio project focused on recommendation systems, data processing, and model deployment by Aadish Tamaskar.
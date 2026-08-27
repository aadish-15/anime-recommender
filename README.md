# Anime Recommender System

A backend-focused anime recommendation system built with Python, Pandas, Scikit-learn, and data engineering practices.

The project currently implements a **content-based recommendation engine** using anime metadata from the Kaggle Anime Recommendations Database, with optional enrichment from the Jikan API.

The pipeline is designed to remain usable even when external enrichment is incomplete or temporarily unavailable.

---

## Features

### Data Ingestion

* Kaggle anime metadata ingestion
* Kaggle user ratings ingestion and validation
* Dataset schema validation
* Missing-value reporting
* Duplicate ID detection
* Referential integrity checks between anime and rating datasets

> User ratings are currently validated and retained for future collaborative filtering. They are **not yet used by the current recommendation model**.

---

### Data Cleaning

Anime metadata cleaning includes:

* Duplicate anime removal
* Invalid ID handling
* Missing title filtering
* Genre normalization
* Numeric episode conversion
* Nullable episode counts
* Rating validation
* Missing rating preservation
* Member count normalization
* Whitespace cleanup

Unknown values are preserved as missing values instead of being replaced with misleading defaults.

---

### Jikan Metadata Enrichment

The enrichment pipeline retrieves additional metadata from the Jikan API:

* Synopsis
* Themes
* Studios
* Demographics
* Genres
* Season
* Year
* Score

The Jikan pipeline includes:

* Resume support
* Checkpointing
* Retry handling
* Exponential backoff
* `Retry-After` support
* Network timeout handling
* HTTP error classification
* Failure tracking
* Atomic checkpoint writes
* Safe interruption handling with `Ctrl+C`

Successful records are saved to:

```text
data/processed/jikan_enriched.csv
```

Failed requests are tracked separately in:

```text
data/processed/jikan_failures.csv
```

Permanent failures such as invalid or missing anime IDs are skipped on future runs, while temporary failures can be retried later.

---

## Partial Jikan Enrichment Support

The recommendation pipeline does **not require complete Jikan enrichment to run**.

Two feature modes are supported.

### Baseline Mode

When Jikan coverage is below the configured enrichment threshold, the model uses:

```text
Kaggle Genres
```

for every anime.

This keeps feature quality consistent across the entire dataset.

### Enriched Mode

Once Jikan coverage reaches the configured threshold, currently:

```python
MIN_ENRICHMENT_COVERAGE = 0.90
```

the model automatically uses:

```text
Genres
Themes
Studios
Demographics
Synopsis
```

This prevents a small number of Jikan-enriched anime from receiving disproportionately richer feature vectors than the rest of the dataset.

---

# Recommendation Engine

The current recommendation engine uses:

* TF-IDF vectorization
* Cosine distance
* Scikit-learn `NearestNeighbors`
* Sparse feature matrices
* Joblib model persistence

Instead of precomputing and storing a full anime-to-anime similarity matrix, recommendations are calculated when requested using nearest-neighbor search.

This avoids the quadratic memory cost of storing similarity scores for every anime pair.

---

## Recommendation Flow

```text
Anime Title
     │
     ▼
Normalized Title Lookup
     │
     ▼
Anime TF-IDF Vector
     │
     ▼
NearestNeighbors
     │
     ▼
Cosine Distance
     │
     ▼
Top-N Similar Anime
```

Title lookup is case-insensitive and whitespace-normalized.

For example:

```text
Steins;Gate
steins;gate
STEINS;GATE
```

resolve to the same normalized title.

Duplicate anime titles are safely represented internally rather than assuming every title is unique.

---

# Data Pipeline

```text
Kaggle Dataset
      │
      ▼
Dataset Validation
      │
      ▼
Anime Metadata Cleaning
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
Jikan Enrichment        Baseline Features
      │                      │
      ▼                      │
Coverage Check               │
      │                      │
      ├── Incomplete ────────┘
      │
      └── Sufficient
              │
              ▼
       Enriched Features
              │
              ▼
       TF-IDF Vectorization
              │
              ▼
       NearestNeighbors
              │
              ▼
       Recommendations
```

---

# Project Structure

```text
anime-recommender/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   │
│   └── processed/
│       └── .gitkeep
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
├── .gitignore
├── README.md
└── requirements.txt
```

Raw datasets, processed datasets, and generated model artifacts are intentionally excluded from Git.

---

# Dataset

## Kaggle Anime Recommendations Database

The project uses the Kaggle Anime Recommendations Database.

The dataset contains approximately:

* 12,000+ anime
* 7.8 million user interactions

### Anime metadata

Typical fields include:

* Anime ID
* Name
* Genre
* Type
* Episodes
* Rating
* Member count

### User ratings

Typical fields include:

* User ID
* Anime ID
* Rating

The original Kaggle dataset uses:

```text
rating = -1
```

to represent an interaction where the user watched or interacted with an anime but did not provide an explicit rating.

These interactions are preserved for future collaborative or implicit-feedback models.

---

# Setup

## 1. Clone the repository

```bash
git clone https://github.com/aadish-15/anime-recommender.git
cd anime-recommender
```

---

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add the Kaggle dataset

Place the source files inside:

```text
data/raw/
```

Expected files:

```text
data/raw/anime.csv
data/raw/rating.csv
```

The raw datasets are intentionally excluded from Git.

---

# Running the Pipeline

## Step 1 — Validate Kaggle data

```bash
python etl/load_kaggle_ratings.py
```

This verifies both source datasets and prints information such as:

* Anime count
* Rating count
* Unique users
* Unique anime IDs
* Missing values
* Duplicate IDs
* Explicit ratings
* Unrated interactions

---

## Step 2 — Clean anime metadata

```bash
python etl/transform.py
```

Output:

```text
data/processed/anime_clean.csv
```

---

## Step 3 — Enrich metadata with Jikan

```bash
python etl/fetch_jikan.py
```

Outputs:

```text
data/processed/jikan_enriched.csv
data/processed/jikan_failures.csv
```

This stage is optional for development.

If Jikan is unavailable or enrichment is incomplete, the rest of the pipeline can still operate using baseline features.

The enrichment script can safely be stopped and resumed later.

---

## Step 4 — Build recommendation features

```bash
python etl/build_features.py
```

Output:

```text
data/processed/anime_features.csv
```

The script automatically determines whether to use:

```text
BASELINE
```

or:

```text
ENRICHED
```

feature mode depending on Jikan coverage.

---

## Step 5 — Train the recommendation model

```bash
python models/train.py
```

Generated artifacts include:

```text
models/tfidf_vectorizer.pkl
models/tfidf_matrix.pkl
models/nearest_neighbors.pkl
models/anime_metadata.pkl
models/anime_indices.pkl
```

These generated files are excluded from Git.

---

## Step 6 — Test recommendations

```bash
python models/content_based.py
```

Example:

```text
Recommendations for Steins;Gate:

1. ...
2. ...
3. ...
```

---

# Current Architecture

The recommendation system currently follows a content-based architecture:

```text
Anime Metadata
      │
      ▼
Feature Text
      │
      ▼
TF-IDF
      │
      ▼
Sparse Feature Matrix
      │
      ▼
NearestNeighbors
      │
      ▼
Cosine Distance
      │
      ▼
Top-N Recommendations
```

The system does not store a full dense similarity matrix.

Similarity is calculated only when recommendations are requested.

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Requests
* Joblib
* tqdm

Planned backend technologies:

* FastAPI
* Uvicorn

---

# Current Status

## Completed

* Kaggle dataset validation
* Anime metadata cleaning
* Genre normalization
* Jikan API enrichment pipeline
* Retry and checkpoint handling
* Failure tracking
* Resume support
* Partial enrichment support
* Baseline feature generation
* Enriched feature generation
* TF-IDF vectorization
* Sparse feature storage
* Nearest-neighbor recommendation model
* Model artifact persistence
* Normalized title lookup
* Duplicate-title-safe indexing

---

## In Progress

* Full Jikan metadata enrichment

---

## Planned

* Recommendation evaluation metrics
* Collaborative filtering
* Hybrid recommendation engine
* Fuzzy / alternative title search
* FastAPI REST API
* API response schemas
* Automated tests
* CI pipeline
* Docker deployment
* SQL analytics
* Recommendation quality evaluation

---

# Future Recommendation Architecture

The long-term goal is a hybrid recommendation system combining:

```text
Content-Based Recommendations
            +
Collaborative Filtering
            +
User Preference Signals
            │
            ▼
     Hybrid Ranking
            │
            ▼
Personalized Recommendations
```

The Kaggle ratings dataset is already part of the ingestion pipeline so it can later be incorporated into collaborative filtering without restructuring the raw data layer.

---

# Engineering Goals

This project is designed as both a recommendation system and an exercise in building a maintainable machine-learning backend.

The architecture emphasizes:

* Separation of ETL and model logic
* Reproducible data processing
* Resumable external API ingestion
* Failure recovery
* Sparse ML representations
* Decoupled training and inference
* Consistent model artifacts
* Extensibility toward API deployment
* Extensibility toward hybrid recommendations

---

Built as a portfolio project demonstrating machine learning engineering, recommendation systems, backend development, and data pipeline design.
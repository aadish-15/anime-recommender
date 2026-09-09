# Anime Recommender System

A backend-focused anime recommendation system built with Python, Pandas, NumPy, Scikit-learn, and data engineering practices.

The project currently implements a **content-based recommendation engine** using anime metadata from the Kaggle Anime Recommendations Database, with optional enrichment from the Jikan API.

The recommendation pipeline is designed to remain usable even when external enrichment is incomplete or temporarily unavailable.

The current system includes:

- Dataset validation
- Anime metadata cleaning
- Optional Jikan enrichment
- Baseline and enriched feature generation
- TF-IDF vectorization
- Sparse nearest-neighbor retrieval
- Rating-aware reranking
- Popularity-aware reranking
- Fuzzy anime title search
- Recommendation evaluation
- Persistent evaluation benchmark history

The long-term goal is to evolve the project into a **hybrid recommendation system** combining content similarity, collaborative filtering, and user preference signals.

---

## Features

### Data Ingestion

The project ingests both anime metadata and user interaction data from the Kaggle Anime Recommendations Database.

Current ingestion functionality includes:

- Kaggle anime metadata ingestion
- Kaggle user ratings ingestion
- Dataset schema validation
- Missing-value reporting
- Duplicate anime ID detection
- Unique user reporting
- Unique anime reporting
- Explicit rating detection
- Unrated interaction detection
- Referential integrity checks between anime and rating datasets

The Kaggle ratings dataset uses:

```text
rating = -1
```

to represent interactions where a user watched or interacted with an anime but did not provide an explicit numerical rating.

These interactions are preserved for future collaborative and implicit-feedback models.

> User ratings are currently used for **offline recommendation evaluation**, but they are **not yet used to train the content-based recommendation model**.

---

### Data Cleaning

Anime metadata cleaning includes:

- Duplicate anime removal
- Invalid anime ID handling
- Missing title filtering
- Genre normalization
- Numeric episode conversion
- Nullable episode counts
- Rating validation
- Missing rating preservation
- Member count normalization
- Anime type normalization
- Whitespace cleanup

Unknown values are preserved as missing values rather than replaced with misleading defaults.

For example, an unknown episode count is preserved as missing instead of being converted to `0`.

The cleaned anime dataset is written to:

```text
data/processed/anime_clean.csv
```

---

## Jikan Metadata Enrichment

The project includes an optional metadata enrichment pipeline using the Jikan API.

The enrichment stage can retrieve:

- Synopsis
- Themes
- Studios
- Demographics
- Genres
- Season
- Year
- Score

The Jikan pipeline includes:

- Resume support
- Checkpointing
- Retry handling
- Exponential backoff
- Random retry jitter
- `Retry-After` support
- Network timeout handling
- HTTP error classification
- HTTP session recreation after gateway errors
- Failure tracking
- Atomic checkpoint writes
- Safe interruption handling with `Ctrl+C`
- Diagnostic output for failed HTTP responses

Successful records are written to:

```text
data/processed/jikan_enriched.csv
```

Failed requests are tracked separately in:

```text
data/processed/jikan_failures.csv
```

Permanent failures such as invalid or missing anime IDs are skipped on future runs, while retryable failures can be attempted again later.

The enrichment process is intentionally designed so that interruptions or temporary API instability do not destroy previous progress.

---

## Jikan Is Optional

The recommendation pipeline does **not require complete Jikan enrichment to run**.

This is an important architectural decision.

External APIs can be:

- Temporarily unavailable
- Rate-limited
- Slow
- Partially degraded
- Inconsistent across thousands of requests

The recommendation engine therefore does not depend on Jikan as a required runtime or training dependency.

If Jikan enrichment is incomplete, the system automatically falls back to Kaggle-only features.

---

## Partial Jikan Enrichment Support

Two feature modes are supported.

### Baseline Mode

When Jikan coverage is below the configured enrichment threshold, the model uses:

```text
Kaggle Genres
```

for every anime.

The current enrichment threshold is:

```python
MIN_ENRICHMENT_COVERAGE = 0.90
```

Using the same feature source across the dataset prevents partially enriched anime from receiving disproportionately richer feature vectors.

For example, it would be undesirable for:

```text
Anime A
Genres + Themes + Studios + Synopsis
```

to compete directly with:

```text
Anime B
Genres only
```

when only a small portion of the dataset has been enriched.

Baseline mode avoids that inconsistency.

### Enriched Mode

Once Jikan coverage reaches the configured threshold, the model automatically uses:

```text
Genres
Themes
Studios
Demographics
Synopsis
```

The feature mode is selected automatically during feature generation.

---

# Recommendation Engine

The current recommendation engine uses:

- TF-IDF vectorization
- Cosine distance
- Scikit-learn `NearestNeighbors`
- Sparse feature matrices
- Joblib model persistence
- Candidate retrieval
- Rating-based reranking
- Popularity-based reranking

Instead of precomputing and storing a full anime-to-anime similarity matrix, recommendations are calculated only when requested.

This avoids the quadratic memory cost of storing similarity scores between every possible anime pair.

For approximately 12,000 anime, a full pairwise similarity matrix would contain roughly:

```text
12,000 × 12,000
=
144,000,000 similarity values
```

The current implementation instead keeps the TF-IDF matrix sparse and calculates nearest neighbors on demand.

---

## Recommendation Flow

```text
Anime Title
     │
     ▼
Exact / Fuzzy Title Resolution
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
Larger Candidate Pool
     │
     ├───────────────┐
     │               │
     ▼               ▼
Anime Rating      Popularity
     │               │
     └───────┬───────┘
             │
             ▼
         Reranking
             │
             ▼
    Top-N Recommendations
```

---

## TF-IDF Feature Representation

Anime metadata is converted into text-based features.

In baseline mode, an anime may be represented by text such as:

```text
action adventure fantasy shounen
```

The TF-IDF vectorizer converts this text into a sparse numerical representation.

Anime with similar metadata therefore receive similar vectors.

The recommendation engine then uses cosine distance to identify nearby anime in the TF-IDF feature space.

---

## Sparse Nearest-Neighbor Retrieval

The project uses:

```python
NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)
```

The model is trained on the sparse TF-IDF matrix.

The project intentionally does **not** save a full dense cosine similarity matrix.

Generated model artifacts include:

```text
models/tfidf_vectorizer.pkl
models/tfidf_matrix.pkl
models/nearest_neighbors.pkl
models/anime_metadata.pkl
models/anime_indices.pkl
```

These artifacts are excluded from Git.

---

## Recommendation Reranking

The original content-based implementation returned nearest neighbors directly in cosine-similarity order.

The current implementation instead retrieves a larger candidate set and reranks those candidates using multiple signals.

Current ranking weights are:

```python
SIMILARITY_WEIGHT = 0.80
RATING_WEIGHT = 0.15
POPULARITY_WEIGHT = 0.05
```

This means the final recommendation score is approximately:

```text
80% Content Similarity
15% Anime Rating
 5% Popularity
```

Content similarity remains the dominant signal.

This prevents extremely popular or highly rated anime from appearing merely because they are popular.

---

## Candidate Pool Reranking

Instead of requesting exactly the final number of recommendations, the model retrieves a larger candidate pool first.

For example:

```text
Requested recommendations
        │
        ▼
       10
        │
        ▼
Retrieve ~50 content-similar candidates
        │
        ▼
Apply rating and popularity signals
        │
        ▼
Sort by final score
        │
        ▼
Return Top 10
```

This allows the secondary ranking signals to improve ordering without replacing the content-similarity stage.

---

## Popularity Normalization

Raw member counts are highly skewed.

For example:

```text
Anime A: 5,000 members
Anime B: 1,500,000 members
```

Using raw member counts directly would allow popularity to dominate the final ranking.

The recommender therefore applies logarithmic normalization:

```text
log(1 + members)
```

before converting popularity into a normalized score.

This allows popularity to influence ranking without overwhelming content similarity.

---

# Fuzzy Anime Title Search

The recommender supports fuzzy anime title matching.

Users do not need to enter the exact database title.

For example:

```text
steins gate
naroto
attack titan
full metal alchemist
deathnote
```

can return likely matching anime.

The search system combines:

- Case-insensitive normalization
- Whitespace normalization
- Character-level similarity
- Token overlap
- Query token coverage
- Substring matching
- Alternative-title support when those fields exist

---

## Exact Title Resolution

Exact normalized matches always take priority.

For example:

```text
Steins;Gate
steins;gate
STEINS;GATE
```

all resolve to the same normalized title.

Normalization is performed using:

```text
strip
casefold
whitespace normalization
```

---

## Conservative Fuzzy Resolution

The recommender intentionally avoids blindly guessing.

A fuzzy result is automatically accepted only when:

```text
Match score >= configured confidence threshold
```

and the best result is sufficiently stronger than the second-best result.

If two matches are too similar in confidence, the recommender returns no automatic resolution rather than silently choosing the wrong anime.

This behavior will become especially useful when the project later exposes search through an API or frontend.

---

## Title Search Interface

Fuzzy search can be used directly:

```python
recommender.search_titles(
    "steins gate"
)
```

Example result structure:

```python
{
    "anime_id": 9253,
    "name": "Steins;Gate",
    "score": 0.91,
    "matched_title": "steins;gate",
    "exact_match": False
}
```

Title resolution can also be used directly:

```python
recommender.resolve_title(
    "steins gate"
)
```

---

## Duplicate Anime Title Handling

Anime titles are not assumed to be unique.

Instead of using:

```text
title -> row_index
```

the model stores:

```text
normalized_title -> [row_index, row_index, ...]
```

This allows multiple anime with identical titles to remain safely represented.

The recommender excludes matching duplicate-title records when generating recommendations for the selected anime.

The evaluation system also avoids ambiguous duplicate-title entries when selecting recommendation seeds.

---

## Recommendation Output

A recommendation currently includes values such as:

```python
{
    "anime_id": ...,
    "name": ...,
    "genre": ...,
    "rating": ...,
    "members": ...,
    "similarity_score": ...,
    "rating_score": ...,
    "popularity_score": ...,
    "final_score": ...
}
```

Keeping the individual component scores visible makes recommendation behavior easier to inspect and debug.

---

# Recommendation Evaluation

The project includes an offline evaluation pipeline:

```text
models/evaluate_content.py
```

The purpose of this stage is to measure recommendation quality instead of relying only on manual inspection.

The Kaggle ratings dataset is used as evaluation ground truth.

Importantly:

```text
rating.csv
```

is **not used to train the content model**.

It is only used to check whether content-based recommendations align with anime users actually rated highly.

---

## Evaluation Strategy

For each sampled user:

```text
User's highly rated anime
        │
        ▼
Select one anime as seed
        │
        ▼
Generate Top-K recommendations
        │
        ▼
Treat remaining highly rated anime
as relevant items
        │
        ▼
Compare recommendation list
against relevant anime
        │
        ▼
Calculate ranking metrics
```

By default, an anime is considered liked when:

```text
rating >= 8
```

Users must have at least:

```text
5 liked anime
```

to be eligible for evaluation.

---

## Evaluation Configuration

Default evaluation settings are:

```text
Users sampled:       500
Top-K:                10
Like threshold:        8
Minimum liked anime:   5
Random seed:          42
```

The fixed random seed makes repeated evaluations reproducible when the dataset and model remain unchanged.

---

## Evaluation Metrics

### Precision@K

Measures the fraction of returned recommendations that are relevant.

```text
Relevant recommendations
------------------------
Total recommendations
```

### Recall@K

Measures the fraction of the user's relevant anime recovered by the recommendation list.

```text
Relevant recommendations found
------------------------------
Total relevant anime
```

Recall can appear low when users have hundreds of highly rated anime but the recommender only returns 10 results.

### Hit Rate@K

Measures whether at least one relevant anime appears in the Top-K recommendations.

For each user:

```text
1 = at least one hit
0 = no hits
```

The final Hit Rate is the average across evaluated users.

### Mean Reciprocal Rank — MRR@K

Measures how early the first relevant recommendation appears.

A hit at rank:

```text
1 -> reciprocal rank 1.0
2 -> reciprocal rank 0.5
3 -> reciprocal rank 0.333...
```

Earlier relevant recommendations therefore receive higher scores.

### NDCG@K

Normalized Discounted Cumulative Gain rewards relevant recommendations more strongly when they appear near the top of the result list.

This gives a more ranking-sensitive measure than basic Hit Rate.

### Catalog Coverage

Catalog Coverage measures how much of the full anime catalog appears across recommendations.

A system with extremely low catalog coverage may repeatedly recommend only a small group of popular anime.

Higher coverage indicates greater recommendation diversity across the catalog.

---

## Current Content Model Baseline

Using:

```text
500 sampled users
Top-K = 10
Like threshold = 8
```

the current model produced approximately:

| Metric | Result |
|---|---:|
| Precision@10 | 0.0732 |
| Recall@10 | 0.0244 |
| Hit Rate@10 | 0.4140 |
| MRR@10 | 0.2789 |
| NDCG@10 | 0.1002 |
| Average Hits per User | 0.7320 |
| Catalog Coverage | 17.63% |

The Hit Rate result means that approximately:

```text
41.4%
```

of evaluated users received at least one anime they had also rated highly within the Top 10 recommendations.

These results are now treated as the current **content-model baseline**.

Future models can be evaluated against the same benchmark structure.

---

## Evaluation Output

Detailed per-user evaluation results are saved to:

```text
data/processed/content_evaluation.csv
```

Each row includes values such as:

```text
user_id
seed_anime_id
seed_title
relevant_count
recommendation_count
hit_count
precision_at_k
recall_at_k
hit_rate_at_k
mrr_at_k
ndcg_at_k
```

---

## Evaluation Benchmark History

Aggregate benchmark results are also saved to:

```text
data/processed/content_evaluation_summary.csv
```

Each evaluation run records:

- Timestamp
- Model name
- Requested users
- Evaluated users
- Skipped users
- Top-K
- Like threshold
- Minimum liked anime
- Similarity weight
- Rating weight
- Popularity weight
- Precision@K
- Recall@K
- Hit Rate@K
- MRR@K
- NDCG@K
- Average hits
- Catalog coverage
- Evaluation runtime

This creates a simple experiment history for comparing future recommendation models and ranking configurations.

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
      ├──────────────────────────────┐
      │                              │
      ▼                              ▼
Optional Jikan Enrichment      Kaggle Features
      │                              │
      ▼                              │
Coverage Check                       │
      │                              │
      ├── Incomplete ────────────────┘
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
       Sparse Feature Matrix
              │
              ▼
       NearestNeighbors
              │
              ▼
       Candidate Retrieval
              │
              ▼
          Reranking
              │
              ▼
       Recommendations
              │
              ▼
       Offline Evaluation
```

---

# Current Architecture

The current recommendation system follows a content-based architecture:

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
Candidate Pool
      │
      ▼
Similarity + Rating + Popularity
      │
      ▼
Final Ranking
```

The system does not store a full dense similarity matrix.

Similarity is calculated only when recommendations are requested.

---

# Project Structure

```text
anime-recommender/
│
├── api/
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
│   ├── content_based.py
│   └── evaluate_content.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Raw datasets, processed datasets, evaluation outputs, and generated model artifacts are intentionally excluded from Git.

---

# Dataset

## Kaggle Anime Recommendations Database

The project uses the Kaggle Anime Recommendations Database.

The dataset contains approximately:

- 12,000+ anime
- 7.8 million user interactions

### Anime Metadata

Typical fields include:

- Anime ID
- Name
- Genre
- Type
- Episodes
- Rating
- Member count

### User Ratings

Typical fields include:

- User ID
- Anime ID
- Rating

The original Kaggle dataset uses:

```text
rating = -1
```

to represent an interaction where a user watched or interacted with an anime but did not provide an explicit rating.

These interactions are preserved for future collaborative or implicit-feedback models.

---

# Setup

## 1. Clone the Repository

```bash
git clone https://github.com/aadish-15/anime-recommender.git
cd anime-recommender
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add the Kaggle Dataset

Place the source files inside:

```text
data/raw/
```

Expected files:

```text
data/raw/anime.csv
data/raw/rating.csv
```

Raw datasets are intentionally excluded from Git.

---

# Running the Pipeline

## Step 1 — Validate Kaggle Data

```bash
python etl/load_kaggle_ratings.py
```

This verifies both source datasets and reports information such as:

- Anime count
- Rating count
- Unique users
- Unique anime IDs
- Missing values
- Duplicate IDs
- Explicit ratings
- Unrated interactions
- Rating anime IDs missing from metadata

---

## Step 2 — Clean Anime Metadata

```bash
python etl/transform.py
```

Output:

```text
data/processed/anime_clean.csv
```

---

## Step 3 — Optional Jikan Enrichment

```bash
python etl/fetch_jikan.py
```

Outputs:

```text
data/processed/jikan_enriched.csv
data/processed/jikan_failures.csv
```

This stage is optional.

If Jikan is unavailable or enrichment is incomplete, the rest of the pipeline continues using baseline Kaggle features.

The enrichment script can safely be interrupted and resumed later.

---

## Step 4 — Build Recommendation Features

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

## Step 5 — Train the Recommendation Model

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

## Step 6 — Test Recommendations

```bash
python models/content_based.py
```

The demonstration currently tests fuzzy title search using:

```text
steins gate
```

and resolves it to:

```text
Steins;Gate
```

before generating recommendations.

---

## Step 7 — Evaluate Recommendation Quality

```bash
python models/evaluate_content.py
```

Default configuration:

```text
500 users
Top-10 recommendations
Rating >= 8 considered liked
```

Custom evaluation options can also be supplied.

For example:

```bash
python models/evaluate_content.py --users 1000 --k 10
```

or:

```bash
python models/evaluate_content.py --users 500 --k 20
```

or:

```bash
python models/evaluate_content.py --users 500 --k 10 --like-threshold 9
```

---

# Technologies

Current technologies include:

- Python
- Pandas
- NumPy
- Scikit-learn
- Requests
- Joblib
- tqdm

Backend dependencies already included for future development:

- FastAPI
- Uvicorn

---

# Current Status

## Completed

### Data Pipeline

- Kaggle anime metadata ingestion
- Kaggle ratings ingestion
- Dataset schema validation
- Referential integrity checks
- Missing-value reporting
- Duplicate ID detection
- Anime metadata cleaning
- Genre normalization
- Rating validation
- Member-count normalization

### Jikan Enrichment

- Jikan API enrichment pipeline
- Resume support
- Checkpointing
- Retry handling
- Exponential backoff
- Retry jitter
- `Retry-After` support
- Session reset handling
- Network timeout handling
- Failure classification
- Failure tracking
- Atomic checkpoint writes
- Safe interruption handling
- Partial enrichment support

### Feature Engineering

- Baseline feature generation
- Enriched feature generation
- Automatic feature-mode selection
- TF-IDF vectorization
- Sparse feature storage

### Content Recommendation Model

- Nearest-neighbor recommendation model
- Cosine-distance retrieval
- Model artifact persistence
- Normalized title lookup
- Duplicate-title-safe indexing
- Candidate-pool retrieval
- Rating-aware reranking
- Popularity-aware reranking
- Log-scaled popularity score
- Multi-signal final ranking

### Search

- Fuzzy anime title search
- Character similarity scoring
- Token-overlap scoring
- Substring matching
- Exact-match priority
- Conservative automatic resolution
- Ambiguous-match protection
- Future alternative-title support

### Evaluation

- Ratings-based offline evaluation
- Reproducible user sampling
- Precision@K
- Recall@K
- Hit Rate@K
- MRR@K
- NDCG@K
- Average hits per user
- Catalog coverage
- Per-user evaluation output
- Aggregate benchmark history
- Baseline performance measurement

---

# Current Baseline

The current content-based model has established the following Top-10 evaluation baseline:

```text
Precision@10:      0.0732
Recall@10:         0.0244
Hit Rate@10:       0.4140
MRR@10:            0.2789
NDCG@10:           0.1002
Average Hits/User: 0.7320
Catalog Coverage:  17.63%
```

These values provide a reference point for measuring future collaborative and hybrid models.

---

# Next Development Phase

The next major phase is:

```text
Collaborative Filtering
```

The Kaggle ratings dataset already contains millions of user-anime interactions, so the project can begin learning recommendation patterns directly from user behavior.

The collaborative model will eventually complement the current content model.

---

# Future Recommendation Architecture

The long-term architecture is:

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

The Kaggle ratings dataset is already part of the ingestion and evaluation pipeline, so collaborative filtering can be introduced without restructuring the raw data layer.

---

# Planned

- Collaborative filtering
- Collaborative model training
- Collaborative model evaluation
- User-specific recommendations
- Hybrid recommendation engine
- Hybrid ranking
- Hybrid evaluation
- FastAPI REST API
- API response schemas
- Search endpoint
- Recommendation endpoints
- Automated tests
- CI pipeline
- Docker deployment
- SQL analytics

---

# Planned API Direction

The future backend may expose endpoints such as:

```text
GET /health

GET /anime/search?q=steins

GET /recommendations/content/{anime_id}

GET /recommendations/user/{user_id}

GET /recommendations/hybrid/{user_id}
```

---

# Engineering Goals

This project is designed as both a recommendation system and an exercise in building a maintainable machine-learning backend.

The architecture emphasizes:

- Separation of ETL and model logic
- Reproducible data processing
- Resumable external API ingestion
- Graceful failure recovery
- Optional external metadata enrichment
- Sparse ML representations
- Efficient nearest-neighbor retrieval
- Decoupled training and inference
- Explainable ranking signals
- Reproducible evaluation
- Measurable recommendation quality
- Persistent benchmark history
- Extensibility toward collaborative filtering
- Extensibility toward hybrid recommendations
- Extensibility toward API deployment

---

Built as a portfolio project demonstrating machine learning engineering, recommendation systems, backend development, data pipelines, fuzzy search, ranking systems, and recommendation evaluation.
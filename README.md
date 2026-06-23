# Anime Recommender System

A backend-focused anime recommendation system built with Python, machine learning, and FastAPI.

The project combines content-based filtering, collaborative filtering, and metadata enrichment to generate personalized anime recommendations.

## Features

* Data ingestion and ETL pipeline
* Anime metadata cleaning and preprocessing
* Content-based recommendation engine using TF-IDF and cosine similarity
* Model artifact generation using Joblib
* Planned Jikan API metadata enrichment
* Planned collaborative filtering using user rating data
* Planned hybrid recommendation system
* FastAPI endpoints for recommendation serving
* SQL analytics for recommendation insights

## Project Structure

```text
anime-recommender/

├── api/
│   └── main.py

├── data/
│   ├── raw/
│   └── processed/

├── etl/
│   ├── load_kaggle_ratings.py
│   ├── transform.py
│   └── fetch_jikan.py

├── models/
│   ├── content_based.py
│   ├── collaborative.py
│   └── train.py

├── notebooks/
│   └── eda.ipynb

├── sql/
│   └── analytics_queries.sql

├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project uses the Anime Recommendations Database dataset from Kaggle.

Dataset contents:

* anime.csv
* rating.csv

Statistics:

* 12,294 anime entries
* 7.8 million user ratings

## Current Pipeline

1. Load anime and ratings datasets
2. Clean and normalize metadata
3. Generate TF-IDF feature vectors
4. Compute cosine similarity matrix
5. Save model artifacts using Joblib
6. Serve recommendations through API endpoints

## Example Recommendation

Input:

```python
recommend("Death Note")
```

Output:

```text
Monster
Higurashi no Naku Koro ni
Mirai Nikki
Psycho-Pass
...
```

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Joblib

## Roadmap

### Completed

* Dataset ingestion
* ETL pipeline
* Metadata cleaning
* TF-IDF feature generation
* Content-based recommendation engine
* Model artifact persistence

### In Progress

* Jikan API metadata enrichment

### Planned

* Collaborative filtering
* Hybrid recommendation system
* FastAPI recommendation endpoints
* SQL analytics layer
* Docker deployment
* Automated retraining pipeline

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run model training:

```bash
python models/train.py
```

Run content recommender:

```bash
python models/content_based.py
```

## Author

Built as a machine learning and backend engineering portfolio project focused on recommendation systems, data processing, and model deployment by Aadish Tamaskar.
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("data/processed/anime_clean.csv")

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    df["genre_clean"]
)

similarity_matrix = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    similarity_matrix,
    "models/content_similarity.pkl"
)

joblib.dump(
    df,
    "models/anime_metadata.pkl"
)

indices = pd.Series(
    df.index,
    index=df["name"]
).drop_duplicates()

joblib.dump(
    indices,
    "models/anime_indices.pkl"
)

print("Artifacts saved successfully")
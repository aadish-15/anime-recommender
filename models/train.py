from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anime_features.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"

VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_FILE = MODELS_DIR / "tfidf_matrix.pkl"
NEIGHBOR_MODEL_FILE = MODELS_DIR / "nearest_neighbors.pkl"
METADATA_FILE = MODELS_DIR / "anime_metadata.pkl"
INDICES_FILE = MODELS_DIR / "anime_indices.pkl"

# Old artifact from the previous implementation.
LEGACY_SIMILARITY_FILE = (
    MODELS_DIR
    / "content_similarity.pkl"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_title(title):
    """
    Normalize anime titles for reliable lookup.

    Example:
        "  Steins;Gate  " -> "steins;gate"
    """

    return " ".join(
        str(title)
        .strip()
        .casefold()
        .split()
    )


def validate_dataframe(df):
    """
    Make sure required columns exist before training.
    """

    required_columns = {
        "anime_id",
        "name",
        "feature_text"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"Missing required columns: {missing}"
        )


def build_title_index(df):
    """
    Build:

        normalized_title -> [row_index, ...]

    A list is used instead of a single row index because
    multiple anime can legitimately have the same title.
    """

    title_indices = {}

    for row_index, title in enumerate(
        df["name"]
    ):

        normalized_title = normalize_title(
            title
        )

        if not normalized_title:
            continue

        title_indices.setdefault(
            normalized_title,
            []
        ).append(row_index)

    return title_indices


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train():
    """
    Train the content-based recommendation artifacts.

    Unlike the previous version, this does NOT create a
    full NxN cosine similarity matrix.

    Instead:

    1. TF-IDF vectors are created.
    2. The sparse TF-IDF matrix is stored.
    3. A NearestNeighbors model is trained.
    4. Similarity is calculated only when recommendations
       are requested.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Feature dataset not found:\n"
            f"{DATA_FILE}\n\n"
            "Run the ETL pipeline first."
        )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Loading dataset:\n{DATA_FILE}"
    )

    df = pd.read_csv(
        DATA_FILE
    )

    validate_dataframe(
        df
    )

    print(
        f"Loaded {len(df):,} anime"
    )

    # ---------------------------------------------
    # Clean feature text
    # ---------------------------------------------

    df["feature_text"] = (
        df["feature_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_features = (
        df["feature_text"] == ""
    ).sum()

    if empty_features:
        print(
            f"Warning: {empty_features:,} anime "
            "have empty feature_text"
        )

    # ---------------------------------------------
    # TF-IDF
    # ---------------------------------------------

    print(
        "\nBuilding TF-IDF matrix..."
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        dtype=np.float32
    )

    tfidf_matrix = (
        vectorizer.fit_transform(
            df["feature_text"]
        )
    )

    print(
        "TF-IDF shape:",
        tfidf_matrix.shape
    )

    print(
        "Vocabulary size:",
        len(vectorizer.vocabulary_)
    )

    # ---------------------------------------------
    # Nearest-neighbor model
    # ---------------------------------------------

    print(
        "\nTraining nearest-neighbor model..."
    )

    neighbor_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    neighbor_model.fit(
        tfidf_matrix
    )

    # ---------------------------------------------
    # Build safe title lookup
    # ---------------------------------------------

    print(
        "\nBuilding title index..."
    )

    title_indices = build_title_index(
        df
    )

    duplicate_titles = sum(
        1
        for indices in title_indices.values()
        if len(indices) > 1
    )

    print(
        f"Unique normalized titles: "
        f"{len(title_indices):,}"
    )

    print(
        f"Duplicate title groups: "
        f"{duplicate_titles:,}"
    )

    # ---------------------------------------------
    # Save artifacts
    # ---------------------------------------------

    print(
        "\nSaving model artifacts..."
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    joblib.dump(
        tfidf_matrix,
        TFIDF_MATRIX_FILE
    )

    joblib.dump(
        neighbor_model,
        NEIGHBOR_MODEL_FILE
    )

    joblib.dump(
        df,
        METADATA_FILE
    )

    joblib.dump(
        title_indices,
        INDICES_FILE
    )

    # Remove artifact belonging to the old
    # NxN similarity implementation.
    if LEGACY_SIMILARITY_FILE.exists():

        LEGACY_SIMILARITY_FILE.unlink()

        print(
            "Removed old artifact:"
            "\ncontent_similarity.pkl"
        )

    print(
        "\nTraining complete."
    )

    print(
        "\nSaved:"
    )

    print(
        f"- {VECTORIZER_FILE.name}"
    )

    print(
        f"- {TFIDF_MATRIX_FILE.name}"
    )

    print(
        f"- {NEIGHBOR_MODEL_FILE.name}"
    )

    print(
        f"- {METADATA_FILE.name}"
    )

    print(
        f"- {INDICES_FILE.name}"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    train()
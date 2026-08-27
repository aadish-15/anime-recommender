from pathlib import Path
import time

import joblib
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

METADATA_FILE = MODELS_DIR / "anime_metadata.pkl"
TFIDF_MATRIX_FILE = MODELS_DIR / "tfidf_matrix.pkl"
NEIGHBOR_MODEL_FILE = MODELS_DIR / "nearest_neighbors.pkl"
INDICES_FILE = MODELS_DIR / "anime_indices.pkl"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_title(title):

    return " ".join(
        str(title)
        .strip()
        .casefold()
        .split()
    )


# ---------------------------------------------------------
# Recommender
# ---------------------------------------------------------

class ContentBasedRecommender:

    def __init__(self):

        self._validate_artifacts()

        self.df = joblib.load(
            METADATA_FILE
        )

        self.tfidf_matrix = joblib.load(
            TFIDF_MATRIX_FILE
        )

        self.neighbor_model = joblib.load(
            NEIGHBOR_MODEL_FILE
        )

        self.indices = joblib.load(
            INDICES_FILE
        )

        self._validate_loaded_data()

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    def _validate_artifacts(self):
        """
        Ensure all required trained artifacts exist.
        """

        required_files = [
            METADATA_FILE,
            TFIDF_MATRIX_FILE,
            NEIGHBOR_MODEL_FILE,
            INDICES_FILE
        ]

        missing_files = [
            path
            for path in required_files
            if not path.exists()
        ]

        if missing_files:

            missing = "\n".join(
                f"- {path.name}"
                for path in missing_files
            )

            raise FileNotFoundError(
                "Missing model artifacts:\n"
                f"{missing}\n\n"
                "Run:\n"
                "python models/train.py"
            )

    def _validate_loaded_data(self):
        """
        Ensure saved model artifacts match the metadata.
        """

        metadata_rows = len(
            self.df
        )

        matrix_rows = (
            self.tfidf_matrix.shape[0]
        )

        if metadata_rows != matrix_rows:

            raise ValueError(
                "Model artifact mismatch.\n"
                f"Metadata rows: {metadata_rows}\n"
                f"TF-IDF rows: {matrix_rows}\n\n"
                "Retrain the model using:\n"
                "python models/train.py"
            )

        required_columns = {
            "anime_id",
            "name",
            "genre",
            "rating"
        }

        missing_columns = (
            required_columns
            - set(self.df.columns)
        )

        if missing_columns:

            missing = ", ".join(
                sorted(missing_columns)
            )

            raise ValueError(
                "Anime metadata is missing "
                f"required columns: {missing}"
            )

    # -----------------------------------------------------
    # Title lookup
    # -----------------------------------------------------

    def _get_title_indices(
        self,
        anime_name
    ):

        normalized_title = normalize_title(
            anime_name
        )

        if not normalized_title:
            return []

        return self.indices.get(
            normalized_title,
            []
        )

    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    def recommend(
        self,
        anime_name,
        top_n=10
    ):
        """
        Return content-based recommendations for an anime.

        Parameters
        ----------
        anime_name : str
            Anime title to search for.

        top_n : int
            Number of recommendations to return.

        Returns
        -------
        list[dict]
            Recommendation records.

            Returns an empty list if the title
            cannot be found.
        """

        if not isinstance(top_n, int):
            raise TypeError(
                "top_n must be an integer."
            )

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than 0."
            )

        matching_indices = (
            self._get_title_indices(
                anime_name
            )
        )

        if not matching_indices:
            return []


        query_index = matching_indices[0]

        total_anime = len(
            self.df
        )

        if total_anime <= 1:
            return []

        # Request a few additional neighbors because
        # the source anime itself and duplicate-title
        # records need to be removed.
        requested_neighbors = min(
            total_anime,
            top_n
            + len(matching_indices)
            + 5
        )

        distances, neighbor_indices = (
            self.neighbor_model.kneighbors(
                self.tfidf_matrix[
                    query_index
                ],
                n_neighbors=requested_neighbors
            )
        )

        source_indices = set(
            matching_indices
        )

        recommendations = []

        for distance, index in zip(
            distances[0],
            neighbor_indices[0]
        ):

            index = int(index)

            # Never recommend the anime itself.
            # Also exclude duplicate rows having
            # the exact same title.
            if index in source_indices:
                continue

            row = self.df.iloc[
                index
            ]

            recommendation = {
                "anime_id": (
                    int(row["anime_id"])
                    if pd.notna(
                        row["anime_id"]
                    )
                    else None
                ),
                "name": row["name"],
                "genre": (
                    row["genre"]
                    if pd.notna(
                        row["genre"]
                    )
                    else None
                ),
                "rating": (
                    float(row["rating"])
                    if pd.notna(
                        row["rating"]
                    )
                    else None
                )
            }

            recommendations.append(
                recommendation
            )

            if len(recommendations) >= top_n:
                break

        return recommendations


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

def main():

    start_time = time.perf_counter()

    recommender = (
        ContentBasedRecommender()
    )

    load_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Recommender loaded in "
        f"{load_time:.2f}s"
    )

    anime_name = "Steins;Gate"

    print(
        f"\nRecommendations for "
        f"{anime_name}:\n"
    )

    recommendations = (
        recommender.recommend(
            anime_name,
            top_n=10
        )
    )

    if not recommendations:

        print(
            "Anime not found or "
            "no recommendations available."
        )

        return

    for position, anime in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{position}. "
            f"{anime['name']} "
            f"(Rating: {anime['rating']})"
        )


if __name__ == "__main__":
    main()
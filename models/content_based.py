from pathlib import Path
from difflib import SequenceMatcher
import math
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
# Ranking configuration
# ---------------------------------------------------------

CANDIDATE_MULTIPLIER = 5
MIN_CANDIDATE_POOL = 50

SIMILARITY_WEIGHT = 0.80
RATING_WEIGHT = 0.15
POPULARITY_WEIGHT = 0.05


# ---------------------------------------------------------
# Search configuration
# ---------------------------------------------------------

# Minimum score for a result to appear in search suggestions.
SEARCH_MIN_SCORE = 0.55

# Minimum score required to automatically treat a fuzzy
# result as the user's intended anime.
AUTO_RESOLVE_MIN_SCORE = 0.75

# If the top two fuzzy matches are extremely close,
# don't automatically choose between them.
AUTO_RESOLVE_MARGIN = 0.025

DEFAULT_SEARCH_LIMIT = 10


# Optional alternative-title columns.
#
# These don't exist in the current Kaggle dataset, but
# supporting them here means future metadata enrichment
# can automatically improve title search.
ALTERNATIVE_TITLE_COLUMNS = (
    "english_name",
    "name_english",
    "title_english",
    "japanese_name",
    "name_japanese",
    "title_japanese",
    "synonyms"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_title(title):
    """
    Normalize anime titles.

    Example:
        "  Steins;Gate  " -> "steins;gate"
    """

    return " ".join(
        str(title)
        .strip()
        .casefold()
        .split()
    )


def clamp(
    value,
    minimum=0.0,
    maximum=1.0
):
    """
    Restrict a value to a fixed numeric range.
    """

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def tokenize_title(title):
    """
    Convert a normalized title into a token set.
    """

    return {
        token
        for token in normalize_title(title).split()
        if token
    }


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

        self._prepare_ranking_metadata()

        self._build_search_index()

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
        Ensure model artifacts match the metadata.
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
    # Ranking metadata
    # -----------------------------------------------------

    def _prepare_ranking_metadata(self):
        """
        Prepare rating and popularity data.
        """

        self.df["rating"] = pd.to_numeric(
            self.df["rating"],
            errors="coerce"
        )

        if "members" in self.df.columns:

            self.df["members"] = pd.to_numeric(
                self.df["members"],
                errors="coerce"
            )

            valid_members = (
                self.df["members"]
                .fillna(0)
                .clip(lower=0)
            )

            if valid_members.max() > 0:

                self.max_log_members = math.log1p(
                    float(
                        valid_members.max()
                    )
                )

            else:

                self.max_log_members = 0.0

        else:

            self.max_log_members = 0.0

    # -----------------------------------------------------
    # Search index
    # -----------------------------------------------------

    def _build_search_index(self):
        """
        Build a search catalog containing canonical titles
        and any alternative-title fields available in the
        metadata.

        Each searchable title maps back to one or more
        dataframe rows.
        """

        self.search_index = {}

        # ---------------------------------------------
        # Canonical titles
        # ---------------------------------------------

        for normalized_title, indices in self.indices.items():

            self.search_index.setdefault(
                normalized_title,
                []
            )

            for index in indices:

                if index not in self.search_index[
                    normalized_title
                ]:

                    self.search_index[
                        normalized_title
                    ].append(
                        index
                    )

        # ---------------------------------------------
        # Optional alternative titles
        # ---------------------------------------------

        available_alt_columns = [
            column
            for column in ALTERNATIVE_TITLE_COLUMNS
            if column in self.df.columns
        ]

        for row_index, row in self.df.iterrows():

            for column in available_alt_columns:

                value = row.get(
                    column
                )

                if pd.isna(value):
                    continue

                # Some future datasets may store synonyms
                # separated by commas or semicolons.
                raw_titles = str(value).replace(
                    ";",
                    ","
                ).split(",")

                for raw_title in raw_titles:

                    normalized = normalize_title(
                        raw_title
                    )

                    if not normalized:
                        continue

                    self.search_index.setdefault(
                        normalized,
                        []
                    )

                    if row_index not in self.search_index[
                        normalized
                    ]:

                        self.search_index[
                            normalized
                        ].append(
                            row_index
                        )

        self.search_terms = list(
            self.search_index.keys()
        )

    # -----------------------------------------------------
    # Search scoring
    # -----------------------------------------------------

    def _search_score(
        self,
        query,
        candidate
    ):
        """
        Calculate fuzzy similarity between two titles.

        Signals used:

        - character similarity
        - token overlap
        - query-token coverage
        - substring relationship

        The score is normalized between 0 and 1.
        """

        if query == candidate:
            return 1.0

        # ---------------------------------------------
        # Character-level similarity
        # ---------------------------------------------

        sequence_score = (
            SequenceMatcher(
                None,
                query,
                candidate
            ).ratio()
        )

        # ---------------------------------------------
        # Token-level similarity
        # ---------------------------------------------

        query_tokens = tokenize_title(
            query
        )

        candidate_tokens = tokenize_title(
            candidate
        )

        intersection = (
            query_tokens
            & candidate_tokens
        )

        union = (
            query_tokens
            | candidate_tokens
        )

        if query_tokens:

            query_coverage = (
                len(intersection)
                / len(query_tokens)
            )

        else:

            query_coverage = 0.0

        if union:

            jaccard_score = (
                len(intersection)
                / len(union)
            )

        else:

            jaccard_score = 0.0

        combined_score = (
            sequence_score * 0.65
            + query_coverage * 0.25
            + jaccard_score * 0.10
        )

        # ---------------------------------------------
        # Substring bonus
        # ---------------------------------------------

        substring_score = 0.0

        if (
            query in candidate
            or candidate in query
        ):

            shorter_length = min(
                len(query),
                len(candidate)
            )

            longer_length = max(
                len(query),
                len(candidate)
            )

            if longer_length:

                coverage = (
                    shorter_length
                    / longer_length
                )

                substring_score = (
                    0.80
                    + 0.20 * coverage
                )

        return clamp(
            max(
                sequence_score,
                combined_score,
                substring_score
            )
        )

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def search_titles(
        self,
        query,
        limit=DEFAULT_SEARCH_LIMIT,
        min_score=SEARCH_MIN_SCORE
    ):
        """
        Search anime titles using fuzzy matching.

        Returns
        -------
        list[dict]

        Example:

            recommender.search_titles(
                "steins gate"
            )
        """

        if not isinstance(limit, int):

            raise TypeError(
                "limit must be an integer."
            )

        if limit <= 0:

            raise ValueError(
                "limit must be greater than 0."
            )

        normalized_query = normalize_title(
            query
        )

        if not normalized_query:
            return []

        # ---------------------------------------------
        # Exact match
        # ---------------------------------------------

        if normalized_query in self.search_index:

            index = self.search_index[
                normalized_query
            ][0]

            row = self.df.iloc[
                index
            ]

            return [
                {
                    "anime_id": int(
                        row["anime_id"]
                    ),
                    "name": row["name"],
                    "score": 1.0,
                    "matched_title": (
                        normalized_query
                    ),
                    "exact_match": True
                }
            ]

        # Avoid aggressive fuzzy matching on extremely
        # short input such as "a" or "go".
        if len(normalized_query) < 3:
            return []

        results_by_anime = {}

        # ---------------------------------------------
        # Score every searchable title
        # ---------------------------------------------

        for search_term in self.search_terms:

            score = self._search_score(
                normalized_query,
                search_term
            )

            if score < min_score:
                continue

            for index in self.search_index[
                search_term
            ]:

                row = self.df.iloc[
                    index
                ]

                anime_id = int(
                    row["anime_id"]
                )

                existing = results_by_anime.get(
                    anime_id
                )

                result = {
                    "anime_id": anime_id,
                    "name": row["name"],
                    "score": round(
                        score,
                        4
                    ),
                    "matched_title": (
                        search_term
                    ),
                    "exact_match": False
                }

                if (
                    existing is None
                    or score > existing["score"]
                ):

                    results_by_anime[
                        anime_id
                    ] = result

        results = list(
            results_by_anime.values()
        )

        results.sort(
            key=lambda result: (
                result["score"],
                result["name"]
            ),
            reverse=True
        )

        return results[
            :limit
        ]

    # -----------------------------------------------------
    # Title resolution
    # -----------------------------------------------------

    def resolve_title(
        self,
        query,
        allow_fuzzy=True
    ):
        """
        Resolve user input to an anime record.

        Exact matches are always preferred.

        Fuzzy matches are automatically accepted only
        when confidence is sufficiently high.

        Returns
        -------
        dict | None
        """

        normalized_query = normalize_title(
            query
        )

        if not normalized_query:
            return None

        # ---------------------------------------------
        # Exact
        # ---------------------------------------------

        if normalized_query in self.search_index:

            indices = self.search_index[
                normalized_query
            ]

            index = indices[0]

            row = self.df.iloc[
                index
            ]

            return {
                "anime_id": int(
                    row["anime_id"]
                ),
                "name": row["name"],
                "indices": indices,
                "score": 1.0,
                "exact_match": True
            }

        if not allow_fuzzy:
            return None

        # ---------------------------------------------
        # Fuzzy
        # ---------------------------------------------

        suggestions = self.search_titles(
            query,
            limit=2,
            min_score=AUTO_RESOLVE_MIN_SCORE
        )

        if not suggestions:
            return None

        best = suggestions[0]

        if (
            best["score"]
            < AUTO_RESOLVE_MIN_SCORE
        ):

            return None

        # If two different results are nearly identical
        # in confidence, don't silently guess.
        if len(suggestions) > 1:

            second = suggestions[1]

            margin = (
                best["score"]
                - second["score"]
            )

            if margin < AUTO_RESOLVE_MARGIN:

                return None

        canonical_normalized = normalize_title(
            best["name"]
        )

        indices = self.indices.get(
            canonical_normalized,
            []
        )

        # If the result came from an alternative title,
        # fall back to its anime ID.
        if not indices:

            matching_rows = self.df.index[
                self.df["anime_id"]
                == best["anime_id"]
            ].tolist()

            indices = matching_rows

        if not indices:
            return None

        return {
            "anime_id": best["anime_id"],
            "name": best["name"],
            "indices": indices,
            "score": best["score"],
            "exact_match": False
        }

    # -----------------------------------------------------
    # Ranking signals
    # -----------------------------------------------------

    def _rating_score(
        self,
        row
    ):

        rating = row.get(
            "rating"
        )

        if pd.isna(rating):
            return 0.0

        try:

            rating = float(
                rating
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.0

        return clamp(
            rating / 10.0
        )

    def _popularity_score(
        self,
        row
    ):

        if (
            "members" not in self.df.columns
            or self.max_log_members <= 0
        ):

            return 0.0

        members = row.get(
            "members"
        )

        if pd.isna(members):
            return 0.0

        try:

            members = max(
                float(members),
                0.0
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.0

        if members <= 0:
            return 0.0

        score = (
            math.log1p(members)
            / self.max_log_members
        )

        return clamp(
            score
        )

    def _final_score(
        self,
        similarity_score,
        rating_score,
        popularity_score
    ):

        score = (
            similarity_score
            * SIMILARITY_WEIGHT
            +
            rating_score
            * RATING_WEIGHT
            +
            popularity_score
            * POPULARITY_WEIGHT
        )

        return clamp(
            score
        )

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    def recommend(
        self,
        anime_name,
        top_n=10,
        allow_fuzzy=True
    ):
        """
        Return reranked content recommendations.

        Titles may optionally be resolved using fuzzy
        matching.
        """

        if not isinstance(top_n, int):

            raise TypeError(
                "top_n must be an integer."
            )

        if top_n <= 0:

            raise ValueError(
                "top_n must be greater than 0."
            )

        resolution = self.resolve_title(
            anime_name,
            allow_fuzzy=allow_fuzzy
        )

        if resolution is None:
            return []

        matching_indices = resolution[
            "indices"
        ]

        query_index = matching_indices[
            0
        ]

        total_anime = len(
            self.df
        )

        if total_anime <= 1:
            return []

        source_indices = set(
            matching_indices
        )

        desired_candidates = max(
            MIN_CANDIDATE_POOL,
            top_n * CANDIDATE_MULTIPLIER
        )

        requested_neighbors = min(
            total_anime,
            desired_candidates
            + len(source_indices)
            + 5
        )

        distances, neighbor_indices = (
            self.neighbor_model.kneighbors(
                self.tfidf_matrix[
                    query_index
                ],
                n_neighbors=(
                    requested_neighbors
                )
            )
        )

        candidates = []

        for distance, index in zip(
            distances[0],
            neighbor_indices[0]
        ):

            index = int(
                index
            )

            if index in source_indices:
                continue

            row = self.df.iloc[
                index
            ]

            similarity_score = clamp(
                1.0 - float(distance)
            )

            rating_score = (
                self._rating_score(
                    row
                )
            )

            popularity_score = (
                self._popularity_score(
                    row
                )
            )

            final_score = (
                self._final_score(
                    similarity_score,
                    rating_score,
                    popularity_score
                )
            )

            members = None

            if "members" in self.df.columns:

                row_members = row.get(
                    "members"
                )

                if pd.notna(
                    row_members
                ):

                    try:

                        members = int(
                            row_members
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        members = None

            candidate = {
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
                ),

                "members": members,

                "similarity_score": round(
                    similarity_score,
                    4
                ),

                "rating_score": round(
                    rating_score,
                    4
                ),

                "popularity_score": round(
                    popularity_score,
                    4
                ),

                "final_score": round(
                    final_score,
                    4
                )
            }

            candidates.append(
                candidate
            )

        candidates.sort(
            key=lambda anime: (
                anime["final_score"],
                anime["similarity_score"],
                anime["rating"] or 0
            ),
            reverse=True
        )

        return candidates[
            :top_n
        ]


# ---------------------------------------------------------
# Demo
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

    # Intentionally imperfect title.
    anime_name = "steins gate"

    print(
        f"\nSearching for: "
        f"{anime_name}\n"
    )

    search_results = (
        recommender.search_titles(
            anime_name,
            limit=5
        )
    )

    print(
        "Search results:"
    )

    for position, result in enumerate(
        search_results,
        start=1
    ):

        print(
            f"{position}. "
            f"{result['name']} "
            f"(match: {result['score']:.4f})"
        )

    resolution = (
        recommender.resolve_title(
            anime_name
        )
    )

    if resolution is None:

        print(
            "\nCould not safely resolve title."
        )

        return

    print(
        f"\nResolved to: "
        f"{resolution['name']}"
    )

    print(
        f"Match score: "
        f"{resolution['score']:.4f}"
    )

    print(
        f"Exact match: "
        f"{resolution['exact_match']}"
    )

    print(
        f"\nRecommendations for "
        f"{resolution['name']}:\n"
    )

    recommendations = (
        recommender.recommend(
            anime_name,
            top_n=10
        )
    )

    if not recommendations:

        print(
            "No recommendations available."
        )

        return

    for position, anime in enumerate(
        recommendations,
        start=1
    ):

        rating = (
            anime["rating"]
            if anime["rating"] is not None
            else "N/A"
        )

        print(
            f"{position}. "
            f"{anime['name']}\n"
            f"   Rating: {rating}\n"
            f"   Similarity: "
            f"{anime['similarity_score']:.4f}\n"
            f"   Final score: "
            f"{anime['final_score']:.4f}\n"
        )


if __name__ == "__main__":
    main()
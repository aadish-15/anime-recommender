from pathlib import Path
from datetime import datetime, timezone
import argparse
import math
import random
import time

import pandas as pd
from tqdm import tqdm

from content_based import (
    ContentBasedRecommender,
    normalize_title,
    SIMILARITY_WEIGHT,
    RATING_WEIGHT,
    POPULARITY_WEIGHT
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rating.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "content_evaluation.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "content_evaluation_summary.csv"
)


# ---------------------------------------------------------
# Default evaluation configuration
# ---------------------------------------------------------

DEFAULT_SAMPLE_USERS = 500

DEFAULT_TOP_K = 10

# Anime rated 8, 9 or 10 are treated as "liked".
DEFAULT_LIKE_THRESHOLD = 8

# User must have enough liked anime so that one can be
# used as a seed and several others remain relevant.
MIN_LIKED_ANIME = 5

RATING_CHUNK_SIZE = 500_000

RANDOM_SEED = 42


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def precision_at_k(
    recommended_ids,
    relevant_ids,
    k
):
    """
    Fraction of Top-K recommendations that are relevant.
    """

    recommended_ids = recommended_ids[:k]

    if not recommended_ids:
        return 0.0

    hits = sum(
        1
        for anime_id in recommended_ids
        if anime_id in relevant_ids
    )

    return hits / k


def recall_at_k(
    recommended_ids,
    relevant_ids,
    k
):
    """
    Fraction of relevant anime recovered in Top-K.
    """

    if not relevant_ids:
        return 0.0

    recommended_ids = recommended_ids[:k]

    hits = sum(
        1
        for anime_id in recommended_ids
        if anime_id in relevant_ids
    )

    return hits / len(
        relevant_ids
    )


def hit_rate_at_k(
    recommended_ids,
    relevant_ids,
    k
):
    """
    1 if at least one relevant anime appears in Top-K.
    """

    recommended_ids = recommended_ids[:k]

    for anime_id in recommended_ids:

        if anime_id in relevant_ids:
            return 1.0

    return 0.0


def reciprocal_rank_at_k(
    recommended_ids,
    relevant_ids,
    k
):
    """
    Reciprocal rank of the first relevant recommendation.
    """

    for rank, anime_id in enumerate(
        recommended_ids[:k],
        start=1
    ):

        if anime_id in relevant_ids:

            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    recommended_ids,
    relevant_ids,
    k
):
    """
    Normalized Discounted Cumulative Gain.

    Uses binary relevance:

        relevant = 1
        irrelevant = 0
    """

    if not relevant_ids:
        return 0.0

    dcg = 0.0

    for rank, anime_id in enumerate(
        recommended_ids[:k],
        start=1
    ):

        if anime_id in relevant_ids:

            dcg += (
                1.0
                / math.log2(
                    rank + 1
                )
            )

    ideal_hits = min(
        len(relevant_ids),
        k
    )

    idcg = sum(
        1.0
        / math.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------
# Ratings loading
# ---------------------------------------------------------

def load_liked_ratings(
    valid_anime_ids,
    like_threshold
):
    """
    Read rating.csv in chunks.

    Only retain:

    - explicit ratings
    - ratings >= like_threshold
    - anime that exist in our recommendation catalog
    """

    if not RATINGS_FILE.exists():

        raise FileNotFoundError(
            "Ratings dataset not found:\n"
            f"{RATINGS_FILE}\n\n"
            "Expected:\n"
            "data/raw/rating.csv"
        )

    print(
        "\nLoading liked user interactions..."
    )

    liked_chunks = []

    total_rows = 0
    liked_rows = 0

    reader = pd.read_csv(
        RATINGS_FILE,
        usecols=[
            "user_id",
            "anime_id",
            "rating"
        ],
        chunksize=RATING_CHUNK_SIZE
    )

    for chunk in tqdm(
        reader,
        desc="Reading ratings"
    ):

        total_rows += len(
            chunk
        )

        chunk["user_id"] = pd.to_numeric(
            chunk["user_id"],
            errors="coerce"
        )

        chunk["anime_id"] = pd.to_numeric(
            chunk["anime_id"],
            errors="coerce"
        )

        chunk["rating"] = pd.to_numeric(
            chunk["rating"],
            errors="coerce"
        )

        chunk = chunk.dropna(
            subset=[
                "user_id",
                "anime_id",
                "rating"
            ]
        )

        chunk["user_id"] = (
            chunk["user_id"]
            .astype("int64")
        )

        chunk["anime_id"] = (
            chunk["anime_id"]
            .astype("int64")
        )

        # Kaggle rating = -1 means interaction without
        # an explicit rating, so it is excluded here.
        chunk = chunk[
            chunk["rating"]
            >= like_threshold
        ]

        chunk = chunk[
            chunk["anime_id"].isin(
                valid_anime_ids
            )
        ]

        if chunk.empty:
            continue

        liked_rows += len(
            chunk
        )

        liked_chunks.append(
            chunk[
                [
                    "user_id",
                    "anime_id"
                ]
            ]
        )

    if not liked_chunks:

        raise RuntimeError(
            "No liked interactions were found."
        )

    liked_df = pd.concat(
        liked_chunks,
        ignore_index=True
    )

    # A user should only contribute one relevance signal
    # per anime.
    liked_df = liked_df.drop_duplicates(
        subset=[
            "user_id",
            "anime_id"
        ]
    )

    print(
        f"\nRatings scanned: "
        f"{total_rows:,}"
    )

    print(
        f"Liked interactions found: "
        f"{liked_rows:,}"
    )

    print(
        f"Unique liked interactions: "
        f"{len(liked_df):,}"
    )

    return liked_df


# ---------------------------------------------------------
# User sampling
# ---------------------------------------------------------

def sample_evaluation_users(
    liked_df,
    sample_users,
    random_seed
):
    """
    Find users with enough liked anime and sample
    a reproducible subset.
    """

    user_counts = (
        liked_df
        .groupby("user_id")
        ["anime_id"]
        .nunique()
    )

    eligible_users = (
        user_counts[
            user_counts
            >= MIN_LIKED_ANIME
        ]
        .index
        .tolist()
    )

    print(
        f"\nEligible users "
        f"(>= {MIN_LIKED_ANIME} liked anime): "
        f"{len(eligible_users):,}"
    )

    if not eligible_users:

        raise RuntimeError(
            "No users have enough liked anime "
            "for evaluation."
        )

    rng = random.Random(
        random_seed
    )

    if sample_users >= len(
        eligible_users
    ):

        selected_users = eligible_users

    else:

        selected_users = rng.sample(
            eligible_users,
            sample_users
        )

    print(
        f"Evaluation users selected: "
        f"{len(selected_users):,}"
    )

    return selected_users


# ---------------------------------------------------------
# Duplicate-title protection
# ---------------------------------------------------------

def is_safe_seed(
    recommender,
    anime_id
):
    """
    Skip ambiguous duplicate-title records as seeds.
    """

    matches = recommender.df[
        recommender.df["anime_id"]
        == anime_id
    ]

    if matches.empty:
        return False

    title = matches.iloc[0][
        "name"
    ]

    normalized = normalize_title(
        title
    )

    indices = recommender.indices.get(
        normalized,
        []
    )

    return len(indices) == 1


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(
    recommender,
    liked_df,
    selected_users,
    top_k,
    random_seed
):
    """
    Evaluate the content recommender.

    For each user:

        1. Pick one liked anime as the seed.
        2. Treat the user's other liked anime as relevant.
        3. Generate Top-K recommendations.
        4. Calculate ranking metrics.
    """

    rng = random.Random(
        random_seed
    )

    selected_set = set(
        selected_users
    )

    evaluation_df = liked_df[
        liked_df["user_id"].isin(
            selected_set
        )
    ].copy()

    user_likes = (
        evaluation_df
        .groupby("user_id")
        ["anime_id"]
        .apply(
            lambda values: list(
                dict.fromkeys(
                    int(value)
                    for value in values
                )
            )
        )
        .to_dict()
    )

    anime_title_lookup = dict(
        zip(
            recommender.df[
                "anime_id"
            ].astype(int),
            recommender.df[
                "name"
            ]
        )
    )

    results = []

    recommended_catalog = set()

    skipped_users = 0

    for user_id in tqdm(
        selected_users,
        desc="Evaluating users"
    ):

        liked_anime = user_likes.get(
            user_id,
            []
        )

        if len(liked_anime) < MIN_LIKED_ANIME:

            skipped_users += 1
            continue

        # ---------------------------------------------
        # Find seeds without duplicate-title ambiguity
        # ---------------------------------------------

        safe_seeds = [
            anime_id
            for anime_id in liked_anime
            if is_safe_seed(
                recommender,
                anime_id
            )
        ]

        if not safe_seeds:

            skipped_users += 1
            continue

        seed_id = rng.choice(
            safe_seeds
        )

        seed_title = anime_title_lookup.get(
            seed_id
        )

        if not seed_title:

            skipped_users += 1
            continue

        relevant_ids = set(
            liked_anime
        )

        relevant_ids.discard(
            seed_id
        )

        if not relevant_ids:

            skipped_users += 1
            continue

        # ---------------------------------------------
        # Recommend
        # ---------------------------------------------

        recommendations = (
            recommender.recommend(
                seed_title,
                top_n=top_k,
                allow_fuzzy=False
            )
        )

        recommended_ids = [
            anime["anime_id"]
            for anime in recommendations
            if anime["anime_id"] is not None
        ]

        recommended_catalog.update(
            recommended_ids
        )

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        precision = precision_at_k(
            recommended_ids,
            relevant_ids,
            top_k
        )

        recall = recall_at_k(
            recommended_ids,
            relevant_ids,
            top_k
        )

        hit_rate = hit_rate_at_k(
            recommended_ids,
            relevant_ids,
            top_k
        )

        reciprocal_rank = (
            reciprocal_rank_at_k(
                recommended_ids,
                relevant_ids,
                top_k
            )
        )

        ndcg = ndcg_at_k(
            recommended_ids,
            relevant_ids,
            top_k
        )

        hit_count = sum(
            1
            for anime_id in recommended_ids
            if anime_id in relevant_ids
        )

        results.append(
            {
                "user_id": int(
                    user_id
                ),
                "seed_anime_id": int(
                    seed_id
                ),
                "seed_title": (
                    seed_title
                ),
                "relevant_count": len(
                    relevant_ids
                ),
                "recommendation_count": len(
                    recommended_ids
                ),
                "hit_count": hit_count,
                "precision_at_k": (
                    precision
                ),
                "recall_at_k": (
                    recall
                ),
                "hit_rate_at_k": (
                    hit_rate
                ),
                "mrr_at_k": (
                    reciprocal_rank
                ),
                "ndcg_at_k": (
                    ndcg
                )
            }
        )

    if not results:

        raise RuntimeError(
            "Evaluation produced no valid results."
        )

    results_df = pd.DataFrame(
        results
    )

    catalog_size = len(
        recommender.df
    )

    catalog_coverage = (
        len(recommended_catalog)
        / catalog_size
        if catalog_size
        else 0.0
    )

    return (
        results_df,
        catalog_coverage,
        skipped_users
    )


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

def print_report(
    results_df,
    catalog_coverage,
    skipped_users,
    top_k
):

    evaluated_users = len(
        results_df
    )

    print(
        "\n"
        "========================================"
    )

    print(
        " CONTENT RECOMMENDER EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"\nEvaluated users: "
        f"{evaluated_users:,}"
    )

    print(
        f"Skipped users: "
        f"{skipped_users:,}"
    )

    print(
        f"Top-K: {top_k}"
    )

    print(
        "\nRanking metrics:"
    )

    print(
        f"Precision@{top_k}: "
        f"{results_df['precision_at_k'].mean():.4f}"
    )

    print(
        f"Recall@{top_k}: "
        f"{results_df['recall_at_k'].mean():.4f}"
    )

    print(
        f"Hit Rate@{top_k}: "
        f"{results_df['hit_rate_at_k'].mean():.4f}"
    )

    print(
        f"MRR@{top_k}: "
        f"{results_df['mrr_at_k'].mean():.4f}"
    )

    print(
        f"NDCG@{top_k}: "
        f"{results_df['ndcg_at_k'].mean():.4f}"
    )

    print(
        "\nCatalog metric:"
    )

    print(
        f"Catalog Coverage: "
        f"{catalog_coverage:.4f} "
        f"({catalog_coverage:.2%})"
    )

    print(
        "\nAverage hits per user:"
    )

    print(
        f"{results_df['hit_count'].mean():.4f}"
    )

    print(
        "\n========================================"
    )


# ---------------------------------------------------------
# Save evaluation summary
# ---------------------------------------------------------

def save_evaluation_summary(
    results_df,
    catalog_coverage,
    skipped_users,
    top_k,
    sample_users,
    like_threshold,
    elapsed_seconds
):
    """
    Save one aggregate benchmark row.

    Each future evaluation run is appended so model
    changes can be compared over time.
    """

    summary = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": "content_based",

        "requested_users": int(
            sample_users
        ),

        "evaluated_users": int(
            len(results_df)
        ),

        "skipped_users": int(
            skipped_users
        ),

        "top_k": int(
            top_k
        ),

        "like_threshold": int(
            like_threshold
        ),

        "min_liked_anime": int(
            MIN_LIKED_ANIME
        ),

        "similarity_weight": float(
            SIMILARITY_WEIGHT
        ),

        "rating_weight": float(
            RATING_WEIGHT
        ),

        "popularity_weight": float(
            POPULARITY_WEIGHT
        ),

        "precision_at_k": float(
            results_df[
                "precision_at_k"
            ].mean()
        ),

        "recall_at_k": float(
            results_df[
                "recall_at_k"
            ].mean()
        ),

        "hit_rate_at_k": float(
            results_df[
                "hit_rate_at_k"
            ].mean()
        ),

        "mrr_at_k": float(
            results_df[
                "mrr_at_k"
            ].mean()
        ),

        "ndcg_at_k": float(
            results_df[
                "ndcg_at_k"
            ].mean()
        ),

        "average_hits": float(
            results_df[
                "hit_count"
            ].mean()
        ),

        "catalog_coverage": float(
            catalog_coverage
        ),

        "elapsed_seconds": round(
            float(elapsed_seconds),
            2
        )
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if SUMMARY_FILE.exists():

        summary_df.to_csv(
            SUMMARY_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        summary_df.to_csv(
            SUMMARY_FILE,
            index=False
        )

    print(
        f"\nSaved evaluation summary:\n"
        f"{SUMMARY_FILE}"
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the content-based anime "
            "recommendation model."
        )
    )

    parser.add_argument(
        "--users",
        type=int,
        default=DEFAULT_SAMPLE_USERS,
        help=(
            "Number of users to sample "
            f"(default: {DEFAULT_SAMPLE_USERS})"
        )
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of recommendations per user "
            f"(default: {DEFAULT_TOP_K})"
        )
    )

    parser.add_argument(
        "--like-threshold",
        type=int,
        default=DEFAULT_LIKE_THRESHOLD,
        help=(
            "Minimum explicit rating considered liked "
            f"(default: {DEFAULT_LIKE_THRESHOLD})"
        )
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    if args.users <= 0:

        raise ValueError(
            "--users must be greater than 0."
        )

    if args.k <= 0:

        raise ValueError(
            "--k must be greater than 0."
        )

    if not 1 <= args.like_threshold <= 10:

        raise ValueError(
            "--like-threshold must be between 1 and 10."
        )

    start_time = time.perf_counter()

    print(
        "Loading recommender..."
    )

    recommender = (
        ContentBasedRecommender()
    )

    valid_anime_ids = set(
        recommender.df[
            "anime_id"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    liked_df = load_liked_ratings(
        valid_anime_ids=valid_anime_ids,
        like_threshold=(
            args.like_threshold
        )
    )

    selected_users = (
        sample_evaluation_users(
            liked_df=liked_df,
            sample_users=args.users,
            random_seed=RANDOM_SEED
        )
    )

    results_df, coverage, skipped = (
        evaluate(
            recommender=recommender,
            liked_df=liked_df,
            selected_users=selected_users,
            top_k=args.k,
            random_seed=RANDOM_SEED
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print_report(
        results_df=results_df,
        catalog_coverage=coverage,
        skipped_users=skipped,
        top_k=args.k
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    save_evaluation_summary(
        results_df=results_df,
        catalog_coverage=coverage,
        skipped_users=skipped,
        top_k=args.k,
        sample_users=args.users,
        like_threshold=args.like_threshold,
        elapsed_seconds=elapsed
    )

    print(
        f"\nSaved detailed results:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nEvaluation completed in "
        f"{elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
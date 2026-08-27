from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

ANIME_FILE = (
    RAW_DIR
    / "anime.csv"
)

RATINGS_FILE = (
    RAW_DIR
    / "rating.csv"
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_anime_columns(df):
    """
    Validate the Kaggle anime metadata structure.
    """

    required_columns = {
        "anime_id",
        "name",
        "genre",
        "type",
        "episodes",
        "rating",
        "members"
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
            "anime.csv is missing required columns: "
            f"{missing}"
        )


def validate_rating_columns(df):

    required_columns = {
        "user_id",
        "anime_id",
        "rating"
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
            "rating.csv is missing required columns: "
            f"{missing}"
        )


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------

def load_anime_data():
    """
    Load and validate anime.csv.
    """

    if not ANIME_FILE.exists():

        raise FileNotFoundError(
            "Anime dataset not found:\n"
            f"{ANIME_FILE}\n\n"
            "Place anime.csv inside data/raw/."
        )

    df = pd.read_csv(
        ANIME_FILE
    )

    validate_anime_columns(
        df
    )

    return df


def load_ratings_data():

    if not RATINGS_FILE.exists():

        raise FileNotFoundError(
            "Ratings dataset not found:\n"
            f"{RATINGS_FILE}\n\n"
            "Place rating.csv inside data/raw/."
        )

    df = pd.read_csv(
        RATINGS_FILE
    )

    validate_rating_columns(
        df
    )

    return df


def load_data():

    anime_df = (
        load_anime_data()
    )

    ratings_df = (
        load_ratings_data()
    )

    return (
        anime_df,
        ratings_df
    )


# ---------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------

def print_dataset_summary(
    anime_df,
    ratings_df
):

    print(
        "\nKaggle dataset loaded successfully."
    )

    print(
        "\nAnime metadata"
    )

    print(
        f"Rows: "
        f"{len(anime_df):,}"
    )

    print(
        f"Columns: "
        f"{len(anime_df.columns):,}"
    )

    print(
        f"Unique anime IDs: "
        f"{anime_df['anime_id'].nunique():,}"
    )

    duplicate_anime_ids = (
        anime_df.duplicated(
            subset=["anime_id"]
        ).sum()
    )

    print(
        f"Duplicate anime IDs: "
        f"{duplicate_anime_ids:,}"
    )

    print(
        "\nRatings"
    )

    print(
        f"Rows: "
        f"{len(ratings_df):,}"
    )

    print(
        f"Columns: "
        f"{len(ratings_df.columns):,}"
    )

    print(
        f"Unique users: "
        f"{ratings_df['user_id'].nunique():,}"
    )

    print(
        f"Unique rated anime: "
        f"{ratings_df['anime_id'].nunique():,}"
    )

    # -----------------------------------------------------
    # Kaggle convention
    # -----------------------------------------------------

    unrated_count = (
        ratings_df["rating"] == -1
    ).sum()

    actual_rating_count = (
        ratings_df["rating"]
        .between(
            1,
            10
        )
    ).sum()

    print(
        f"Explicit ratings (1-10): "
        f"{actual_rating_count:,}"
    )

    print(
        f"Unrated interactions (-1): "
        f"{unrated_count:,}"
    )

    # -----------------------------------------------------
    # Referential integrity
    # -----------------------------------------------------

    anime_ids = set(
        anime_df[
            "anime_id"
        ].dropna()
    )

    rating_anime_ids = set(
        ratings_df[
            "anime_id"
        ].dropna()
    )

    unknown_anime_ids = (
        rating_anime_ids
        - anime_ids
    )

    print(
        f"Rating anime IDs missing from anime.csv: "
        f"{len(unknown_anime_ids):,}"
    )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    print(
        "\nAnime missing values:"
    )

    anime_missing = (
        anime_df
        .isna()
        .sum()
    )

    anime_missing = (
        anime_missing[
            anime_missing > 0
        ]
    )

    if anime_missing.empty:

        print(
            "None"
        )

    else:

        print(
            anime_missing
        )

    print(
        "\nRatings missing values:"
    )

    ratings_missing = (
        ratings_df
        .isna()
        .sum()
    )

    ratings_missing = (
        ratings_missing[
            ratings_missing > 0
        ]
    )

    if ratings_missing.empty:

        print(
            "None"
        )

    else:

        print(
            ratings_missing
        )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

def main():

    anime_df, ratings_df = (
        load_data()
    )

    print_dataset_summary(
        anime_df,
        ratings_df
    )


if __name__ == "__main__":
    main()
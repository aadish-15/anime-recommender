from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "anime.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "anime_clean.csv"
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_columns(df):

    required_columns = {
        "anime_id",
        "name",
        "genre",
        "episodes",
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
            "Input dataset is missing "
            f"required columns: {missing}"
        )


# ---------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------

def clean_genres(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(",", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )


# ---------------------------------------------------------
# Transformation
# ---------------------------------------------------------

def clean_anime_data(
    input_path=INPUT_FILE
):

    input_path = Path(
        input_path
    )

    if not input_path.exists():

        raise FileNotFoundError(
            "Anime dataset not found:\n"
            f"{input_path}\n\n"
            "Place anime.csv inside data/raw/."
        )

    print(
        f"Loading:\n{input_path}"
    )

    df = pd.read_csv(
        input_path
    )

    print(
        f"\nOriginal rows: {len(df):,}"
    )

    validate_columns(
        df
    )

    # -----------------------------------------------------
    # Anime ID
    # -----------------------------------------------------

    df["anime_id"] = pd.to_numeric(
        df["anime_id"],
        errors="coerce"
    )

    invalid_ids = (
        df["anime_id"].isna()
    ).sum()

    if invalid_ids:

        print(
            f"Removing {invalid_ids:,} rows "
            "with invalid anime_id"
        )

        df = df.dropna(
            subset=["anime_id"]
        )

    df["anime_id"] = (
        df["anime_id"]
        .astype("int64")
    )

    # -----------------------------------------------------
    # Remove duplicate records
    # -----------------------------------------------------

    duplicate_ids = (
        df.duplicated(
            subset=["anime_id"]
        )
    ).sum()

    if duplicate_ids:

        print(
            f"Removing {duplicate_ids:,} "
            "duplicate anime_id rows"
        )

        df = df.drop_duplicates(
            subset=["anime_id"],
            keep="first"
        )

    # -----------------------------------------------------
    # Anime name
    # -----------------------------------------------------

    df["name"] = (
        df["name"]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    missing_names = (
        df["name"] == ""
    ).sum()

    if missing_names:

        print(
            f"Removing {missing_names:,} rows "
            "with missing anime names"
        )

        df = df[
            df["name"] != ""
        ].copy()

    # -----------------------------------------------------
    # Genres
    # -----------------------------------------------------

    df["genre"] = (
        df["genre"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["genre_clean"] = (
        clean_genres(
            df["genre"]
        )
    )

    # -----------------------------------------------------
    # Episodes
    # -----------------------------------------------------

    df["episodes"] = pd.to_numeric(
        df["episodes"],
        errors="coerce"
    )

    df["episodes"] = (
        df["episodes"]
        .round()
        .astype("Int64")
    )

    # -----------------------------------------------------
    # Rating
    # -----------------------------------------------------

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Ratings outside the expected MAL range are invalid.
    invalid_rating_mask = (
        df["rating"].notna()
        & ~df["rating"].between(
            0,
            10
        )
    )

    invalid_ratings = (
        invalid_rating_mask.sum()
    )

    if invalid_ratings:

        print(
            f"Setting {invalid_ratings:,} "
            "invalid ratings to missing"
        )

        df.loc[
            invalid_rating_mask,
            "rating"
        ] = pd.NA

    # -----------------------------------------------------
    # Members
    # -----------------------------------------------------

    if "members" in df.columns:

        df["members"] = pd.to_numeric(
            df["members"],
            errors="coerce"
        ).astype("Int64")

    # -----------------------------------------------------
    # Type
    # -----------------------------------------------------

    if "type" in df.columns:

        df["type"] = (
            df["type"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # -----------------------------------------------------
    # Final cleanup
    # -----------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    print(
        f"Cleaned rows: {len(df):,}"
    )

    print(
        "\nMissing values:"
    )

    print(
        df[
            [
                "genre",
                "episodes",
                "rating"
            ]
        ]
        .isna()
        .sum()
    )

    return df


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    anime_df = (
        clean_anime_data()
    )

    anime_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nCleaning complete."
    )

    print(
        f"Saved:\n{OUTPUT_FILE}"
    )
if __name__ == "__main__":
    main()
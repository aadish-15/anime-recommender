from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ANIME_FILE = (
    PROCESSED_DIR
    / "anime_clean.csv"
)

JIKAN_FILE = (
    PROCESSED_DIR
    / "jikan_enriched.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "anime_features.csv"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MIN_ENRICHMENT_COVERAGE = 0.90


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_text(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


def validate_anime_data(df):

    required_columns = {
        "anime_id",
        "name",
        "genre",
        "genre_clean"
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
            "anime_clean.csv is missing "
            f"required columns: {missing}"
        )


def prepare_jikan_data(df):

    required_columns = {
        "anime_id"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "jikan_enriched.csv does not contain "
            "anime_id."
        )

    df = df.copy()

    df["anime_id"] = pd.to_numeric(
        df["anime_id"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["anime_id"]
    )

    df["anime_id"] = (
        df["anime_id"]
        .astype("int64")
    )

    duplicate_count = (
        df.duplicated(
            subset=["anime_id"]
        )
    ).sum()

    if duplicate_count:

        print(
            f"Removing {duplicate_count:,} duplicate "
            "Jikan records"
        )

        df = df.drop_duplicates(
            subset=["anime_id"],
            keep="last"
        )

    expected_columns = [
        "synopsis",
        "year",
        "season",
        "score",
        "genres_jikan",
        "themes",
        "studios",
        "demographics"
    ]

    for column in expected_columns:

        if column not in df.columns:
            df[column] = pd.NA

    return df


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------

def load_anime_data():

    if not ANIME_FILE.exists():

        raise FileNotFoundError(
            "Clean anime dataset not found:\n"
            f"{ANIME_FILE}\n\n"
            "Run:\n"
            "python etl/transform.py"
        )

    anime_df = pd.read_csv(
        ANIME_FILE
    )

    validate_anime_data(
        anime_df
    )

    anime_df["anime_id"] = pd.to_numeric(
        anime_df["anime_id"],
        errors="coerce"
    )

    anime_df = anime_df.dropna(
        subset=["anime_id"]
    )

    anime_df["anime_id"] = (
        anime_df["anime_id"]
        .astype("int64")
    )

    anime_df = anime_df.drop_duplicates(
        subset=["anime_id"],
        keep="first"
    )

    return anime_df


def load_jikan_data():

    if not JIKAN_FILE.exists():

        print(
            "\nNo Jikan enrichment file found."
        )

        print(
            "Building baseline Kaggle-only features."
        )

        return pd.DataFrame(
            columns=["anime_id"]
        )

    try:

        jikan_df = pd.read_csv(
            JIKAN_FILE
        )

    except pd.errors.EmptyDataError:

        print(
            "\nJikan enrichment file is empty."
        )

        return pd.DataFrame(
            columns=["anime_id"]
        )

    if jikan_df.empty:
        return pd.DataFrame(
            columns=["anime_id"]
        )

    return prepare_jikan_data(
        jikan_df
    )


# ---------------------------------------------------------
# Feature construction
# ---------------------------------------------------------

def build_features():

    anime_df = load_anime_data()

    jikan_df = load_jikan_data()

    print(
        f"\nKaggle records: "
        f"{len(anime_df):,}"
    )

    print(
        f"Jikan records: "
        f"{len(jikan_df):,}"
    )

    # -----------------------------------------------------
    # Determine actual enrichment coverage
    # -----------------------------------------------------

    anime_ids = set(
        anime_df["anime_id"]
    )

    jikan_ids = set(
        jikan_df["anime_id"]
    )

    matched_ids = (
        anime_ids
        & jikan_ids
    )

    coverage = (
        len(matched_ids)
        / len(anime_df)
        if len(anime_df)
        else 0
    )

    print(
        f"Jikan coverage: "
        f"{coverage:.2%}"
    )

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    df = anime_df.merge(
        jikan_df,
        on="anime_id",
        how="left",
        validate="one_to_one"
    )

    if len(df) != len(anime_df):

        raise RuntimeError(
            "Feature merge changed the number "
            "of anime records."
        )

    print(
        f"Merged records: "
        f"{len(df):,}"
    )

    # -----------------------------------------------------
    # Normalize text fields
    # -----------------------------------------------------

    text_columns = [
        "genre_clean",
        "genres_jikan",
        "themes",
        "studios",
        "demographics",
        "synopsis"
    ]

    for column in text_columns:

        if column not in df.columns:
            df[column] = ""

        df[column] = normalize_text(
            df[column]
        )

    # -----------------------------------------------------
    # Genre source
    # -----------------------------------------------------

    df["genre_features"] = (
        df["genres_jikan"]
        .where(
            df["genres_jikan"] != "",
            df["genre_clean"]
        )
    )

    # -----------------------------------------------------
    # Feature mode
    # -----------------------------------------------------

    if (
        coverage
        >= MIN_ENRICHMENT_COVERAGE
    ):

        feature_mode = "enriched"

        print(
            "\nFeature mode: ENRICHED"
        )

        print(
            "Jikan coverage is sufficient to use "
            "synopsis/themes/studios/demographics."
        )

        df["feature_text"] = (
            df["genre_features"]
            + " "
            + df["themes"]
            + " "
            + df["studios"]
            + " "
            + df["demographics"]
            + " "
            + df["synopsis"]
        )

    else:

        feature_mode = "baseline"

        print(
            "\nFeature mode: BASELINE"
        )

        print(
            "Jikan coverage is incomplete."
        )

        print(
            "Using consistent Kaggle genre features "
            "for all anime."
        )

        df["feature_text"] = (
            df["genre_clean"]
        )

    # -----------------------------------------------------
    # Final feature cleanup
    # -----------------------------------------------------

    df["feature_text"] = (
        df["feature_text"]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    empty_features = (
        df["feature_text"] == ""
    ).sum()

    if empty_features:

        print(
            f"\nWarning: {empty_features:,} anime "
            "have empty feature_text."
        )

    # Save the mode for inspection/debugging.
    df["feature_mode"] = (
        feature_mode
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nFeature engineering complete."
    )

    print(
        f"Saved:\n{OUTPUT_FILE}"
    )

    print(
        f"\nFeature mode: "
        f"{feature_mode}"
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Jikan coverage: "
        f"{coverage:.2%}"
    )

    # Show one non-empty example.
    examples = df[
        df["feature_text"] != ""
    ]

    if not examples.empty:

        example = examples.iloc[0]

        print(
            "\nExample:"
        )

        print(
            f"Anime: {example['name']}"
        )

        print(
            "Feature text:"
        )

        print(
            example["feature_text"][:800]
        )

    return df


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    build_features()
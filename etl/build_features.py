import pandas as pd

def build_features():

    anime_df = pd.read_csv(
        "data/processed/anime_clean.csv"
    )

    jikan_df = pd.read_csv(
        "data/processed/jikan_enriched.csv"
    )

    print(f"Kaggle records: {len(anime_df)}")
    print(f"Jikan records : {len(jikan_df)}")

    df = anime_df.merge(
        jikan_df,
        on="anime_id",
        how="left"
    )

    print(f"Merged records : {len(df)}")

    text_columns = [
        "genre_clean",
        "genres_jikan",
        "themes",
        "studios",
        "demographics",
        "synopsis"
    ]

    for col in text_columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.lower()
        )

    df["feature_text"] = (
        df["genre_clean"] + " " +
        df["genres_jikan"] + " " +
        df["themes"] + " " +
        df["studios"] + " " +
        df["demographics"] + " " +
        df["synopsis"]
    )

    df["feature_text"] = (
        df["feature_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df.to_csv(
        "data/processed/anime_features.csv",
        index=False
    )

    print("\nFeature engineering complete.")
    print("Saved: data/processed/anime_features.csv")

    print("\nExample feature text:\n")

    print(df.loc[
        0,
        "feature_text"
    ][:800])

if __name__ == "__main__":
    build_features()
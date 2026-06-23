import pandas as pd

def clean_anime_data(input_path="data/raw/anime.csv"):

    df = pd.read_csv(input_path)

    print("Original Shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates(subset=["anime_id"])

    # Fill missing genres
    df["genre"] = df["genre"].fillna("Unknown")

    # Fix ratings
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # Episodes sometimes contain "Unknown"
    df["episodes"] = pd.to_numeric(
        df["episodes"],
        errors="coerce"
    )

    df["episodes"] = df["episodes"].fillna(0)
    # Remove commas from genres
    df["genre_clean"] = (
        df["genre"]
        .str.replace(",", " ", regex=False)
        .str.lower()
    )
    print("Cleaned Shape:", df.shape)
    return df

if __name__ == "__main__":
    anime_df = clean_anime_data()
    anime_df.to_csv(
        "data/processed/anime_clean.csv",
        index=False
    )
    print(
        "\nSaved:",
        "data/processed/anime_clean.csv"
    )
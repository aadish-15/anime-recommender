import pandas as pd

def load_data():
    anime_df = pd.read_csv("data/raw/anime.csv")
    ratings_df = pd.read_csv("data/raw/rating.csv")

    print("\nAnime Dataset")
    print(anime_df.head())

    print("\nRatings Dataset")
    print(ratings_df.head())

    print("\nAnime Shape:", anime_df.shape)
    print("Ratings Shape:", ratings_df.shape)

    return anime_df, ratings_df

if __name__ == "__main__":
    load_data()
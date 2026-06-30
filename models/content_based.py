import joblib


class ContentBasedRecommender:

    def __init__(self):

        self.df = joblib.load(
            "models/anime_metadata.pkl"
        )

        self.similarity_matrix = joblib.load(
            "models/content_similarity.pkl"
        )

        self.indices = joblib.load(
            "models/anime_indices.pkl"
        )

    def recommend(self, anime_name, top_n=10):

        if anime_name not in self.indices:
            return []

        idx = self.indices[anime_name]

        similarity_scores = list(
            enumerate(self.similarity_matrix[idx])
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = similarity_scores[1:top_n + 1]

        anime_indices = [
            i[0]
            for i in similarity_scores
        ]

        return self.df.iloc[
            anime_indices
        ][["name", "genre", "rating"]]


if __name__ == "__main__":

    recommender = ContentBasedRecommender()

    print(
        recommender.recommend(
            "Steins;Gate"
        )
    )

import time

start = time.time()
recommender = ContentBasedRecommender()
print(
    f"Loaded in {time.time() - start:.2f}s"
)
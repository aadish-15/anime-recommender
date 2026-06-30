import os
import time
import requests
import pandas as pd

from tqdm import tqdm

INPUT_FILE = "data/processed/anime_clean.csv"
OUTPUT_FILE = "data/processed/jikan_enriched.csv"

SAVE_INTERVAL = 100
REQUEST_DELAY = 1

MAX_RETRIES = 3
BACKOFF_FACTOR = 2

session = requests.Session()

session.headers.update({
    "User-Agent": "Anime-Recommender-ETL/1.0"
})

def fetch_jikan_metadata(anime_id):

    url = f"https://api.jikan.moe/v4/anime/{anime_id}"

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                url,
                timeout=15
            )

            if response.status_code == 200:

                data = response.json()["data"]

                genres = " ".join(
                    genre["name"]
                    for genre in data.get(
                        "genres",
                        []
                    )
                )

                themes = " ".join(
                    theme["name"]
                    for theme in data.get(
                        "themes",
                        []
                    )
                )

                studios = " ".join(
                    studio["name"]
                    for studio in data.get(
                        "studios",
                        []
                    )
                )

                demographics = " ".join(
                    demo["name"]
                    for demo in data.get(
                        "demographics",
                        []
                    )
                )

                return {
                    "anime_id": anime_id,
                    "synopsis": data.get(
                        "synopsis",
                        ""
                    ),
                    "year": data.get(
                        "year"
                    ),
                    "season": data.get(
                        "season"
                    ),
                    "score": data.get(
                        "score"
                    ),
                    "genres_jikan": genres,
                    "themes": themes,
                    "studios": studios,
                    "demographics": demographics
                }

            elif response.status_code in (
                429,
                500,
                502,
                503,
                504
            ):

                wait_time = BACKOFF_FACTOR ** attempt

                print(
                    f"[Retry {attempt}/{MAX_RETRIES}] "
                    f"Anime {anime_id} "
                    f"returned {response.status_code}. "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            else:

                print(
                    f"Skipping {anime_id}: "
                    f"HTTP {response.status_code}"
                )

                return None

        except requests.exceptions.RequestException as e:

            wait_time = BACKOFF_FACTOR ** attempt

            print(
                f"[Retry {attempt}/{MAX_RETRIES}] "
                f"Network error for {anime_id}: {e}"
            )

            time.sleep(wait_time)

    print(
        f"Giving up on anime_id {anime_id}"
    )

    return None


def save_batch(existing_df, batch):

    if not batch:
        return existing_df

    batch_df = pd.DataFrame(batch)

    combined_df = pd.concat(
        [existing_df, batch_df],
        ignore_index=True
    )

    combined_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Checkpoint saved: "
        f"{len(combined_df)} records"
    )

    return combined_df

def main():

    anime_df = pd.read_csv(
        INPUT_FILE
    )

    if os.path.exists(
        OUTPUT_FILE
    ):

        existing_df = pd.read_csv(
            OUTPUT_FILE
        )

        processed_ids = set(
            existing_df["anime_id"]
        )

        print(
            f"Found existing file."
        )

        print(
            f"Skipping "
            f"{len(processed_ids)} "
            f"already processed anime."
        )

    else:

        existing_df = pd.DataFrame()

        processed_ids = set()

    enriched_batch = []

    remaining_df = anime_df[
        ~anime_df["anime_id"].isin(
            processed_ids
        )
    ]

    print(
        f"Remaining anime: "
        f"{len(remaining_df)}"
    )

    for _, row in tqdm(
        remaining_df.iterrows(),
        total=len(remaining_df)
    ):

        anime_id = row["anime_id"]

        result = fetch_jikan_metadata(
            anime_id
        )

        if result:

            enriched_batch.append(
                result
            )

        if len(enriched_batch) >= SAVE_INTERVAL:

            existing_df = save_batch(
                existing_df,
                enriched_batch
            )

            enriched_batch = []

        time.sleep(
            REQUEST_DELAY
        )

    if enriched_batch:

        existing_df = save_batch(
            existing_df,
            enriched_batch
        )

    print("\nDone.")

    print(
        f"Total records saved: "
        f"{len(existing_df)}"
    )

if __name__ == "__main__":
    main()
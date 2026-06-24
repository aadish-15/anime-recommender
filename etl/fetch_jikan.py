import os
import time
import requests
import pandas as pd

from tqdm import tqdm

INPUT_FILE = "data/processed/anime_clean.csv"
OUTPUT_FILE = "data/processed/jikan_enriched.csv"

SAVE_INTERVAL = 25
REQUEST_DELAY = 1

def fetch_jikan_metadata(anime_id):

    url = f"https://api.jikan.moe/v4/anime/{anime_id}"

    try:

        response = requests.get(
            url,
            timeout=15
        )

        if response.status_code != 200:

            print(
                f"Failed for {anime_id}: "
                f"{response.status_code}"
            )

            return None
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
    except Exception as e:
        print(
            f"Error for anime_id "
            f"{anime_id}: {e}"
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
    # ----------------------------------
    # Resume support
    # ----------------------------------
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
        if (
            len(enriched_batch)
            >= SAVE_INTERVAL
        ):
            existing_df = save_batch(
                existing_df,
                enriched_batch
            )
            enriched_batch = []
        time.sleep(
            REQUEST_DELAY
        )
    # ----------------------------------
    # Final save
    # ----------------------------------
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
from pathlib import Path
import os
import time
from datetime import datetime, timezone
import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "anime_clean.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "jikan_enriched.csv"
)

FAILURE_FILE = (
    PROCESSED_DIR
    / "jikan_failures.csv"
)


# ---------------------------------------------------------
# Request configuration
# ---------------------------------------------------------

BASE_URL = "https://api.jikan.moe/v4/anime"

REQUEST_TIMEOUT = 20

# Conservative delay between requests.
REQUEST_DELAY = 1.25
MAX_RETRIES = 5
BACKOFF_FACTOR = 2

# Save progress after this many attempted anime,
# not merely successful requests.
SAVE_INTERVAL = 100


# HTTP responses that may succeed if retried later.
RETRYABLE_STATUS_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504
}

# Responses that normally indicate there is no point
# repeatedly requesting the same anime ID.
PERMANENT_STATUS_CODES = {
    400,
    404
}


# ---------------------------------------------------------
# Session
# ---------------------------------------------------------

session = requests.Session()

session.headers.update(
    {
        "User-Agent": (
            "Anime-Recommender-ETL/1.0"
        ),
        "Accept": "application/json"
    }
)


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def join_names(items):


    if not isinstance(items, list):
        return ""

    names = []

    for item in items:

        if not isinstance(item, dict):
            continue

        name = item.get("name")

        if name:
            names.append(
                str(name).strip()
            )

    return " ".join(
        names
    )


def atomic_write_csv(
    df,
    output_path
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    df.to_csv(
        temp_path,
        index=False
    )

    os.replace(
        temp_path,
        output_path
    )


# ---------------------------------------------------------
# Existing checkpoint loading
# ---------------------------------------------------------

def load_existing_successes():
    """
    Load previously enriched records.
    """

    if not OUTPUT_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        OUTPUT_FILE
    )

    if df.empty:
        return df

    if "anime_id" not in df.columns:
        raise ValueError(
            f"{OUTPUT_FILE.name} does not contain "
            "an anime_id column."
        )

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

    df = df.drop_duplicates(
        subset=["anime_id"],
        keep="last"
    )

    df = df.reset_index(
        drop=True
    )

    return df


def load_existing_failures():
    """
    Load previously recorded request failures.
    """

    if not FAILURE_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        FAILURE_FILE
    )

    if df.empty:
        return df

    if "anime_id" not in df.columns:
        raise ValueError(
            f"{FAILURE_FILE.name} does not contain "
            "an anime_id column."
        )

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

    df = df.drop_duplicates(
        subset=["anime_id"],
        keep="last"
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ---------------------------------------------------------
# Failure records
# ---------------------------------------------------------

def make_failure_record(
    anime_id,
    error,
    status_code=None,
    retryable=True
):
    """
    Create a consistent failure record.
    """

    return {
        "anime_id": int(anime_id),
        "status_code": status_code,
        "error": str(error),
        "retryable": bool(retryable),
        "last_attempt": utc_now()
    }


# ---------------------------------------------------------
# API response parsing
# ---------------------------------------------------------

def parse_jikan_response(
    anime_id,
    payload
):
    """
    Parse and validate a successful Jikan response.
    """

    if not isinstance(payload, dict):
        raise ValueError(
            "Response body is not a JSON object."
        )

    data = payload.get("data")

    if not isinstance(data, dict):
        raise ValueError(
            "Response does not contain a valid "
            "'data' object."
        )

    return {
        "anime_id": int(anime_id),

        "synopsis": clean_text(
            data.get("synopsis")
        ),

        "year": data.get(
            "year"
        ),

        "season": clean_text(
            data.get("season")
        ),

        "score": data.get(
            "score"
        ),

        "genres_jikan": join_names(
            data.get("genres")
        ),

        "themes": join_names(
            data.get("themes")
        ),

        "studios": join_names(
            data.get("studios")
        ),

        "demographics": join_names(
            data.get("demographics")
        )
    }


# ---------------------------------------------------------
# Request logic
# ---------------------------------------------------------

def get_retry_delay(
    response,
    attempt
):

    retry_after = response.headers.get(
        "Retry-After"
    )

    if retry_after:

        try:
            return max(
                float(retry_after),
                0
            )

        except ValueError:
            pass

    return BACKOFF_FACTOR ** attempt


def fetch_jikan_metadata(
    anime_id
):
    """
    Fetch one anime from Jikan.

    Returns
    -------
    tuple:
        (result, failure)

    Exactly one will normally contain a value.
    """

    url = (
        f"{BASE_URL}/{int(anime_id)}"
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT
            )

        except requests.exceptions.RequestException as exc:

            if attempt < MAX_RETRIES:

                wait_time = (
                    BACKOFF_FACTOR ** attempt
                )

                print(
                    f"\n[Retry {attempt}/{MAX_RETRIES}] "
                    f"Network error for {anime_id}: "
                    f"{exc}. "
                    f"Waiting {wait_time:.1f}s..."
                )

                time.sleep(
                    wait_time
                )

                continue

            return (
                None,
                make_failure_record(
                    anime_id=anime_id,
                    error=(
                        "Network request failed after "
                        f"{MAX_RETRIES} attempts: {exc}"
                    ),
                    retryable=True
                )
            )

        # -------------------------------------------------
        # Success
        # -------------------------------------------------

        if response.status_code == 200:

            try:

                payload = response.json()

                result = (
                    parse_jikan_response(
                        anime_id,
                        payload
                    )
                )

                return result, None

            except (
                ValueError,
                requests.exceptions.JSONDecodeError
            ) as exc:

                if attempt < MAX_RETRIES:

                    wait_time = (
                        BACKOFF_FACTOR ** attempt
                    )

                    print(
                        f"\n[Retry {attempt}/{MAX_RETRIES}] "
                        f"Invalid response for "
                        f"{anime_id}: {exc}. "
                        f"Waiting {wait_time:.1f}s..."
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

                return (
                    None,
                    make_failure_record(
                        anime_id=anime_id,
                        status_code=200,
                        error=(
                            "Invalid Jikan response "
                            f"after retries: {exc}"
                        ),
                        retryable=True
                    )
                )

        # -------------------------------------------------
        # Retryable HTTP response
        # -------------------------------------------------

        if (
            response.status_code
            in RETRYABLE_STATUS_CODES
        ):

            if attempt < MAX_RETRIES:

                wait_time = get_retry_delay(
                    response,
                    attempt
                )

                print(
                    f"\n[Retry {attempt}/{MAX_RETRIES}] "
                    f"Anime {anime_id} returned "
                    f"HTTP {response.status_code}. "
                    f"Waiting {wait_time:.1f}s..."
                )

                time.sleep(
                    wait_time
                )

                continue

            return (
                None,
                make_failure_record(
                    anime_id=anime_id,
                    status_code=(
                        response.status_code
                    ),
                    error=(
                        "Retryable HTTP error persisted "
                        f"after {MAX_RETRIES} attempts."
                    ),
                    retryable=True
                )
            )

        # -------------------------------------------------
        # Permanent HTTP response
        # -------------------------------------------------

        if (
            response.status_code
            in PERMANENT_STATUS_CODES
        ):

            return (
                None,
                make_failure_record(
                    anime_id=anime_id,
                    status_code=(
                        response.status_code
                    ),
                    error=(
                        f"Permanent HTTP "
                        f"{response.status_code}"
                    ),
                    retryable=False
                )
            )

        # -------------------------------------------------
        # Unexpected HTTP response
        # -------------------------------------------------

        return (
            None,
            make_failure_record(
                anime_id=anime_id,
                status_code=(
                    response.status_code
                ),
                error=(
                    f"Unexpected HTTP "
                    f"{response.status_code}"
                ),
                retryable=True
            )
        )

    return (
        None,
        make_failure_record(
            anime_id=anime_id,
            error="Unknown request failure.",
            retryable=True
        )
    )


# ---------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------

def merge_success_records(
    existing_df,
    new_records
):
    """
    Merge new successful records into the existing
    checkpoint while keeping one row per anime ID.
    """

    if not new_records:
        return existing_df

    batch_df = pd.DataFrame(
        new_records
    )

    combined_df = pd.concat(
        [
            existing_df,
            batch_df
        ],
        ignore_index=True
    )

    combined_df = (
        combined_df
        .drop_duplicates(
            subset=["anime_id"],
            keep="last"
        )
        .sort_values(
            "anime_id"
        )
        .reset_index(
            drop=True
        )
    )

    return combined_df


def merge_failure_records(
    existing_df,
    new_records
):
    """
    Merge failure records while keeping only the latest
    failure for each anime ID.
    """

    if not new_records:
        return existing_df

    batch_df = pd.DataFrame(
        new_records
    )

    combined_df = pd.concat(
        [
            existing_df,
            batch_df
        ],
        ignore_index=True
    )

    combined_df = (
        combined_df
        .drop_duplicates(
            subset=["anime_id"],
            keep="last"
        )
        .sort_values(
            "anime_id"
        )
        .reset_index(
            drop=True
        )
    )

    return combined_df


def remove_resolved_failures(
    failure_df,
    successful_ids
):

    if (
        failure_df.empty
        or not successful_ids
    ):
        return failure_df

    return (
        failure_df[
            ~failure_df[
                "anime_id"
            ].isin(
                successful_ids
            )
        ]
        .reset_index(
            drop=True
        )
    )


def save_checkpoint(
    success_df,
    failure_df
):

    atomic_write_csv(
        success_df,
        OUTPUT_FILE
    )

    if not failure_df.empty:

        atomic_write_csv(
            failure_df,
            FAILURE_FILE
        )

    elif FAILURE_FILE.exists():

        FAILURE_FILE.unlink()

    print(
        "\nCheckpoint saved:"
        f"\n  Successful: {len(success_df):,}"
        f"\n  Failures:   {len(failure_df):,}"
    )


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Clean anime dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run:\n"
            "python etl/transform.py"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    anime_df = pd.read_csv(
        INPUT_FILE
    )

    if "anime_id" not in anime_df.columns:

        raise ValueError(
            f"{INPUT_FILE.name} does not contain "
            "an anime_id column."
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
        subset=["anime_id"]
    )

    existing_successes = (
        load_existing_successes()
    )

    existing_failures = (
        load_existing_failures()
    )

    successful_ids = set(
        existing_successes[
            "anime_id"
        ].tolist()
    ) if not existing_successes.empty else set()

    # Permanent failures are skipped on future runs.
    # Retryable failures are intentionally attempted again.
    permanent_failure_ids = set()

    if not existing_failures.empty:

        if "retryable" in existing_failures.columns:

            retryable_values = (
                existing_failures[
                    "retryable"
                ]
                .astype(str)
                .str.casefold()
            )

            permanent_mask = (
                retryable_values
                .isin(
                    {
                        "false",
                        "0"
                    }
                )
            )

            permanent_failure_ids = set(
                existing_failures.loc[
                    permanent_mask,
                    "anime_id"
                ].tolist()
            )

    processed_ids = (
        successful_ids
        | permanent_failure_ids
    )

    remaining_df = anime_df[
        ~anime_df[
            "anime_id"
        ].isin(
            processed_ids
        )
    ].copy()

    print(
        f"Total anime: "
        f"{len(anime_df):,}"
    )

    print(
        f"Already enriched: "
        f"{len(successful_ids):,}"
    )

    print(
        f"Permanent failures skipped: "
        f"{len(permanent_failure_ids):,}"
    )

    print(
        f"Remaining: "
        f"{len(remaining_df):,}"
    )

    if remaining_df.empty:

        print(
            "\nNothing to fetch."
        )

        return

    success_batch = []
    failure_batch = []

    attempted_since_save = 0

    try:

        for anime_id in tqdm(
            remaining_df[
                "anime_id"
            ].tolist(),
            desc="Jikan enrichment"
        ):

            result, failure = (
                fetch_jikan_metadata(
                    anime_id
                )
            )

            if result is not None:

                success_batch.append(
                    result
                )

            if failure is not None:

                failure_batch.append(
                    failure
                )

            attempted_since_save += 1

            if (
                attempted_since_save
                >= SAVE_INTERVAL
            ):

                new_success_ids = {
                    record["anime_id"]
                    for record
                    in success_batch
                }

                existing_successes = (
                    merge_success_records(
                        existing_successes,
                        success_batch
                    )
                )

                existing_failures = (
                    merge_failure_records(
                        existing_failures,
                        failure_batch
                    )
                )

                existing_failures = (
                    remove_resolved_failures(
                        existing_failures,
                        new_success_ids
                    )
                )

                save_checkpoint(
                    existing_successes,
                    existing_failures
                )

                success_batch = []
                failure_batch = []

                attempted_since_save = 0

            time.sleep(
                REQUEST_DELAY
            )

    except KeyboardInterrupt:

        print(
            "\n\nInterrupted by user."
        )

    finally:

        # Save anything still held in memory, including
        # progress made before Ctrl+C or an exception.

        new_success_ids = {
            record["anime_id"]
            for record
            in success_batch
        }

        existing_successes = (
            merge_success_records(
                existing_successes,
                success_batch
            )
        )

        existing_failures = (
            merge_failure_records(
                existing_failures,
                failure_batch
            )
        )

        existing_failures = (
            remove_resolved_failures(
                existing_failures,
                new_success_ids
            )
        )

        save_checkpoint(
            existing_successes,
            existing_failures
        )

        print(
            "\nJikan enrichment finished."
        )

        print(
            f"Successful records: "
            f"{len(existing_successes):,}"
        )

        print(
            f"Failure records: "
            f"{len(existing_failures):,}"
        )


if __name__ == "__main__":
    main()
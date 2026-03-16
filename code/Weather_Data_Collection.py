"""
Weather Data Collection Script
================================
Collects 20 years of daily weather data (2005-2024) for 166 coordinate points
across New York State from the Open-Meteo Archive API.

Outputs:
    ../data/new-york-weather.csv      — Appended daily weather records
    ../data/failed_coordinates.csv    — Coordinates that failed after all retries

Run this script once before executing Analysis.ipynb. The output CSV is large;
collection takes several hours due to API rate limiting.
"""

import os
import time
import random

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 166 representative coordinate points (latitude, longitude) covering all
# 62 counties of New York State. Three points per county where possible:
# county centroid plus two secondary points for spatial resolution.
COORDINATES: list[tuple[float, float]] = [
    (42.6, -73.97), (42.5273, -73.908), (42.6632, -73.8903),
    (42.25, -78.02), (42.3408, -77.9462), (42.1825, -77.966),
    (40.85, -73.8667),
    (42.17, -75.82), (42.0935, -75.8327), (42.2335, -75.8336),
    (42.25, -78.68), (42.1957, -78.6287), (42.3213, -78.6293),
    (42.95, -76.56), (43.0382, -76.524), (43.0324, -76.6587),
    (42.25, -79.37), (42.3169, -79.3912), (42.2706, -79.4473),
    (42.15, -76.75), (42.1932, -76.8433), (42.2487, -76.6562),
    (42.5, -75.6), (42.509, -75.5603), (42.5852, -75.5167),
    (44.75, -73.7), (44.7984, -73.7165), (44.819, -73.7095),
    (42.25, -73.63), (42.1683, -73.687), (42.1781, -73.5694),
    (42.6, -76.0), (42.6139, -76.086), (42.5666, -76.0895),
    (42.2, -74.95), (42.213, -74.982), (42.1821, -74.9847),
    (41.75, -73.74), (41.8488, -73.757), (41.7248, -73.6964),
    (42.77, -78.64), (42.8206, -78.673), (42.839, -78.7131),
    (44.1, -73.77), (44.0899, -73.8463), (44.0396, -73.6853),
    (44.59, -74.34), (44.4999, -74.4129), (44.6333, -74.3266),
    (43.1, -74.43), (43.1869, -74.5102), (43.038, -74.3446),
    (43.0, -78.19), (42.9548, -78.0943), (42.932, -78.1732),
    (42.3, -74.13), (42.2518, -74.1249), (42.3921, -74.2229),
    (43.67, -74.5), (43.6537, -74.4334), (43.7283, -74.5597),
    (43.42, -74.98), (43.5023, -75.0579), (43.4821, -74.8937),
    (44.0, -75.92), (44.0255, -76.0046), (44.0407, -75.9948),
    (40.65, -73.95),
    (43.79, -75.46), (43.8139, -75.5525), (43.848, -75.4359),
    (42.73, -77.78), (42.6432, -77.706), (42.7847, -77.7621),
    (42.9, -75.7), (42.9976, -75.7697), (42.8931, -75.6475),
    (43.15, -77.62), (43.1521, -77.5636), (43.1053, -77.6019),
    (42.9, -74.42), (42.9634, -74.4109), (42.8675, -74.47),
    (40.74, -73.64),
    (40.7831, -73.9712),
    (43.31, -78.97), (43.3969, -78.972), (43.3256, -78.9341),
    (43.24, -75.43), (43.3357, -75.3502), (43.1585, -75.5288),
    (43.0, -76.2), (42.9936, -76.1988), (43.014, -76.1712),
    (42.85, -77.28), (42.9361, -77.2098), (42.815, -77.1844),
    (41.4, -74.3), (41.4114, -74.3941), (41.4477, -74.2001),
    (43.25, -78.23), (43.3, -78.2743), (43.2886, -78.1854),
    (43.46, -76.27), (43.487, -76.2653), (43.5173, -76.2727),
    (42.66, -75.0), (42.5666, -75.0284), (42.7171, -75.0154),
    (41.43, -73.74),
    (40.742, -73.7694),
    (42.71, -73.51), (42.6241, -73.4122), (42.6497, -73.5621),
    (40.5795, -74.1502),
    (41.15, -73.95),
    (44.45, -75.0), (44.3836, -75.0768), (44.425, -74.9352),
    (43.1, -73.86), (43.0371, -73.9361), (43.1261, -73.8703),
    (42.81, -73.95),
    (42.66, -74.31), (42.6428, -74.4031), (42.6698, -74.3269),
    (42.38, -76.9), (42.4407, -76.915), (42.3183, -76.8097),
    (42.83, -76.83), (42.9109, -76.7693), (42.7416, -76.8044),
    (42.27, -77.38), (42.3368, -77.3559), (42.2979, -77.4301),
    (40.96, -72.69), (40.8938, -72.6109), (40.8888, -72.6118),
    (41.7, -74.77), (41.6114, -74.8241), (41.616, -74.7274),
    (42.17, -76.35), (42.1724, -76.3101), (42.2179, -76.298),
    (42.45, -76.47), (42.5157, -76.4528), (42.5222, -76.4449),
    (41.93, -74.27), (42.0144, -74.1736), (42.0277, -74.2152),
    (43.58, -73.77), (43.6572, -73.8086), (43.5552, -73.6773),
    (43.35, -73.42), (43.2725, -73.4727), (43.3773, -73.4402),
    (43.07, -77.0), (43.1199, -77.0886), (43.0837, -77.0676),
    (41.15, -73.75), (41.0824, -73.7235), (41.1497, -73.8129),
    (42.73, -78.21), (42.6543, -78.2517), (42.7225, -78.1886),
    (42.65, -77.1),
]

WEATHER_FEATURES: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "daylight_duration",
    "sunshine_duration",
    "uv_index_max",
    "uv_index_clear_sky_max",
    "rain_sum",
    "showers_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

DATA_START_DATE = "2005-01-01"
DATA_END_DATE = "2024-12-31"
TIMEZONE = "America/New_York"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "new-york-weather.csv")
FAILED_COORDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "failed_coordinates.csv")
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Rate limiting thresholds. Open-Meteo's free tier allows 5,000 requests/hour
# and 10,000 requests/day. We pause every 3 requests to stay well under the
# per-minute burst limit.
REQUESTS_PER_PAUSE = 3
PAUSE_MIN_SECONDS = 2.5
PAUSE_MAX_SECONDS = 4.5
MINUTELY_LIMIT_PAUSE = 65    # Slightly over 60s to clear the window
HOURLY_LIMIT_PAUSE = 3600
HOURLY_REQUEST_LIMIT = 5000
DAILY_REQUEST_LIMIT = 10000
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def _build_openmeteo_client() -> openmeteo_requests.Client:
    """
    Build an Open-Meteo API client with caching and retry logic.

    Uses requests_cache to avoid redundant network calls during development,
    and retry_requests to handle transient network errors automatically.

    Returns:
        Configured openmeteo_requests.Client instance.
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# Build the client once at module level so all calls share the cache.
openmeteo = _build_openmeteo_client()


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_weather_data(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch daily weather data for a single coordinate from the Open-Meteo Archive API.

    Requests twelve daily weather variables for the full collection window
    (DATA_START_DATE to DATA_END_DATE). Results are returned as a flat DataFrame
    with one row per day plus latitude/longitude columns for later spatial joins.

    Args:
        lat: Latitude of the target location.
        lon: Longitude of the target location.

    Returns:
        DataFrame with columns for date, all WEATHER_FEATURES, latitude, longitude.

    Raises:
        Exception: Propagates any API or network error to the caller, which handles
            retries and rate-limit pauses.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": DATA_START_DATE,
        "end_date": DATA_END_DATE,
        "daily": WEATHER_FEATURES,
        "timezone": TIMEZONE,
    }

    response = openmeteo.weather_api(API_URL, params=params)[0]
    daily = response.Daily()

    n_values = len(daily.Variables(0).ValuesAsNumpy())
    daily_data: dict = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ),
        "latitude": [lat] * n_values,
        "longitude": [lon] * n_values,
    }

    for idx, feature in enumerate(WEATHER_FEATURES):
        daily_data[feature] = daily.Variables(idx).ValuesAsNumpy()

    return pd.DataFrame(data=daily_data)


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect_weather_data(
    coordinates: list[tuple[float, float]],
    max_retries: int = MAX_RETRIES,
    daily_api_limit: int = DAILY_REQUEST_LIMIT,
) -> tuple[pd.DataFrame, list[tuple[float, float]]]:
    """
    Collect weather data for all coordinates with rate limiting and retry logic.

    Iterates over the coordinate list, fetching data for each point. Pauses
    every REQUESTS_PER_PAUSE calls to stay within Open-Meteo's burst limits.
    Detects API error messages in exceptions to apply the correct pause duration
    (minutely vs. hourly limit). Stops early if daily_api_limit is reached.

    Args:
        coordinates: List of (latitude, longitude) tuples to collect.
        max_retries: Number of retry attempts per coordinate before marking as failed.
        daily_api_limit: Maximum total requests before stopping collection.

    Returns:
        Tuple of (combined_dataframe, failed_coordinates) where:
            - combined_dataframe: All successfully collected records concatenated.
            - failed_coordinates: Coordinates that failed after all retry attempts.
    """
    dataframes: list[pd.DataFrame] = []
    failed_coordinates: list[tuple[float, float]] = []
    request_count = 0
    hourly_request_count = 0
    daily_request_count = 0

    for i, (lat, lon) in enumerate(coordinates, 1):
        if daily_request_count >= daily_api_limit:
            print(
                f"Daily API limit of {daily_api_limit} requests reached. "
                "Stopping collection early."
            )
            break

        retries = 0
        success = False

        while retries < max_retries and not success:
            try:
                print(
                    f"Fetching ({lat}, {lon}) — location {i}/{len(coordinates)}, "
                    f"attempt {retries + 1}"
                )
                df = fetch_weather_data(lat, lon)
                dataframes.append(df)

                request_count += 1
                hourly_request_count += 1
                daily_request_count += 1
                success = True

                # Pause every REQUESTS_PER_PAUSE calls to avoid burst limits.
                if request_count % REQUESTS_PER_PAUSE == 0:
                    delay = random.uniform(PAUSE_MIN_SECONDS, PAUSE_MAX_SECONDS)
                    print(f"Pausing {delay:.2f}s to manage request rate...")
                    time.sleep(delay)

                # Pause for an hour if hourly limit is approached.
                if hourly_request_count >= HOURLY_REQUEST_LIMIT:
                    print("Hourly limit reached. Pausing 3600s...")
                    time.sleep(HOURLY_LIMIT_PAUSE)
                    hourly_request_count = 0

            except Exception as e:
                error_message = str(e)
                retries += 1

                if "Minutely API request limit exceeded" in error_message:
                    print("Minutely rate limit hit. Pausing 65s...")
                    time.sleep(MINUTELY_LIMIT_PAUSE)
                elif "Hourly API request limit exceeded" in error_message:
                    print("Hourly rate limit hit. Pausing 3600s...")
                    time.sleep(HOURLY_LIMIT_PAUSE)
                    hourly_request_count = 0
                else:
                    print(
                        f"Error: {error_message}. "
                        f"Retrying in 10s (attempt {retries}/{max_retries})..."
                    )
                    time.sleep(10)

        if not success:
            print(f"Failed after {max_retries} attempts: ({lat}, {lon})")
            failed_coordinates.append((lat, lon))

    if failed_coordinates:
        print(f"\n{len(failed_coordinates)} coordinates failed:")
        for coord in failed_coordinates:
            print(f"  {coord}")

    print(f"Total requests made: {daily_request_count}")

    if not dataframes:
        return pd.DataFrame(), failed_coordinates

    return pd.concat(dataframes, ignore_index=True), failed_coordinates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        combined_df, failed_coords = collect_weather_data(COORDINATES)

        if not combined_df.empty:
            combined_df.to_csv(OUTPUT_PATH, mode="a", index=False)
            print(f"Weather data saved to {OUTPUT_PATH}")
        else:
            print("No data collected — output file not written.")

        if failed_coords:
            failed_df = pd.DataFrame(failed_coords, columns=["Latitude", "Longitude"])
            failed_df.to_csv(FAILED_COORDS_PATH, index=False)
            print(f"Failed coordinates saved to {FAILED_COORDS_PATH}")

    except Exception as e:
        print(f"Fatal error during data collection: {e}")
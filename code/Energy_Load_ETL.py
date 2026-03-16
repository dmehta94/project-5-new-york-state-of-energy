"""
Energy Load ETL Pipeline
=========================
Extracts, transforms, and loads NYISO (New York Independent System Operator)
energy load data from downloaded zip archives into a single consolidated CSV.

The raw NYISO data ships as yearly folders of weekly zip files, each containing
a CSV with hourly load readings by zone. This script:
    1. Unzips all archives into a flat directory.
    2. Reads and aggregates every CSV to a daily average load per zone.
    3. Writes the combined result to data/Newyork_state_load_data.csv.

NYISO data source: https://www.nyiso.com/load-data

Run this script before Analysis.ipynb. The zip archives must be pre-downloaded
and placed in data/Load Data/<year>/ directories.
"""

import os
import zipfile
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
LOAD_DATA_DIR = os.path.join(BASE_DIR, "data", "Load Data")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "Newyork_state_load_data.csv")

# The NYISO data available in the downloaded archives spans 2008-2024.
DATA_START_YEAR = 2008
DATA_END_YEAR = 2025  # exclusive upper bound for range()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def unzip_folder(zip_path: str, extract_to: str) -> None:
    """
    Extract a zip archive to the specified directory.

    Args:
        zip_path: Absolute path to the .zip file.
        extract_to: Directory into which contents are extracted.

    Raises:
        zipfile.BadZipFile: If the file at zip_path is not a valid zip archive.
        FileNotFoundError: If zip_path does not exist.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted: {os.path.basename(zip_path)} → {extract_to}")


def unzip_all_archives(base_path: str, start_year: int, end_year: int) -> None:
    """
    Iterate over yearly subdirectories and extract all zip archives found.

    Expects the directory structure: base_path/<year>/*.zip

    Args:
        base_path: Root directory containing year-named subdirectories.
        start_year: First year to process (inclusive).
        end_year: Last year to process (exclusive).
    """
    for year in range(start_year, end_year):
        year_dir = os.path.join(base_path, str(year))

        if not os.path.exists(year_dir):
            print(f"Directory not found, skipping: {year_dir}")
            continue

        zip_files = [f for f in os.listdir(year_dir) if f.endswith(".zip")]

        if not zip_files:
            print(f"No zip files found in {year_dir}")
            continue

        for zip_file in zip_files:
            zip_path = os.path.join(year_dir, zip_file)
            try:
                unzip_folder(zip_path, base_path)
            except (zipfile.BadZipFile, FileNotFoundError) as e:
                print(f"Could not extract {zip_file}: {e}")


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------

def compile_data(file_name: str, base_path: str) -> Optional[pd.DataFrame]:
    """
    Read a single NYISO load CSV and aggregate to daily average load per zone.

    Each raw CSV contains hourly load readings with columns:
        Time Stamp | Time Zone | Name | PTID | Load

    This function groups by Name and date, averages the hourly Load readings,
    and returns one row per (Name, Date) pair.

    Args:
        file_name: Name of the CSV file (not the full path).
        base_path: Directory containing the CSV file.

    Returns:
        DataFrame with columns [Name, Load, Date], or None if reading fails.
    """
    file_path = os.path.join(base_path, file_name)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Could not read {file_name}: {e}")
        return None

    # Extract date from the timestamp and drop columns not needed downstream.
    df["Date"] = pd.to_datetime(df["Time Stamp"]).dt.date
    df = df.drop(columns=["Time Stamp", "Time Zone", "PTID"])

    # Average load per zone per day.
    daily_avg = df.groupby("Name")["Load"].mean().reset_index()
    # All rows in a single file share the same date after extraction.
    daily_avg["Date"] = df["Date"].iloc[0]

    return daily_avg


def compile_all_data(base_path: str) -> pd.DataFrame:
    """
    Compile and concatenate daily average load data from all CSV files.

    Scans base_path for CSV files, calls compile_data on each, and
    concatenates the results into a single DataFrame.

    Args:
        base_path: Directory containing extracted NYISO CSV files.

    Returns:
        DataFrame with columns [Name, Load, Date] covering all files.

    Raises:
        ValueError: If no valid CSV files are found in base_path.
    """
    csv_files = [f for f in os.listdir(base_path) if f.endswith(".csv")]

    if not csv_files:
        raise ValueError(f"No CSV files found in {base_path}")

    dataframes = []
    for file_name in csv_files:
        result = compile_data(file_name, base_path)
        if result is not None:
            dataframes.append(result)

    if not dataframes:
        raise ValueError("No data successfully compiled — check CSV file formats.")

    return pd.concat(dataframes, ignore_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Step 1: Extracting zip archives...")
    unzip_all_archives(LOAD_DATA_DIR, DATA_START_YEAR, DATA_END_YEAR)

    print("\nStep 2: Compiling load data from extracted CSVs...")
    try:
        load_df = compile_all_data(LOAD_DATA_DIR)
        load_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nLoad data saved to {OUTPUT_PATH} ({len(load_df):,} rows)")
    except ValueError as e:
        print(f"ETL failed: {e}")
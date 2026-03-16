"""
County Forecasting ETL Pipeline
=================================
Trains a Prophet time-series model for each New York State county's projected
energy load and writes forecasted values through 2050 to CSV, then extracts
a filtered subset covering 2005–2030 for use in the analysis.

This script picks up where Analysis.ipynb's load data section leaves off:
    Input:  data/Counties load data.csv   — pivoted county load DataFrame
                                            (Date index, one column per county)
    Output: data/Counties_Forecast.csv    — wide-format forecasts, one column per county
            data/load_data_from_<start>_to_<end>.csv — date-filtered extract

Run this script after Energy_Load_ETL.py and after the pivoting steps in
Analysis.ipynb have produced the Counties load data.csv file.
"""

import os

import pandas as pd
from prophet import Prophet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_FILE = "Counties load data.csv"
FORECAST_OUTPUT_FILE = "Counties_Forecast.csv"

# Date range for the filtered extract written by extract_date_range().
# Covers the full historical + near-term forecast window used in the analysis.
EXTRACT_START_DATE = "2005-01-01"
EXTRACT_END_DATE = "2030-12-31"

# Prophet forecasts through this year. Each county model projects daily load
# from the last observed date forward to FORECAST_THROUGH_YEAR-12-31,
# then extract_date_range() filters down to EXTRACT_END_DATE for analysis.
FORECAST_THROUGH_YEAR = 2050


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def prepare_data_for_prophet(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Rename columns to Prophet's required 'ds' and 'y' format.

    Args:
        df: Source DataFrame containing at least date_col and value_col.
        date_col: Name of the column containing dates.
        value_col: Name of the column containing the target variable.

    Returns:
        DataFrame with columns ['ds', 'y'] ready for Prophet.fit().
    """
    return df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})


def forecast_county(
    data: pd.DataFrame,
    county: str,
    forecast_through_year: int = FORECAST_THROUGH_YEAR,
) -> pd.DataFrame:
    """
    Train a Prophet model on one county's load history and forecast forward.

    Uses default Prophet settings (additive trend, yearly seasonality). The
    future DataFrame extends from the last observed date to December 31 of
    forecast_through_year, producing daily predictions.

    Args:
        data: Wide-format DataFrame with a 'Date' column and one column per county.
        county: Name of the county column to forecast.
        forecast_through_year: Final year of the forecast horizon.

    Returns:
        DataFrame with columns ['ds', 'yhat'] covering historical fit + forecast.
    """
    county_data = prepare_data_for_prophet(data, "Date", county)

    model = Prophet()
    model.fit(county_data)

    last_date = county_data["ds"].max()
    future_periods = (forecast_through_year - last_date.year) * 365
    future_df = model.make_future_dataframe(periods=future_periods)

    forecast = model.predict(future_df)
    return forecast[["ds", "yhat"]]


def run_county_forecasts(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run Prophet forecasts for every county in the dataset and merge results.

    Iterates over all county columns (everything except 'Date'), trains a
    Prophet model for each, and merges the 'yhat' series into a wide DataFrame
    keyed on 'ds' (date).

    Args:
        data: Wide-format load DataFrame with 'Date' column and county columns.

    Returns:
        Wide DataFrame with 'Date' column and one forecast column per county.
    """
    counties = [col for col in data.columns if col != "Date"]
    all_forecasts = pd.DataFrame()

    for county in counties:
        print(f"Forecasting: {county}...")
        forecast = forecast_county(data, county)
        forecast = forecast.rename(columns={"yhat": county})

        if all_forecasts.empty:
            all_forecasts = forecast
        else:
            all_forecasts = pd.merge(all_forecasts, forecast, on="ds", how="outer")

    all_forecasts = all_forecasts.rename(columns={"ds": "Date"})
    return all_forecasts


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_date_range(
    file_name: str,
    start_date: str,
    end_date: str,
    data_dir: str = DATA_DIR,
) -> None:
    """
    Filter a date-indexed CSV to a specified range and write the result.

    Reads the file, parses the 'Date' column, keeps rows within [start_date,
    end_date], and saves the result as a new CSV named by the date range.

    Args:
        file_name: Name of the CSV to filter (relative to data_dir).
        start_date: Inclusive start date in 'YYYY-MM-DD' format.
        end_date: Inclusive end date in 'YYYY-MM-DD' format.
        data_dir: Directory containing file_name and where output is saved.
    """
    input_path = os.path.join(data_dir, file_name)
    output_name = f"load_data_from_{start_date}_to_{end_date}.csv"
    output_path = os.path.join(data_dir, output_name)

    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    df.to_csv(output_path, index=False)
    print(f"Date-range extract saved to {output_path} ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_path = os.path.join(DATA_DIR, INPUT_FILE)

    print(f"Loading input data from {input_path}...")
    try:
        data = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run Energy_Load_ETL.py and the Analysis.ipynb pivoting steps first."
        )

    # Ensure Date is parsed before passing to Prophet.
    data["Date"] = pd.to_datetime(data["Date"])

    print("\nRunning county-level Prophet forecasts...")
    all_forecasts = run_county_forecasts(data)

    forecast_output_path = os.path.join(DATA_DIR, FORECAST_OUTPUT_FILE)
    all_forecasts.to_csv(forecast_output_path, index=False)
    print(f"\nCombined forecast saved to {forecast_output_path}")

    print("\nExtracting date-range subset...")
    extract_date_range(
        FORECAST_OUTPUT_FILE,
        EXTRACT_START_DATE,
        EXTRACT_END_DATE,
    )
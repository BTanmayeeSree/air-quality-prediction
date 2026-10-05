"""
Preprocessing utilities for the Air Quality Prediction project.

Data source: Delhi daily weather + PM2.5 observations, 2013-2016
(4 CSVs, one per year). Columns: T, TM, Tm, SLP, H, VV, V, VM, PM 2.5.

The raw data has NO AQI column -- only PM2.5 concentration (ug/m3).
We derive AQI from PM2.5 using India's CPCB (Central Pollution Control
Board) National AQI sub-index breakpoints for PM2.5 (24-hr average).
This is a single-pollutant AQI (PM2.5 sub-index), since no other
pollutants (PM10, NO2, SO2, CO, O3) are present in this dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

FEATURE_COLUMNS = ["T", "TM", "Tm", "SLP", "H", "VV", "V", "VM"]
TARGET_COLUMN = "AQI"
PM25_COLUMN = "PM 2.5"

# CPCB PM2.5 (24-hour average, ug/m3) concentration/sub-index breakpoints:
# (C_low, C_high, I_low, I_high). Concentrations above the final breakpoint
# are capped at the CPCB AQI scale maximum rather than extrapolated.
CPCB_PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (31, 60, 51, 100),
    (61, 90, 101, 200),
    (91, 120, 201, 300),
    (121, 250, 301, 400),
    (251, 500, 401, 500),
]

CPCB_CATEGORIES = [
    (0, 50, "Good"),
    (50, 100, "Satisfactory"),
    (100, 200, "Moderate"),
    (200, 300, "Poor"),
    (300, 400, "Very Poor"),
    (400, 500, "Severe"),
]


def pm25_to_aqi(pm25: float) -> float:
    """Convert a PM2.5 concentration (ug/m3) to CPCB AQI sub-index using
    linear interpolation between CPCB's integer breakpoints."""
    if pd.isna(pm25):
        return np.nan
    pm25 = max(float(pm25), 0.0)
    if pm25 > CPCB_PM25_BREAKPOINTS[-1][1]:
        return 500.0
    pm25 = float(np.floor(pm25 + 0.5))
    for c_low, c_high, i_low, i_high in CPCB_PM25_BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            value = i_low + (i_high - i_low) / (c_high - c_low) * (pm25 - c_low)
            return float(np.clip(value, 0, 500))
    return np.nan


def aqi_to_category(aqi: float) -> str:
    """Map a CPCB AQI value to its official category label."""
    if pd.isna(aqi):
        return "Unknown"
    aqi = float(np.clip(aqi, 0, 500))
    for index, (lo, hi, label) in enumerate(CPCB_CATEGORIES):
        if (lo <= aqi if index == 0 else lo < aqi) and aqi <= hi:
            return label
    return "Severe"


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate the four yearly CSVs, tagging each row with its year."""
    frames = []
    for year in [2013, 2014, 2015, 2016]:
        path = RAW_DATA_DIR / f"real_{year}.csv"
        df = pd.read_csv(path)
        df["year"] = year
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize values, retain source years, and remove unusable targets.

    Exact-zero PM2.5 values are treated as unverified missing placeholders:
    the source has no sensor metadata and exactly zero is implausible for
    ambient Delhi measurements. Raw CSVs remain unchanged.
    """
    df = df.copy()
    expected_columns = FEATURE_COLUMNS + [PM25_COLUMN]
    missing_columns = sorted(set(expected_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required dataset columns: {missing_columns}")

    for column in expected_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[expected_columns] = df[expected_columns].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    duplicate_columns = expected_columns
    df = df.drop_duplicates(subset=duplicate_columns)
    dupes_removed = before - len(df)

    zero_pm25 = int((df[PM25_COLUMN] == 0).sum())
    df.loc[df[PM25_COLUMN] == 0, PM25_COLUMN] = np.nan
    df.loc[df[PM25_COLUMN] < 0, PM25_COLUMN] = np.nan

    # Out-of-range weather values become missing and are imputed in the
    # training pipeline; high, non-negative PM2.5 observations are retained.
    feature_ranges = {
        "T": (-50, 60), "TM": (-50, 70), "Tm": (-90, 50),
        "SLP": (800, 1100), "H": (0, 100), "VV": (0, 100),
        "V": (0, 200), "VM": (0, 200),
    }
    for column, (lower, upper) in feature_ranges.items():
        df.loc[~df[column].between(lower, upper), column] = np.nan
    df.loc[df["TM"] < df["T"], "TM"] = np.nan
    df.loc[df["Tm"] > df["T"], "Tm"] = np.nan
    df.loc[df["VM"] < df["V"], "VM"] = np.nan

    # PM2.5 is the target source, so missing/invalid targets cannot be imputed.
    before = len(df)
    df = df.dropna(subset=[PM25_COLUMN])
    missing_target_removed = before - len(df)

    print(f"[clean_data] duplicates removed: {dupes_removed}")
    print(f"[clean_data] exact-zero PM2.5 treated as missing: {zero_pm25}")
    print(f"[clean_data] rows dropped for missing/invalid PM2.5: {missing_target_removed}")
    print(f"[clean_data] final row count: {len(df)}")

    return df.reset_index(drop=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the AQI target column from PM2.5 using the CPCB sub-index."""
    df = df.copy()
    df[TARGET_COLUMN] = df[PM25_COLUMN].apply(pm25_to_aqi).clip(0, 500)
    df["AQI_Category"] = df[TARGET_COLUMN].apply(aqi_to_category)
    return df


def build_dataset() -> pd.DataFrame:
    """Full pipeline: load -> clean -> derive target. Returns the
    processed dataframe and also writes it to data/processed/."""
    df = load_raw_data()
    df = clean_data(df)
    df = add_target(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "processed_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"[build_dataset] wrote {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    build_dataset()

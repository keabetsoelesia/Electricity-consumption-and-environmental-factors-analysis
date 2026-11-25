"""
prepare_data.py
Reproducible data-preparation steps for the Week 5 AnalystLab Africa assignment
(Morocco Electricity Consumption and Environmental Factors Analysis, Track 1).

Usage (from the project root, with the virtual environment active):
    python prepare_data.py

Input : data/powerconsumption.csv
Output: data/powerconsumption_clean.csv
        reports/data_quality_log.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

RAW_PATH = Path("data/powerconsumption.csv")
CLEAN_PATH = Path("data/powerconsumption_clean.csv")
LOG_PATH = Path("reports/data_quality_log.txt")

ZONES = [
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3",
]
EXPECTED_INTERVAL = pd.Timedelta(minutes=10)

log_lines = []


def log(message=""):
    """Print a line and keep it for the data-quality log file."""
    print(message)
    log_lines.append(message)


# ---------------------------------------------------------------- Step 1: load
log("STEP 1: Load data")
df = pd.read_csv(RAW_PATH)
log(f"Shape: {df.shape}")
log(f"Columns: {list(df.columns)}")

# --------------------------------------------- Step 2: data quality inspection
log("\nSTEP 2: Inspect data quality (before any changes)")
log(f"Missing values per column:\n{df.isna().sum().to_string()}")
log(f"Fully duplicated rows: {df.duplicated().sum()}")

# ----------------------------------------------- Step 3: convert Datetime
log("\nSTEP 3: Convert Datetime")
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
log(f"Unparseable Datetime values: {df['Datetime'].isna().sum()}")
df = df.sort_values("Datetime").reset_index(drop=True)
log(f"Range: {df['Datetime'].min()} to {df['Datetime'].max()}")
log(f"Distinct days: {df['Datetime'].dt.date.nunique()}")
log(f"Duplicate timestamps: {df['Datetime'].duplicated().sum()}")

# ------------------------------------------- Step 4: interval consistency
log("\nSTEP 4: Check 10-minute interval consistency")
gaps = df["Datetime"].diff().dropna()
n_off = int((gaps != EXPECTED_INTERVAL).sum())
log(f"Intervals that are not exactly 10 minutes: {n_off}")

# ---------------------------------------------------- Step 5: sanity checks
log("\nSTEP 5: Value sanity checks")
log(f"Humidity range: {df['Humidity'].min()} to {df['Humidity'].max()} (expected 0-100)")
log(f"Negative consumption values: {(df[ZONES] < 0).sum().sum()}")
log("Diffuse flow columns: dictionary description is unclear; excluded from conclusions.")

# -------------------------------------------- Step 6: time-based columns
log("\nSTEP 6: Create time-based columns")
df["Year"] = df["Datetime"].dt.year
df["Month"] = df["Datetime"].dt.month
df["Month_Name"] = df["Datetime"].dt.month_name()
df["Day"] = df["Datetime"].dt.day
df["Hour"] = df["Datetime"].dt.hour
df["DayOfWeek"] = df["Datetime"].dt.day_name()
df["DayOfWeek_Num"] = df["Datetime"].dt.dayofweek  # Monday = 0
df["Date"] = df["Datetime"].dt.date
df["IsWeekend"] = df["DayOfWeek_Num"] >= 5

# ------------------------------------------ Step 7: flag (do not delete) outliers
log("\nSTEP 7: Flag extreme values with the IQR rule (rows are kept)")
for zone in ZONES:
    q1, q3 = df[zone].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    lower = q1 - 1.5 * (q3 - q1)
    flag = f"{zone.replace('PowerConsumption_', '')}_OutlierFlag"
    df[flag] = (df[zone] > upper) | (df[zone] < lower)
    log(f"{zone}: bounds {lower:,.0f} to {upper:,.0f} | flagged: {int(df[flag].sum())}")

# Zone 3 high readings are a summer-evening pattern (reviewed manually), so they are kept.

# ---------------------------------------- Step 8: wind speed level helper
log("\nSTEP 8: Add WindSpeed_Level helper (WindSpeed is recorded in two clusters)")
df["WindSpeed_Level"] = np.where(df["WindSpeed"] > 2, "High (~4.9)", "Low (~0.1)")
log(df["WindSpeed_Level"].value_counts().to_string())

# ----------------------------------------------------------- Step 9: save
log("\nSTEP 9: Save cleaned dataset")
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)
log(f"Saved {len(df):,} rows and {df.shape[1]} columns to {CLEAN_PATH}")

LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
print(f"\nData-quality log written to {LOG_PATH}")

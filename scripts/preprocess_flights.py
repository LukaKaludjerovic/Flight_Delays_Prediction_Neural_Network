import pandas as pd
import glob
import os
import time
from pathlib import Path

# 1. Path to directory where CSV files are stored
raw_data_dir = "/Users/lukakaludjerovic/Downloads/BTS Data"
csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))

if not csv_files:
  raise FileNotFoundError(f"CSV files not found in {raw_data_dir}.")

print(f"Found {len(csv_files)} monthly files. Starting processing...")
start_time = time.time()

# Columns to extract
use_cols = [
  "Year",
  "Month",
  "DayofMonth",
  "DayOfWeek",
  "FlightDate",
  "Reporting_Airline",
  "Tail_Number",
  "Flight_Number_Reporting_Airline",
  "Origin",
  "Dest",
  "Distance",
  "CRSDepTime",
  "DepTime",
  "CRSArrTime",
  "ArrTime",
  "CRSElapsedTime",
  "ActualElapsedTime",
  "AirTime",
  "TaxiOut",
  "TaxiIn",
  "DepDelay",
  "DepDel15",
  "ArrDelay",
  "ArrDel15",
  "CarrierDelay",
  "WeatherDelay",
  "NASDelay",
  "SecurityDelay",
  "LateAircraftDelay",
  "Cancelled",
  "Diverted"
]

# Target airlines (Delta, American, United, Southwest)
target_airlines = ["DL", "AA", "UA", "WN"]

# Cause of delay columns
delay_cause_cols = [
  "CarrierDelay",
  "WeatherDelay",
  "NASDelay",
  "SecurityDelay",
  "LateAircraftDelay"
]

processed_chunks = []

for file in sorted(csv_files):
  print(f"Processing: {os.path.basename(file)}")
  
  chunk_iter = pd.read_csv(
    file,
    low_memory=False,
    chunksize=150000,
    usecols=use_cols
  )
    
  for chunk in chunk_iter:
    # 1. Keep only target airlines
    chunk = chunk[chunk["Reporting_Airline"].isin(target_airlines)]
    
    # 2. Remove cancelled and diverted flights
    chunk = chunk[(chunk["Cancelled"] == 0) & (chunk["Diverted"] == 0)]
    
    # 3. Remove rows without basic data or target delays
    chunk = chunk.dropna(subset=[
      "Tail_Number",
      "DepDelay",
      "ArrDelay",
      "DepDel15",
      "ArrDel15"
    ]).copy()
    
    # 4. Fill NaN with 0 in delay cause columns
    chunk[delay_cause_cols] = chunk[delay_cause_cols].fillna(0)
    
    # 5. Delete cancelled and diverted columns
    chunk = chunk.drop(columns=["Cancelled", "Diverted"])
    
    processed_chunks.append(chunk)

print("Merging all parts into main DataFrame...")
master_df = pd.concat(processed_chunks, ignore_index=True)

# Convert dates and sort chronologically per aircraft (Tail_Number)
print("Sorting flights per aircraft rotations (Tail_Number)...")
master_df["FlightDate"] = pd.to_datetime(master_df["FlightDate"])
master_df = master_df.sort_values(by=["Tail_Number", "FlightDate", "CRSDepTime"]).reset_index(drop=True)

# Optimisation of data types in order to save memory
int_cols = ["Year", "Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime", "DepDel15", "ArrDel15"]
for col in int_cols:
  master_df[col] = master_df[col].astype("int32")

project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "data" / "parquet"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "flights_2024_2025_clean.parquet"

print(f"Saving in {output_file}...")
master_df.to_parquet(output_file, compression="snappy")

elapsed = (time.time() - start_time) / 60
print(f"\nFinished in {elapsed:.2f} minutes!")
print(f"Total number of clean flights: {len(master_df):,}")
print(f"File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")

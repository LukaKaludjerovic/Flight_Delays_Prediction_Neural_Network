from pathlib import Path
import numpy as np
import pandas as pd

data_path = Path("data/parquet/flights_2024_2025_clean.parquet")
out_dir = Path("data/features")

carriers = {"DL", "AA", "UA", "WN"}
hubs = {"ATL", "ORD", "DFW", "DEN", "JFK", "LAX"}


def parse_hhmm_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
  s = series.fillna(0).astype(int).astype(str).str.zfill(4)
  hh = pd.to_numeric(s.str[:2], errors="coerce").fillna(0).clip(0, 23).astype(int)
  mm = pd.to_numeric(s.str[2:4], errors="coerce").fillna(0).clip(0, 59).astype(int)

  return hh, mm


def build_timestamps(df: pd.DataFrame) -> pd.DataFrame:
  base_date = pd.to_datetime(df["FlightDate"])
  dep_h, dep_m = parse_hhmm_series(df["CRSDepTime"])
  arr_h, arr_m = parse_hhmm_series(df["CRSArrTime"])

  sched_dep_dt = (
    base_date
    + pd.to_timedelta(dep_h, unit="h")
    + pd.to_timedelta(dep_m, unit="m")
  )

  raw_arr_dt = (
    base_date
    + pd.to_timedelta(arr_h, unit="h")
    + pd.to_timedelta(arr_m, unit="m")
  )

  sched_arr_dt = np.where(
    raw_arr_dt < sched_dep_dt,
    raw_arr_dt + pd.Timedelta(days=1),
    raw_arr_dt,
  )

  df["sched_dep_dt"] = sched_dep_dt
  df["sched_arr_dt"] = pd.to_datetime(sched_arr_dt)
  df["dep_hour"] = dep_h.astype(np.int8)

  return df


def add_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.sort_values(["Tail_Number", "sched_dep_dt", "sched_arr_dt"]).reset_index(drop=True)
  grp = df.groupby("Tail_Number", sort=False)

  prev_sched_arr = grp["sched_arr_dt"].shift(1)
  prev_dep_delay = grp["DepDelay"].shift(1)
  prev_arr_delay = grp["ArrDelay"].shift(1)
  prev_arr_del15 = grp["ArrDel15"].shift(1)
  prev_dest = grp["Dest"].shift(1)

  gap_hours = (df["sched_dep_dt"] - prev_sched_arr).dt.total_seconds() / 3600.0
  valid_chain = (
    df["Tail_Number"].notna()
    & (df["Tail_Number"] != "")
    & prev_sched_arr.notna()
    & (gap_hours >= -2.0)
    & (gap_hours <= 18.0)
  )

  df["has_prev_leg"] = valid_chain.astype(np.int8)
  df["sched_turnaround_min"] = np.where(
    valid_chain,
    (df["sched_dep_dt"] - prev_sched_arr).dt.total_seconds() / 60.0,
    np.nan,
  ).astype(np.float32)

  df["prev_dep_delay"] = np.where(valid_chain, prev_dep_delay, 0.0).astype(np.float32)
  df["prev_arr_delay"] = np.where(valid_chain, prev_arr_delay, 0.0).astype(np.float32)
  df["prev_arr_del15"] = np.where(valid_chain, prev_arr_del15, 0.0).astype(np.float32)
  df["effective_turnaround_min"] = (
    df["sched_turnaround_min"] - df["prev_arr_delay"]
  ).astype(np.float32)
  df["airport_continuity"] = np.where(
    valid_chain, (prev_dest == df["Origin"]).astype(np.int8), 0
  )

  return df


def add_hub_and_time_features(df: pd.DataFrame) -> pd.DataFrame:
  df["is_origin_hub"] = df["Origin"].isin(hubs).astype(np.int8)
  df["is_dest_hub"] = df["Dest"].isin(hubs).astype(np.int8)

  # Strictly causal prior-2h window at Origin before scheduled departure
  df = df.reset_index(drop=True)
  row_id = np.arange(len(df))
  query_now = df["sched_dep_dt"] - pd.Timedelta(seconds=1)
  query_2h = query_now - pd.Timedelta(hours=2)

  right = df.sort_values("sched_dep_dt")[["sched_dep_dt", "Origin", "DepDelay", "ArrDelay"]].copy()
  right["cum_cnt"] = right.groupby("Origin").cumcount() + 1
  right["cum_dep"] = right.groupby("Origin")["DepDelay"].cumsum()
  right["cum_arr"] = right.groupby("Origin")["ArrDelay"].cumsum()

  left_now = pd.DataFrame({"row_id": row_id, "Origin": df["Origin"], "q": query_now}).sort_values("q")
  left_2h = pd.DataFrame({"row_id": row_id, "Origin": df["Origin"], "q": query_2h}).sort_values("q")

  m_now = pd.merge_asof(
    left_now,
    right[["sched_dep_dt", "Origin", "cum_cnt", "cum_dep", "cum_arr"]],
    left_on="q",
    right_on="sched_dep_dt",
    by="Origin",
    direction="backward",
  ).sort_values("row_id").reset_index(drop=True)

  m_2h = pd.merge_asof(
    left_2h,
    right[["sched_dep_dt", "Origin", "cum_cnt", "cum_dep", "cum_arr"]],
    left_on="q",
    right_on="sched_dep_dt",
    by="Origin",
    direction="backward",
  ).sort_values("row_id").reset_index(drop=True)

  cnt_2h = (m_now["cum_cnt"].fillna(0) - m_2h["cum_cnt"].fillna(0)).clip(lower=0)
  sum_dep_2h = m_now["cum_dep"].fillna(0.0) - m_2h["cum_dep"].fillna(0.0)
  sum_arr_2h = m_now["cum_arr"].fillna(0.0) - m_2h["cum_arr"].fillna(0.0)

  df["origin_deps_prev_2h"] = cnt_2h.astype(np.float32)
  df["origin_mean_dep_delay_prev_2h"] = np.where(
    cnt_2h > 0, sum_dep_2h / cnt_2h.replace(0, np.nan), 0.0
  ).astype(np.float32)
  df["origin_mean_arr_delay_prev_2h"] = np.where(
    cnt_2h > 0, sum_arr_2h / cnt_2h.replace(0, np.nan), 0.0
  ).astype(np.float32)

  df["hour_sin"] = np.sin(2 * np.pi * df["dep_hour"] / 24.0).astype(np.float32)
  df["hour_cos"] = np.cos(2 * np.pi * df["dep_hour"] / 24.0).astype(np.float32)
  df["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7.0).astype(np.float32)
  df["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7.0).astype(np.float32)

  return df


def main() -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  df = pd.read_parquet(data_path)

  df = build_timestamps(df)
  df = add_rotation_features(df)
  df = add_hub_and_time_features(df)

  scoped = df[
    df["Reporting_Airline"].isin(carriers)
    & (df["Origin"].isin(hubs) | df["Dest"].isin(hubs))
  ].sort_values("sched_dep_dt").reset_index(drop=True)

  ym = scoped["Year"] * 100 + scoped["Month"]
  train_df = scoped[ym <= 202506]
  val_df = scoped[(ym >= 202507) & (ym <= 202509)]
  test_df = scoped[ym >= 202510]

  scoped.to_parquet(out_dir / "features_scoped.parquet", index=False)
  train_df.to_parquet(out_dir / "train.parquet", index=False)
  val_df.to_parquet(out_dir / "val.parquet", index=False)
  test_df.to_parquet(out_dir / "test.parquet", index=False)

  print(f"Total scoped rows: {len(scoped):,}")
  print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")


if __name__ == "__main__":
    main()
    
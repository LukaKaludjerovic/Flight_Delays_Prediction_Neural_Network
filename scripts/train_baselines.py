from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (
  mean_absolute_error,
  root_mean_squared_error,
  accuracy_score,
  f1_score,
  roc_auc_score,
)

features_dir = Path("data/features")
results_dir = Path("results")

num_cols = [
  "CRSElapsedTime",
  "Distance",
  "has_prev_leg",
  "sched_turnaround_min",
  "prev_dep_delay",
  "prev_arr_delay",
  "prev_arr_del15",
  "effective_turnaround_min",
  "airport_continuity",
  "is_origin_hub",
  "is_dest_hub",
  "origin_deps_prev_2h",
  "origin_mean_dep_delay_prev_2h",
  "origin_mean_arr_delay_prev_2h",
  "hour_sin",
  "hour_cos",
  "dow_sin",
  "dow_cos",
]

cat_cols = ["Reporting_Airline", "Origin", "Dest"]

reg_targets = ["DepDelay", "ArrDelay"]
clf_targets = ["DepDel15", "ArrDel15"]


def prepare_splits():
  cols_needed = list(set(num_cols + cat_cols + reg_targets + clf_targets))
  train_df = pd.read_parquet(features_dir / "train.parquet", columns=cols_needed)
  val_df = pd.read_parquet(features_dir / "val.parquet", columns=cols_needed)
  test_df = pd.read_parquet(features_dir / "test.parquet", columns=cols_needed)

  # Drop rows with missing targets if any exist
  train_df = train_df.dropna(subset=reg_targets + clf_targets).reset_index(drop=True)
  val_df = val_df.dropna(subset=reg_targets + clf_targets).reset_index(drop=True)
  test_df = test_df.dropna(subset=reg_targets + clf_targets).reset_index(drop=True)

  # Fill turnaround NaNs (occur when has_prev_leg == 0) safely with 0.0
  for df in (train_df, val_df, test_df):
    df["sched_turnaround_min"] = df["sched_turnaround_min"].fillna(0.0)
    df["effective_turnaround_min"] = df["effective_turnaround_min"].fillna(0.0)

  # Encode categorical columns on train only
  enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
  X_cat_train = enc.fit_transform(train_df[cat_cols])
  X_cat_val = enc.transform(val_df[cat_cols])
  X_cat_test = enc.transform(test_df[cat_cols])

  # Numeric matrices
  X_num_train = train_df[num_cols].to_numpy(dtype=np.float32)
  X_num_val = val_df[num_cols].to_numpy(dtype=np.float32)
  X_num_test = test_df[num_cols].to_numpy(dtype=np.float32)

  # Scaled numeric matrices for linear/logistic models
  scaler = StandardScaler()
  X_num_train_sc = scaler.fit_transform(X_num_train)
  X_num_val_sc = scaler.transform(X_num_val)
  X_num_test_sc = scaler.transform(X_num_test)

  # One-hot categories for linear/logistic models
  ohe_train = pd.get_dummies(train_df[cat_cols], dtype=np.float32)
  ohe_cols = ohe_train.columns
  ohe_val = pd.get_dummies(val_df[cat_cols], dtype=np.float32).reindex(columns=ohe_cols, fill_value=0.0)
  ohe_test = pd.get_dummies(test_df[cat_cols], dtype=np.float32).reindex(columns=ohe_cols, fill_value=0.0)

  X_lin = {
    "train": np.hstack([X_num_train_sc, ohe_train.to_numpy(dtype=np.float32)]),
    "val": np.hstack([X_num_val_sc, ohe_val.to_numpy(dtype=np.float32)]),
    "test": np.hstack([X_num_test_sc, ohe_test.to_numpy(dtype=np.float32)]),
  }

  X_tree = {
    "train": np.hstack([X_num_train, X_cat_train]),
    "val": np.hstack([X_num_val, X_cat_val]),
    "test": np.hstack([X_num_test, X_cat_test]),
  }

  y = {
    split: {
      col: df[col].to_numpy(dtype=np.float32 if col in reg_targets else np.int32)
      for col in reg_targets + clf_targets
    }
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]
  }

  return X_lin, X_tree, y


def eval_regression(y_true, y_pred) -> dict:
  return {
    "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
    "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 4),
  }


def eval_classification(y_true, y_prob, threshold: float = 0.5) -> dict:
  y_pred = (y_prob >= threshold).astype(np.int32)
  return {
    "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    "F1": round(float(f1_score(y_true, y_pred)), 4),
    "AUC_ROC": round(float(roc_auc_score(y_true, y_prob)), 4),
  }


def main() -> None:
  results_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  print("Loading and preparing feature splits...")
  X_lin, X_tree, y = prepare_splits()
  print(f"Prepared splits in {time.time() - t0:.1f}s | Train shape: {X_tree["train"].shape}")

  records = []
  cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

  # 1. Regression Baselines
  for target in reg_targets:
    # Ridge
    t1 = time.time()
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_lin["train"], y["train"][target])
    for split in ("val", "test"):
      preds = ridge.predict(X_lin[split])
      metrics = eval_regression(y[split][target], preds)
      records.append({"model": "Ridge", "target": target, "split": split, **metrics})
    print(f"[Ridge - {target}] done in {time.time() - t1:.1f}s | Test: {records[-1]}")

    # HistGradientBoostingRegressor
    t1 = time.time()
    hgb_reg = HistGradientBoostingRegressor(
      max_iter=150,
      learning_rate=0.08,
      max_leaf_nodes=63,
      categorical_features=cat_indices,
      random_state=42,
    )
    hgb_reg.fit(X_tree["train"], y["train"][target])
    for split in ("val", "test"):
      preds = hgb_reg.predict(X_tree[split])
      metrics = eval_regression(y[split][target], preds)
      records.append({"model": "HistGB_Reg", "target": target, "split": split, **metrics})
    print(f"[HistGB_Reg - {target}] done in {time.time() - t1:.1f}s | Test: {records[-1]}")

  # 2. Classification Baselines
  for target in clf_targets:
    # Logistic Regression via SGD (fast on 2.75M rows)
    t1 = time.time()
    sgd_clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=15, tol=1e-3, random_state=42)
    sgd_clf.fit(X_lin["train"], y["train"][target])
    for split in ("val", "test"):
      probs = sgd_clf.predict_proba(X_lin[split])[:, 1]
      metrics = eval_classification(y[split][target], probs)
      records.append({"model": "Logistic_SGD", "target": target, "split": split, **metrics})
    print(f"[Logistic_SGD - {target}] done in {time.time() - t1:.1f}s | Test: {records[-1]}")

    # HistGradientBoostingClassifier
    t1 = time.time()
    hgb_clf = HistGradientBoostingClassifier(
      max_iter=150,
      learning_rate=0.08,
      max_leaf_nodes=63,
      categorical_features=cat_indices,
      random_state=42,
    )
    hgb_clf.fit(X_tree["train"], y["train"][target])
    for split in ("val", "test"):
      probs = hgb_clf.predict_proba(X_tree[split])[:, 1]
      metrics = eval_classification(y[split][target], probs)
      records.append({"model": "HistGB_Clf", "target": target, "split": split, **metrics})
    print(f"[HistGB_Clf - {target}] done in {time.time() - t1:.1f}s | Test: {records[-1]}")

  results_df = pd.DataFrame(records)
  results_df.to_csv(results_dir / "baseline_metrics.csv", index=False)
  with open(results_dir / "baseline_metrics.json", "w") as f:
    json.dump(records, f, indent=2)

  print("\n=== Baseline Summary (Test Split) ===")
  print(results_df[results_df["split"] == "test"].to_string(index=False))


if __name__ == "__main__":
  main()

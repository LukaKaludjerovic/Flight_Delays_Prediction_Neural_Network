from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
  mean_absolute_error,
  root_mean_squared_error,
  accuracy_score,
  f1_score,
  roc_auc_score,
)

features_dir = Path("data/features")
results_dir = Path("results")

base_num_cols = [
  "CRSElapsedTime",
  "Distance",
  "hour_sin",
  "hour_cos",
  "dow_sin",
  "dow_cos",
]

rotation_num_cols = [
  "has_prev_leg",
  "sched_turnaround_min",
  "prev_dep_delay",
  "prev_arr_delay",
  "prev_arr_del15",
  "effective_turnaround_min",
  "airport_continuity",
]

hub_num_cols = [
  "is_origin_hub",
  "is_dest_hub",
  "origin_deps_prev_2h",
  "origin_mean_dep_delay_prev_2h",
  "origin_mean_arr_delay_prev_2h",
]

ablations = {
  "base_only": base_num_cols,
  "rotation_only": base_num_cols + rotation_num_cols,
  "hub_only": base_num_cols + hub_num_cols,
  "full_hybrid": base_num_cols + rotation_num_cols + hub_num_cols,
}

cat_cols = ["Reporting_Airline", "Origin", "Dest", "Tail_Number"]
reg_targets = ["DepDelay", "ArrDelay"]
clf_targets = ["DepDel15", "ArrDel15"]


class FlightDataset(Dataset):
  def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray):
    self.x_num = torch.from_numpy(x_num.copy())
    self.x_cat = torch.from_numpy(x_cat.copy())
    self.y_reg = torch.from_numpy(y_reg.copy())
    self.y_clf = torch.from_numpy(y_clf.copy())

  def __len__(self) -> int:
    return len(self.x_num)

  def __getitem__(self, idx: int):
    return self.x_num[idx], self.x_cat[idx], self.y_reg[idx], self.y_clf[idx]


class DelayPropagationNet(nn.Module):
  def __init__(self, vocab_sizes: list[int], num_features: int, hidden_dim: int = 256, dropout: float = 0.2):
    super().__init__()
    self.embeddings = nn.ModuleList([
      nn.Embedding(num_classes, emb_dim)
      for num_classes, emb_dim in zip(vocab_sizes, [8, 16, 16, 32])
    ])
    emb_total = sum(e.embedding_dim for e in self.embeddings)
    in_dim = emb_total + num_features

    self.input_norm = nn.LayerNorm(in_dim)
    self.fc1 = nn.Linear(in_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.act = nn.SiLU()
    self.drop = nn.Dropout(dropout)

    self.reg_head = nn.Linear(hidden_dim, 2)
    self.clf_head = nn.Linear(hidden_dim, 2)

  def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
    emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
    x = torch.cat(emb_outs + [x_num], dim=1)
    x = self.input_norm(x)

    h1 = self.drop(self.act(self.fc1(x)))
    h2 = self.drop(self.act(self.fc2(h1))) + h1

    return self.reg_head(h2), self.clf_head(h2)


def load_raw_splits() -> tuple[dict[str, pd.DataFrame], list[int]]:
  all_num = sorted(set(base_num_cols + rotation_num_cols + hub_num_cols))
  needed = list(set(all_num + cat_cols + reg_targets + clf_targets))

  dfs = {
    split: pd.read_parquet(features_dir / f"{split}.parquet", columns=needed).dropna(
      subset=reg_targets + clf_targets
    ).reset_index(drop=True)
    for split in ("train", "val", "test")
  }

  for df in dfs.values():
    df["sched_turnaround_min"] = df["sched_turnaround_min"].fillna(0.0)
    df["effective_turnaround_min"] = df["effective_turnaround_min"].fillna(0.0)
    for col in cat_cols:
      df[col] = df[col].fillna("UNK").astype(str)

  vocabs = {}
  vocab_sizes = []
  for col in cat_cols:
    uniq = sorted(dfs["train"][col].unique().tolist())
    vocabs[col] = {val: idx + 1 for idx, val in enumerate(uniq)}
    vocab_sizes.append(len(uniq) + 1)

  for df in dfs.values():
    for i, col in enumerate(cat_cols):
      df[f"_cat_{i}"] = df[col].map(vocabs[col]).fillna(0).astype(np.int64)

  return dfs, vocab_sizes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
  model.eval()
  reg_preds_list, reg_trues_list = [], []
  clf_probs_list, clf_trues_list = [], []

  for x_num, x_cat, y_reg, y_clf in loader:
    x_num = x_num.to(device, non_blocking=True)
    x_cat = x_cat.to(device, non_blocking=True)
    reg_out, clf_logits = model(x_num, x_cat)

    reg_preds_list.append(reg_out.cpu().numpy())
    reg_trues_list.append(y_reg.numpy())
    clf_probs_list.append(torch.sigmoid(clf_logits).cpu().numpy())
    clf_trues_list.append(y_clf.numpy())

  reg_preds = np.vstack(reg_preds_list)
  reg_trues = np.vstack(reg_trues_list)
  clf_probs = np.vstack(clf_probs_list)
  clf_trues = np.vstack(clf_trues_list)

  metrics = {}
  for i, col in enumerate(reg_targets):
    metrics[f"{col}_MAE"] = round(float(mean_absolute_error(reg_trues[:, i], reg_preds[:, i])), 4)
    metrics[f"{col}_RMSE"] = round(float(root_mean_squared_error(reg_trues[:, i], reg_preds[:, i])), 4)

  for i, col in enumerate(clf_targets):
    y_bin = clf_trues[:, i].astype(np.int32)
    p_bin = clf_probs[:, i]
    pred_bin = (p_bin >= 0.5).astype(np.int32)
    metrics[f"{col}_Accuracy"] = round(float(accuracy_score(y_bin, pred_bin)), 4)
    metrics[f"{col}_F1"] = round(float(f1_score(y_bin, pred_bin)), 4)
    metrics[f"{col}_AUC_ROC"] = round(float(roc_auc_score(y_bin, p_bin)), 4)

  return metrics


def run_variant(
  name: str,
  num_cols: list[str],
  dfs: dict[str, pd.DataFrame],
  vocab_sizes: list[int],
  device: torch.device,
  epochs: int = 6,
  batch_size: int = 4096,
) -> dict:
  scaler = StandardScaler()
  scaler.fit(dfs["train"][num_cols].to_numpy(dtype=np.float32))

  cat_cols_idx = [f"_cat_{i}" for i in range(len(cat_cols))]
  loaders = {}
  for split in ("train", "val", "test"):
    x_num = scaler.transform(dfs[split][num_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    x_cat = dfs[split][cat_cols_idx].to_numpy(dtype=np.int64)
    y_reg = dfs[split][reg_targets].to_numpy(dtype=np.float32)
    y_clf = dfs[split][clf_targets].to_numpy(dtype=np.float32)
    ds = FlightDataset(x_num, x_cat, y_reg, y_clf)
    loaders[split] = DataLoader(
      ds,
      batch_size=batch_size if split == "train" else batch_size * 2,
      shuffle=(split == "train"),
    )

  model = DelayPropagationNet(vocab_sizes=vocab_sizes, num_features=len(num_cols)).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
  reg_loss_fn = nn.SmoothL1Loss(beta=15.0)
  clf_loss_fn = nn.BCEWithLogitsLoss()

  best_val_score = float("inf")
  best_weights = None

  for _ in range(1, epochs + 1):
    model.train()
    for x_num, x_cat, y_reg, y_clf in loaders["train"]:
      x_num = x_num.to(device)
      x_cat = x_cat.to(device)
      y_reg = y_reg.to(device)
      y_clf = y_clf.to(device)

      optimizer.zero_grad()
      reg_out, clf_logits = model(x_num, x_cat)
      loss = (reg_loss_fn(reg_out, y_reg) / 15.0) + clf_loss_fn(clf_logits, y_clf)
      loss.backward()
      optimizer.step()

    val_m = evaluate(model, loaders["val"], device)
    val_score = val_m["DepDelay_MAE"] + val_m["ArrDelay_MAE"]
    if val_score < best_val_score:
      best_val_score = val_score
      best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

  if best_weights is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

  test_metrics = evaluate(model, loaders["test"], device)
  return {"variant": name, "num_features": len(num_cols), **test_metrics}


def main() -> None:
  torch.manual_seed(42)
  np.random.seed(42)
  results_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
  )
  print(f"Using device: {device}")

  t0 = time.time()
  dfs, vocab_sizes = load_raw_splits()
  print(f"Loaded splits in {time.time() - t0:.1f}s")

  results = []
  for name, cols in ablations.items():
    t_v = time.time()
    print(f"\nRunning ablation variant: {name} ({len(cols)} numeric features)...")
    res = run_variant(name, cols, dfs, vocab_sizes, device, epochs=6)
    res["runtime_sec"] = round(time.time() - t_v, 1)
    results.append(res)
    print(
      f"[{name}] done in {res["runtime_sec"]}s | "
      f"Dep MAE: {res["DepDelay_MAE"]} | Arr MAE: {res["ArrDelay_MAE"]} | "
      f"Dep F1: {res["DepDel15_F1"]} | Arr F1: {res["ArrDel15_F1"]}"
    )

  out_df = pd.DataFrame(results)
  out_df.to_csv(results_dir / "ablation_metrics.csv", index=False)
  with open(results_dir / "ablation_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

  print("\n=== Ablation Summary (Test Split) ===")
  print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

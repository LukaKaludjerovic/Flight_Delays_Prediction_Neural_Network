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

seq_len = 3
seq_feat_dim = 5  # dep_delay, arr_delay, arr_del15, gap_min, valid_mask

current_num_cols = [
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

cat_cols = ["Reporting_Airline", "Origin", "Dest", "Tail_Number"]
reg_targets = ["DepDelay", "ArrDelay"]
clf_targets = ["DepDel15", "ArrDel15"]


class RotationFlightDataset(Dataset):
  def __init__(
    self,
    x_seq: np.ndarray,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y_reg: np.ndarray,
    y_clf: np.ndarray,
  ):
    self.x_seq = torch.from_numpy(x_seq.copy())
    self.x_num = torch.from_numpy(x_num.copy())
    self.x_cat = torch.from_numpy(x_cat.copy())
    self.y_reg = torch.from_numpy(y_reg.copy())
    self.y_clf = torch.from_numpy(y_clf.copy())

  def __len__(self) -> int:
    return len(self.x_num)

  def __getitem__(self, idx: int):
    return (
      self.x_seq[idx],
      self.x_num[idx],
      self.x_cat[idx],
      self.y_reg[idx],
      self.y_clf[idx],
    )


class RotationGRUNet(nn.Module):
  def __init__(
    self,
    vocab_sizes: list[int],
    num_features: int,
    seq_feat_dim: int = seq_feat_dim,
    gru_hidden: int = 64,
    hidden_dim: int = 256,
    dropout: float = 0.2,
  ):
    super().__init__()
    self.embeddings = nn.ModuleList([
      nn.Embedding(num_classes, emb_dim)
      for num_classes, emb_dim in zip(vocab_sizes, [8, 16, 16, 32])
    ])
    self.gru = nn.GRU(
      input_size=seq_feat_dim,
      hidden_size=gru_hidden,
      num_layers=1,
      batch_first=True,
    )

    emb_total = sum(e.embedding_dim for e in self.embeddings)
    in_dim = gru_hidden + emb_total + num_features

    self.input_norm = nn.LayerNorm(in_dim)
    self.fc1 = nn.Linear(in_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.act = nn.SiLU()
    self.drop = nn.Dropout(dropout)

    self.reg_head = nn.Linear(hidden_dim, 2)
    self.clf_head = nn.Linear(hidden_dim, 2)

  def forward(self, x_seq: torch.Tensor, x_num: torch.Tensor, x_cat: torch.Tensor):
    _, h_n = self.gru(x_seq)
    seq_repr = h_n[-1]  # (B, gru_hidden)

    emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
    x = torch.cat([seq_repr] + emb_outs + [x_num], dim=1)
    x = self.input_norm(x)

    h1 = self.drop(self.act(self.fc1(x)))
    h2 = self.drop(self.act(self.fc2(h1))) + h1

    return self.reg_head(h2), self.clf_head(h2)


def build_rotation_sequences(df: pd.DataFrame) -> np.ndarray:
  """
  For each flight at index i in Tail_Number rotation order, builds a causal tensor
  of shape (N, SEQ_LEN, SEQ_FEAT_DIM) representing [lag-3, lag-2, lag-1] prior legs.
  """
  n = len(df)
  seq_tensor = np.zeros((n, seq_len, seq_feat_dim), dtype=np.float32)

  tail_vals = df["Tail_Number"].to_numpy()
  sched_dep = pd.to_datetime(df["sched_dep_dt"]).to_numpy(dtype="datetime64[s]")
  sched_arr = pd.to_datetime(df["sched_arr_dt"]).to_numpy(dtype="datetime64[s]")
  dep_delay = df["DepDelay"].to_numpy(dtype=np.float32)
  arr_delay = df["ArrDelay"].to_numpy(dtype=np.float32)
  arr_del15 = df["ArrDel15"].to_numpy(dtype=np.float32)

  # Group row indices by Tail_Number (df is already sorted by Tail_Number, sched_dep_dt)
  unique_tails, start_idx, counts = np.unique(tail_vals, return_index=True, return_counts=True)

  for tail, start, cnt in zip(unique_tails, start_idx, counts):
    if tail == "" or tail == "UNK" or cnt <= 1:
      continue
    end = start + cnt
    for pos in range(start + 1, end):
      # Collect up to SEQ_LEN immediate predecessors on the same aircraft
      collected = []
      curr_idx = pos
      for _ in range(seq_len):
        prev_idx = curr_idx - 1
        if prev_idx < start:
          break
        gap_min = float((sched_dep[curr_idx] - sched_arr[prev_idx]) / np.timedelta64(1, "m"))
        if gap_min < -120.0 or gap_min > 1080.0:
          break
        collected.append([
          dep_delay[prev_idx],
          arr_delay[prev_idx],
          arr_del15[prev_idx],
          np.clip(gap_min, -120.0, 1080.0),
          1.0,
        ])
        curr_idx = prev_idx

      if collected:
        # Reverse so order is chronological: oldest lag -> most recent lag
        collected.reverse()
        k = len(collected)
        seq_tensor[pos, seq_len - k :, :] = np.asarray(collected, dtype=np.float32)

  return seq_tensor


def prepare_data():
  needed = list(
    set(
      current_num_cols
      + cat_cols
      + reg_targets
      + clf_targets
      + ["sched_dep_dt", "sched_arr_dt"]
    )
  )

  dfs = {}
  seqs = {}
  for split in ("train", "val", "test"):
    df = pd.read_parquet(features_dir / f"{split}.parquet", columns=needed)
    df = df.dropna(subset=reg_targets + clf_targets).copy()
    df["sched_turnaround_min"] = df["sched_turnaround_min"].fillna(0.0)
    df["effective_turnaround_min"] = df["effective_turnaround_min"].fillna(0.0)
    for col in cat_cols:
      df[col] = df[col].fillna("UNK").astype(str)

    # Sort by rotation chain first so sequence lags are exact, then preserve row order
    df = df.sort_values(["Tail_Number", "sched_dep_dt", "sched_arr_dt"]).reset_index(drop=True)
    seqs[split] = build_rotation_sequences(df)

    # Shuffle row order for train after computing causal sequences
    if split == "train":
      perm = np.random.permutation(len(df))
      df = df.iloc[perm].reset_index(drop=True)
      seqs[split] = seqs[split][perm]

    dfs[split] = df

  vocabs = {}
  vocab_sizes = []
  for col in cat_cols:
    uniq = sorted(dfs["train"][col].unique().tolist())
    vocabs[col] = {val: idx + 1 for idx, val in enumerate(uniq)}
    vocab_sizes.append(len(uniq) + 1)

  # Fit scalers strictly on train
  num_scaler = StandardScaler()
  num_scaler.fit(dfs["train"][current_num_cols].to_numpy(dtype=np.float32))

  # Scale continuous sequence features (cols 0, 1, 3) while keeping binary mask/del15 clean
  seq_cont_idx = [0, 1, 3]
  seq_scaler = StandardScaler()
  train_seq_flat = seqs["train"][:, :, seq_cont_idx].reshape(-1, len(seq_cont_idx))
  seq_scaler.fit(train_seq_flat)

  arrays = {}
  for split, df in dfs.items():
    x_num = num_scaler.transform(df[current_num_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    x_cat = np.column_stack([
      df[col].map(vocabs[col]).fillna(0).to_numpy(dtype=np.int64)
      for col in cat_cols
    ])

    x_seq = seqs[split].copy()
    flat_c = x_seq[:, :, seq_cont_idx].reshape(-1, len(seq_cont_idx))
    x_seq[:, :, seq_cont_idx] = seq_scaler.transform(flat_c).reshape(x_seq.shape[0], seq_len, len(seq_cont_idx))
    # Zero out padded timesteps after scaling so padding stays exact 0.0
    mask = x_seq[:, :, 4:5]
    x_seq = (x_seq * mask).astype(np.float32)

    y_reg = df[reg_targets].to_numpy(dtype=np.float32)
    y_clf = df[clf_targets].to_numpy(dtype=np.float32)
    arrays[split] = (x_seq, x_num, x_cat, y_reg, y_clf)

  return arrays, vocab_sizes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
  model.eval()
  reg_preds_list, reg_trues_list = [], []
  clf_probs_list, clf_trues_list = [], []

  for x_seq, x_num, x_cat, y_reg, y_clf in loader:
    x_seq = x_seq.to(device)
    x_num = x_num.to(device)
    x_cat = x_cat.to(device)

    reg_out, clf_logits = model(x_seq, x_num, x_cat)
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
  arrays, vocab_sizes = prepare_data()
  print(f"Prepared rotation sequences + tensors in {time.time() - t0:.1f}s")

  batch_size = 4096
  train_ds = RotationFlightDataset(*arrays["train"])
  val_ds = RotationFlightDataset(*arrays["val"])
  test_ds = RotationFlightDataset(*arrays["test"])

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
  test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

  model = RotationGRUNet(vocab_sizes=vocab_sizes, num_features=len(current_num_cols)).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
  reg_loss_fn = nn.SmoothL1Loss(beta=15.0)
  clf_loss_fn = nn.BCEWithLogitsLoss()

  epochs = 6
  best_val_score = float("inf")
  best_weights = None

  for epoch in range(1, epochs + 1):
    t_ep = time.time()
    model.train()
    running_loss = 0.0

    for x_seq, x_num, x_cat, y_reg, y_clf in train_loader:
      x_seq = x_seq.to(device)
      x_num = x_num.to(device)
      x_cat = x_cat.to(device)
      y_reg = y_reg.to(device)
      y_clf = y_clf.to(device)

      optimizer.zero_grad()
      reg_out, clf_logits = model(x_seq, x_num, x_cat)
      loss = (reg_loss_fn(reg_out, y_reg) / 15.0) + clf_loss_fn(clf_logits, y_clf)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * len(x_num)

    val_m = evaluate(model, val_loader, device)
    val_score = val_m["DepDelay_MAE"] + val_m["ArrDelay_MAE"]

    print(
      f"Epoch {epoch}/{epochs} ({time.time() - t_ep:.1f}s) | "
      f"TrainLoss: {running_loss / len(train_ds):.4f} | "
      f"Val Dep MAE: {val_m["DepDelay_MAE"]} | "
      f"Val Arr MAE: {val_m["ArrDelay_MAE"]} | "
      f"Val Dep F1: {val_m["DepDel15_F1"]} | "
      f"Val Arr F1: {val_m["ArrDel15_F1"]}"
    )

    if val_score < best_val_score:
      best_val_score = val_score
      best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

  if best_weights is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

  test_metrics = evaluate(model, test_loader, device)
  print("\n=== GRU Rotation Model Test Metrics ===")
  for k, v in test_metrics.items():
    print(f"  {k}: {v}")

  with open(results_dir / "gru_rotation_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
  main()

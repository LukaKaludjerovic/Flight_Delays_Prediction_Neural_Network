from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, f1_score

features_dir = Path("data/features")
results_dir = Path("results")
plots_dir = results_dir / "plots"

seq_len = 3
seq_feat_dim = 5
hubs = ["ATL", "ORD", "DFW", "DEN", "JFK", "LAX"]

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
  def __init__(self, x_seq, x_num, x_cat, y_reg, y_clf):
    self.x_seq = torch.from_numpy(x_seq.copy())
    self.x_num = torch.from_numpy(x_num.copy())
    self.x_cat = torch.from_numpy(x_cat.copy())
    self.y_reg = torch.from_numpy(y_reg.copy())
    self.y_clf = torch.from_numpy(y_clf.copy())

  def __len__(self):
    return len(self.x_num)

  def __getitem__(self, idx):
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

  def forward(self, x_seq, x_num, x_cat):
    _, h_n = self.gru(x_seq)
    seq_repr = h_n[-1]
    emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
    x = torch.cat([seq_repr] + emb_outs + [x_num], dim=1)
    x = self.input_norm(x)

    h1 = self.drop(self.act(self.fc1(x)))
    h2 = self.drop(self.act(self.fc2(h1))) + h1
    return self.reg_head(h2), self.clf_head(h2)


def build_rotation_sequences(df: pd.DataFrame) -> np.ndarray:
  n = len(df)
  seq_tensor = np.zeros((n, seq_len, seq_feat_dim), dtype=np.float32)

  tail_vals = df["Tail_Number"].to_numpy()
  sched_dep = pd.to_datetime(df["sched_dep_dt"]).to_numpy(dtype="datetime64[s]")
  sched_arr = pd.to_datetime(df["sched_arr_dt"]).to_numpy(dtype="datetime64[s]")
  dep_delay = df["DepDelay"].to_numpy(dtype=np.float32)
  arr_delay = df["ArrDelay"].to_numpy(dtype=np.float32)
  arr_del15 = df["ArrDel15"].to_numpy(dtype=np.float32)

  unique_tails, start_idx, counts = np.unique(tail_vals, return_index=True, return_counts=True)

  for tail, start, cnt in zip(unique_tails, start_idx, counts):
    if tail == "" or tail == "UNK" or cnt <= 1:
      continue
    end = start + cnt
    for pos in range(start + 1, end):
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
        collected.reverse()
        k = len(collected)
        seq_tensor[pos, seq_len - k :, :] = np.asarray(collected, dtype=np.float32)

  return seq_tensor


def compute_slice_metrics(sub: pd.DataFrame, group_name: str) -> dict:
  return {
    "group": group_name,
    "count": len(sub),
    "DepDelay_MAE": round(float(mean_absolute_error(sub["DepDelay"], sub["pred_DepDelay"])), 4),
    "DepDelay_RMSE": round(float(root_mean_squared_error(sub["DepDelay"], sub["pred_DepDelay"])), 4),
    "ArrDelay_MAE": round(float(mean_absolute_error(sub["ArrDelay"], sub["pred_ArrDelay"])), 4),
    "ArrDelay_RMSE": round(float(root_mean_squared_error(sub["ArrDelay"], sub["pred_ArrDelay"])), 4),
    "DepDel15_F1": round(float(f1_score(sub["DepDel15"], sub["pred_DepDel15"], zero_division=0)), 4),
    "ArrDel15_F1": round(float(f1_score(sub["ArrDel15"], sub["pred_ArrDel15"], zero_division=0)), 4),
  }


def main() -> None:
  torch.manual_seed(42)
  np.random.seed(42)
  plots_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
  )
  print(f"Using device: {device}")

  needed = list(
    set(
      current_num_cols
      + cat_cols
      + reg_targets
      + clf_targets
      + ["sched_dep_dt", "sched_arr_dt", "dep_hour"]
    )
  )

  dfs, seqs = {}, {}
  for split in ("train", "val", "test"):
    df = pd.read_parquet(features_dir / f"{split}.parquet", columns=needed)
    df = df.dropna(subset=reg_targets + clf_targets).copy()
    df["raw_eff_turnaround"] = df["effective_turnaround_min"]
    df["sched_turnaround_min"] = df["sched_turnaround_min"].fillna(0.0)
    df["effective_turnaround_min"] = df["effective_turnaround_min"].fillna(0.0)
    for col in cat_cols:
      df[col] = df[col].fillna("UNK").astype(str)

    df = df.sort_values(["Tail_Number", "sched_dep_dt", "sched_arr_dt"]).reset_index(drop=True)
    seqs[split] = build_rotation_sequences(df)

    if split == "train":
      perm = np.random.permutation(len(df))
      df = df.iloc[perm].reset_index(drop=True)
      seqs[split] = seqs[split][perm]

    dfs[split] = df

  vocabs, vocab_sizes = {}, []
  for col in cat_cols:
    uniq = sorted(dfs["train"][col].unique().tolist())
    vocabs[col] = {val: idx + 1 for idx, val in enumerate(uniq)}
    vocab_sizes.append(len(uniq) + 1)

  num_scaler = StandardScaler().fit(dfs["train"][current_num_cols].to_numpy(dtype=np.float32))
  seq_cont_idx = [0, 1, 3]
  seq_scaler = StandardScaler().fit(seqs["train"][:, :, seq_cont_idx].reshape(-1, len(seq_cont_idx)))

  loaders = {}
  for split, df in dfs.items():
    x_num = num_scaler.transform(df[current_num_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    x_cat = np.column_stack([df[col].map(vocabs[col]).fillna(0).to_numpy(dtype=np.int64) for col in cat_cols])
    x_seq = seqs[split].copy()
    flat_c = x_seq[:, :, seq_cont_idx].reshape(-1, len(seq_cont_idx))
    x_seq[:, :, seq_cont_idx] = seq_scaler.transform(flat_c).reshape(x_seq.shape[0], seq_len, len(seq_cont_idx))
    x_seq = (x_seq * x_seq[:, :, 4:5]).astype(np.float32)

    y_reg = df[reg_targets].to_numpy(dtype=np.float32)
    y_clf = df[clf_targets].to_numpy(dtype=np.float32)
    ds = RotationFlightDataset(x_seq, x_num, x_cat, y_reg, y_clf)
    loaders[split] = DataLoader(ds, batch_size=4096 if split == "train" else 8192, shuffle=(split == "train"))

  model = RotationGRUNet(vocab_sizes=vocab_sizes, num_features=len(current_num_cols)).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
  reg_loss_fn = nn.SmoothL1Loss(beta=15.0)
  clf_loss_fn = nn.BCEWithLogitsLoss()

  best_val, best_weights = float("inf"), None
  for epoch in range(1, 7):
    t0 = time.time()
    model.train()
    for x_seq, x_num, x_cat, y_reg, y_clf in loaders["train"]:
      x_seq, x_num, x_cat = x_seq.to(device), x_num.to(device), x_cat.to(device)
      y_reg, y_clf = y_reg.to(device), y_clf.to(device)
      optimizer.zero_grad()
      reg_out, clf_logits = model(x_seq, x_num, x_cat)
      loss = (reg_loss_fn(reg_out, y_reg) / 15.0) + clf_loss_fn(clf_logits, y_clf)
      loss.backward()
      optimizer.step()

    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
      for x_seq, x_num, x_cat, y_reg, _ in loaders["val"]:
        reg_out, _ = model(x_seq.to(device), x_num.to(device), x_cat.to(device))
        val_preds.append(reg_out.cpu().numpy())
        val_trues.append(y_reg.numpy())
    vp, vt = np.vstack(val_preds), np.vstack(val_trues)
    score = mean_absolute_error(vt[:, 0], vp[:, 0]) + mean_absolute_error(vt[:, 1], vp[:, 1])
    print(f"Epoch {epoch}/6 ({time.time() - t0:.1f}s) | Val combined MAE: {score:.4f}")
    if score < best_val:
      best_val = score
      best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

  model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
  model.eval()

  test_reg_preds, test_clf_probs = [], []
  with torch.no_grad():
    for x_seq, x_num, x_cat, _, _ in loaders["test"]:
      reg_out, clf_logits = model(x_seq.to(device), x_num.to(device), x_cat.to(device))
      test_reg_preds.append(reg_out.cpu().numpy())
      test_clf_probs.append(torch.sigmoid(clf_logits).cpu().numpy())

  reg_p = np.vstack(test_reg_preds)
  clf_p = np.vstack(test_clf_probs)

  test_df = dfs["test"].copy()
  test_df["pred_DepDelay"] = reg_p[:, 0]
  test_df["pred_ArrDelay"] = reg_p[:, 1]
  test_df["pred_DepDel15"] = (clf_p[:, 0] >= 0.5).astype(int)
  test_df["pred_ArrDel15"] = (clf_p[:, 1] >= 0.5).astype(int)

  # 1. Carrier Breakdown
  carrier_rows = [
    compute_slice_metrics(test_df[test_df["Reporting_Airline"] == c], c)
    for c in ["DL", "AA", "UA", "WN"]
  ]
  carrier_df = pd.DataFrame(carrier_rows)
  carrier_df.to_csv(results_dir / "error_by_carrier.csv", index=False)

  # 2. Core Hub Breakdown (Origin or Dest == Hub)
  hub_rows = [
    compute_slice_metrics(test_df[(test_df["Origin"] == h) | (test_df["Dest"] == h)], h)
    for h in hubs
  ]
  hub_df = pd.DataFrame(hub_rows)
  hub_df.to_csv(results_dir / "error_by_hub.csv", index=False)

  # 3. Delay Severity Regime (based on actual ArrDelay)
  regime_rows = [
    compute_slice_metrics(test_df[test_df["ArrDelay"] < 15], "OnTime (<15m)"),
    compute_slice_metrics(test_df[(test_df["ArrDelay"] >= 15) & (test_df["ArrDelay"] < 60)], "Moderate (15-59m)"),
    compute_slice_metrics(test_df[test_df["ArrDelay"] >= 60], "Severe (>=60m)"),
  ]
  regime_df = pd.DataFrame(regime_rows)
  regime_df.to_csv(results_dir / "error_by_severity.csv", index=False)

  # 4. Effective Turnaround Slack Breakdown
  valid_rot = test_df[test_df["has_prev_leg"] == 1]
  slack_rows = [
    compute_slice_metrics(test_df[test_df["has_prev_leg"] == 0], "First_Leg_No_Chain"),
    compute_slice_metrics(valid_rot[valid_rot["raw_eff_turnaround"] < 45], "Tight_Slack (<45m)"),
    compute_slice_metrics(
      valid_rot[(valid_rot["raw_eff_turnaround"] >= 45) & (valid_rot["raw_eff_turnaround"] <= 90)],
      "Standard_Slack (45-90m)",
    ),
    compute_slice_metrics(valid_rot[valid_rot["raw_eff_turnaround"] > 90], "Generous_Slack (>90m)"),
  ]
  slack_df = pd.DataFrame(slack_rows)
  slack_df.to_csv(results_dir / "error_by_turnaround_slack.csv", index=False)

  print("\n=== Error by Carrier ===")
  print(carrier_df.to_string(index=False))
  print("\n=== Error by Core Hub ===")
  print(hub_df.to_string(index=False))
  print("\n=== Error by Delay Severity ===")
  print(regime_df.to_string(index=False))
  print("\n=== Error by Turnaround Slack ===")
  print(slack_df.to_string(index=False))

  # Generate Summary Plots for Report
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
  x_hub = np.arange(len(hub_df))
  w = 0.36
  axes[0].bar(x_hub - w / 2, hub_df["DepDelay_MAE"], width=w, label="DepDelay MAE", color="#2b5c8f")
  axes[0].bar(x_hub + w / 2, hub_df["ArrDelay_MAE"], width=w, label="ArrDelay MAE", color="#d97724")
  axes[0].set_xticks(x_hub)
  axes[0].set_xticklabels(hub_df["group"])
  axes[0].set_ylim(0, hub_df[["DepDelay_MAE", "ArrDelay_MAE"]].to_numpy().max() * 1.18)
  axes[0].set_ylabel("MAE (minutes)")
  axes[0].set_title("Test MAE by Core Airport Hub")
  axes[0].legend()
  axes[0].grid(axis="y", alpha=0.3)

  x_slk = np.arange(len(slack_df))
  axes[1].bar(x_slk - w / 2, slack_df["DepDelay_MAE"], width=w, label="DepDelay MAE", color="#2b5c8f")
  axes[1].bar(x_slk + w / 2, slack_df["ArrDelay_MAE"], width=w, label="ArrDelay MAE", color="#d97724")
  axes[1].set_xticks(x_slk)
  axes[1].set_xticklabels(["No Chain", "<45m", "45-90m", ">90m"])
  axes[1].set_ylim(0, slack_df[["DepDelay_MAE", "ArrDelay_MAE"]].to_numpy().max() * 1.18)
  axes[1].set_ylabel("MAE (minutes)")
  axes[1].set_title("Test MAE by Effective Turnaround Slack")
  axes[1].legend()
  axes[1].grid(axis="y", alpha=0.3)

  plt.tight_layout()
  plt.savefig(plots_dir / "error_analysis_hubs_slack.png", dpi=200)
  plt.close()
  print(f"\nSaved plots to {plots_dir / "error_analysis_hubs_slack.png"}")


if __name__ == "__main__":
  main()

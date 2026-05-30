# main.py — TR-GNN training entry point
# Converted from TR_GNN.ipynb
# wandb replaces SummaryWriter; diagonal_ratio logged every epoch.
#
# Run:  python main.py
# or:   uv run python main.py

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import json

import numpy as np
import torch
import wandb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from dataset_classes import ISO_NE
from debug_probes import diagonal_ratio
from helper_functions import build_correlation_priors, test_model, test_model_stepwise, train_model
from models_with_temporal_graph import TR_GNN_MultiScale

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HPARAMS: dict = {
    # Data
    "T_in": 72,           # input window  — 3 days of hourly data
    "T_out": 240,         # forecast horizon — 10 days
    # Model
    "d": 32,
    "hidden_dim": 64,
    "GCN_Layer": 5,
    "dropout_forecast": 0.1,
    "dropout_gcn": 0.2,
    "dropout_temporal": 0.2,
    "kernel_size": 7,
    "dilation": 3,
    # Training
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 100,
    "scheduler_patience": 3,
    # Adjacency fix flags — set to True to enable each fix
    "fix_no_identity": True,    # Fix 1: remove A = A + I
    "fix_no_lrelu": True,       # Fix 3: remove LeakyReLU before softmax
    "fix_soft_temperature": True,  # Fix 2: attention scale d**-0.25 (default d**-0.5)
    "use_correlation_prior": True,   # Fix 5: bias attention with feature |corr| (old-style structure)
    "corr_prior_strength": 1.0,
    "lambda_sparse": 0.0,       # Fix 4: 0 = disabled; re-enable only after graph is non-collapsed
    "lambda_smooth": 0.0,
    "lambda_entropy": 0.0,      # off: was flattening A toward uniform 1/N
}

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join("selected_data_ISONE.csv")
SAVE_DIR = "Runs"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = ISO_NE(
        csv_path=CSV_PATH,
        T_in=HPARAMS["T_in"],
        T_out=HPARAMS["T_out"],
        lag_hours=[1, 12, 24, 168],
        rolling_windows=[12, 24],
    )

    total_len = len(dataset.df_numeric)
    train_split_idx = int(0.6 * total_len)
    val_split_idx   = int(0.8 * total_len)

    print(f"Raw rows: {total_len}  |  train split: {train_split_idx}  |  val split: {val_split_idx}")

    # Fit scaler on training rows only
    scaler = StandardScaler()
    scaler.fit(dataset.df_numeric.iloc[:train_split_idx].values.astype(np.float32))

    print("Generating feature clusters and applying scaler...")
    dataset.apply_scaler(scaler)
    dataset.scaler = scaler

    # Sample index windows (non-overlapping train / val / test)
    T_in, T_out = HPARAMS["T_in"], HPARAMS["T_out"]
    effective_len = len(dataset)

    train_end  = min(train_split_idx - T_in - T_out, effective_len)
    val_start  = train_split_idx - T_in
    val_end    = min(val_split_idx  - T_in - T_out, effective_len)
    test_start = val_split_idx - T_in

    train_idx = range(0, train_end)
    val_idx   = range(val_start, val_end)
    test_idx  = range(test_start, effective_len)

    print(f"Samples — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    bs = HPARAMS["batch_size"]
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs, shuffle=False)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=bs, shuffle=False)
    print(f"Batches — train: {len(train_loader)}  val: {len(val_loader)}  test: {len(test_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    HPARAMS["N"] = dataset.N   # number of features / graph nodes

    corr_prior = None
    if HPARAMS.get("use_correlation_prior"):
        corr_prior, _ = build_correlation_priors(dataset.df_numeric)
        corr_prior = corr_prior.to(device)

    model = TR_GNN_MultiScale(
        N=HPARAMS["N"],
        T_in=HPARAMS["T_in"],
        T_out=HPARAMS["T_out"],
        d=HPARAMS["d"],
        hidden_dim=HPARAMS["hidden_dim"],
        GCN_Layer=HPARAMS["GCN_Layer"],
        dropout_gcn=HPARAMS["dropout_gcn"],
        dropout_temporal=HPARAMS["dropout_temporal"],
        kernel_size=HPARAMS["kernel_size"],
        dilation=HPARAMS["dilation"],
        fix_no_identity=HPARAMS["fix_no_identity"],
        fix_no_lrelu=HPARAMS["fix_no_lrelu"],
        fix_soft_temperature=HPARAMS["fix_soft_temperature"],
        corr_prior=corr_prior,
        corr_prior_strength=HPARAMS.get("corr_prior_strength", 2.5),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Run name & wandb ─────────────────────────────────────────────────────
    corr_tag = "corr" if HPARAMS.get("use_correlation_prior") else ""
    fix_digits = "".join(
        str(int(HPARAMS[k])) for k in ["fix_no_identity", "fix_no_lrelu", "fix_soft_temperature"]
    )
    run_name = (
        f"TR_GNN_ISO_NE"
        f"_GCN{HPARAMS['GCN_Layer']}"
        f"_H{HPARAMS['hidden_dim']}"
        f"_K{HPARAMS['kernel_size']}"
        f"_D{HPARAMS['dilation']}"
        f"_LR{HPARAMS['lr']}"
        f"{'_' + corr_tag if corr_tag else ''}"
        f"_fix{fix_digits}"
    )
    feature_names = list(dataset.df_numeric.columns)

    run = wandb.init(
        project="TR-GNN",
        name=run_name,
        config=HPARAMS,
        group="adjacency_fix",
    )
    print(f"wandb run: {run.name}  |  url: {run.url}")

    save_path = os.path.join(SAVE_DIR, f"{run_name}_best.pth")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Training ─────────────────────────────────────────────────────────────
    print("\nTraining TR-GNN MultiScale on ISO-NE...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=HPARAMS["epochs"],
        lr=HPARAMS["lr"],
        device=device,
        scheduler_patience=HPARAMS["scheduler_patience"],
        weight_decay=HPARAMS["weight_decay"],
        save_path=save_path,
        # wandb-aware args — train_model should call wandb.log() internally;
        # see helper_functions.py for the expected signature.
        wandb_run=run,
        lambda_sparse=HPARAMS["lambda_sparse"],
        lambda_smooth=HPARAMS["lambda_smooth"],
        lambda_entropy=HPARAMS["lambda_entropy"],
        feature_names=feature_names,
    )

    # ── Quick diagonal_ratio check on one validation batch ──────────────────
    print("\nRunning post-training diagonal_ratio check...")
    model.eval()
    X_probe, _ = next(iter(val_loader))
    X_probe = X_probe.to(device)
    with torch.no_grad():
        _, A_probe = model(X_probe)       # (B, N, N)
    ratio = diagonal_ratio(A_probe)
    print(f"  diagonal_ratio (val batch): {ratio:.2f}")
    print(f"  {'HEALTHY' if ratio < 5 else 'COLLAPSED — check fixes'}")
    wandb.log({"final_diagonal_ratio": ratio})

    # ── Testing ──────────────────────────────────────────────────────────────
    print("\nTesting model...")
    preds, trues = test_model(
        dataset=dataset,
        model=model,
        test_loader=test_loader,
        device=device,
    )

    wandb.finish()
    print(f"\nDone. Best model saved to: {save_path}")
    print(f"Run diagnosis: uv run python diagnose.py --checkpoint {save_path}")


if __name__ == "__main__":
    main()

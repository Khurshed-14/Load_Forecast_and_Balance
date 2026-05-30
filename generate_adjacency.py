#!/usr/bin/env python3
"""
generate_adjacency.py — captures A directly from TemporalGraphLearning.

Hooks model.graph_learn, runs validation batches, and saves heatmaps.
No wandb. No full-model output needed.

Usage:
    python generate_adjacency.py
    python generate_adjacency.py --checkpoint Runs/TR_GNN_ISO_NE_corr_fix111_best.pth
    python generate_adjacency.py --checkpoint Runs/best.pth --n-batches 50
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from dataset_classes import ISO_NE
from debug_probes import diagonal_ratio, plot_adjacency_heatmap
from helper_functions import build_correlation_priors
from models_with_temporal_graph import TR_GNN_MultiScale

CSV_PATH   = "selected_data_ISONE.csv"
T_IN, T_OUT = 72, 240
HIDDEN_DIM  = 64
GCN_LAYER   = 5
KERNEL_SIZE = 7
DILATION    = 3


def _parse_fix_flags(name: str) -> tuple[bool, bool, bool, bool]:
    """Infer fix flags + corr-prior from a checkpoint filename (_corr_fix111 convention)."""
    use_corr = "_corr" in name
    if "_fix" not in name:
        return False, False, False, use_corr
    digits = ""
    for ch in name.split("_fix", 1)[1]:
        if ch.isdigit():
            digits += ch
        else:
            break
    return (
        len(digits) > 0 and digits[0] == "1",   # fix_no_identity
        len(digits) > 1 and digits[1] == "1",   # fix_no_lrelu
        len(digits) > 2 and digits[2] == "1",   # fix_soft_temperature
        use_corr,
    )


def collect_adjacency(model: torch.nn.Module, loader: DataLoader,
                      n_batches: int, device: str) -> torch.Tensor:
    """Hook model.graph_learn and collect its output A across n_batches."""
    captured: list[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        captured.append(output.detach().cpu())

    handle = model.graph_learn.register_forward_hook(_hook)
    model.eval()
    try:
        with torch.no_grad():
            for i, (X, _) in enumerate(loader):
                if i >= n_batches:
                    break
                model(X.to(device))
    finally:
        handle.remove()

    A = torch.cat(captured, dim=0)   # (total_samples, N, N)
    print(f"  Collected {A.shape[0]} samples over {min(i + 1, n_batches)} batches.")
    return A


def print_stats(A: torch.Tensor, feature_names: list[str]) -> None:
    N = A.shape[-1]
    A_mean  = A.mean(0)
    dr      = diagonal_ratio(A)
    off     = A_mean[~torch.eye(N, dtype=torch.bool)]

    print(f"\n  diagonal_ratio : {dr:.3f}  "
          f"({'collapsed >10' if dr > 10 else 'borderline' if dr > 5 else 'selective' if dr > 1.2 else 'uniform ~1/N'})")
    print(f"  diag  — mean {A_mean.diagonal().mean():.4f}  std {A_mean.diagonal().std():.4f}")
    print(f"  off   — mean {off.mean():.4f}  std {off.std():.4f}")

    # Strongest off-diagonal edges
    A_np = A_mean.numpy().copy()
    np.fill_diagonal(A_np, -np.inf)
    top = np.argsort(A_np.flatten())[-5:][::-1]
    print("  Top off-diagonal edges:")
    for idx in top:
        r, c = divmod(int(idx), N)
        src = feature_names[r] if feature_names else str(r)
        tgt = feature_names[c] if feature_names else str(c)
        print(f"    {src}  →  {tgt}  :  {A_mean[r, c]:.4f}")


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-batches",  type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir",    default="adjacency_output")
    p.add_argument("--device",     default=None)
    p.add_argument("--no-annotate", action="store_true")
    # Manual flag overrides (auto-parsed from checkpoint name if omitted)
    p.add_argument("--fix-no-identity",     action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fix-no-lrelu",        action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fix-soft-temperature",action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use-correlation-prior",action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--corr-prior-strength", type=float, default=1.0)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve flags
    ckpt_name = os.path.basename(args.checkpoint or "")
    fix_no_id, fix_no_lr, fix_soft_t, use_corr = _parse_fix_flags(ckpt_name)
    if args.fix_no_identity       is not None: fix_no_id  = args.fix_no_identity
    if args.fix_no_lrelu          is not None: fix_no_lr  = args.fix_no_lrelu
    if args.fix_soft_temperature  is not None: fix_soft_t = args.fix_soft_temperature
    if args.use_correlation_prior is not None: use_corr   = args.use_correlation_prior

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    dataset = ISO_NE(csv_path=CSV_PATH, T_in=T_IN, T_out=T_OUT,
                     lag_hours=[1, 12, 24, 168], rolling_windows=[12, 24])
    total      = len(dataset.df_numeric)
    train_end  = int(0.6 * total)
    val_end    = int(0.8 * total)

    scaler = StandardScaler()
    scaler.fit(dataset.df_numeric.iloc[:train_end].values.astype(np.float32))
    dataset.apply_scaler(scaler)

    val_idx    = range(train_end - T_IN, min(val_end - T_IN - T_OUT, len(dataset)))
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    feature_names = list(dataset.df_numeric.columns)
    print(f"  N={dataset.N}  val samples={len(val_idx)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    corr_prior = None
    if use_corr:
        corr_prior, corr_display = build_correlation_priors(dataset.df_numeric)
        corr_prior = corr_prior.to(device)

    model = TR_GNN_MultiScale(
        N=dataset.N, T_in=T_IN, T_out=T_OUT,
        hidden_dim=HIDDEN_DIM, GCN_Layer=GCN_LAYER,
        kernel_size=KERNEL_SIZE, dilation=DILATION,
        fix_no_identity=fix_no_id, fix_no_lrelu=fix_no_lr,
        fix_soft_temperature=fix_soft_t,
        corr_prior=corr_prior, corr_prior_strength=args.corr_prior_strength,
    ).to(device)

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 1
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:    print(f"  Missing keys: {missing[:4]}")
        if unexpected: print(f"  Unexpected keys: {unexpected[:4]}")
        print(f"  Loaded: {args.checkpoint}")
    else:
        print("  No checkpoint — using random init.")

    # ── Collect A from graph_learn ────────────────────────────────────────────
    print(f"\n[3/4] Hooking graph_learn, running {args.n_batches} val batch(es)...")
    A = collect_adjacency(model, val_loader, args.n_batches, device)
    print_stats(A, feature_names)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving...")
    stem = ("adj_" + os.path.splitext(ckpt_name)[0]) if args.checkpoint else "adj_random_init"
    annotate = not args.no_annotate

    auto_path = os.path.join(args.out_dir, f"{stem}_auto.png")
    plot_adjacency_heatmap(A, feature_names, "Graph Learner A (auto scale)",
                           auto_path, color_scale="auto", annotate=annotate)
    print(f"  {auto_path}")

    rel_path = os.path.join(args.out_dir, f"{stem}_per_row.png")
    plot_adjacency_heatmap(A, feature_names, "Graph Learner A (row-relative)",
                           rel_path, color_scale="per_row", annotate=annotate)
    print(f"  {rel_path}")

    torch.save(A, os.path.join(args.out_dir, f"{stem}.pt"))
    np.save(os.path.join(args.out_dir, f"{stem}.npy"), A.numpy())
    print(f"  {stem}.pt / .npy")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
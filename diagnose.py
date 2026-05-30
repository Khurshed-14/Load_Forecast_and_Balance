#!/usr/bin/env python3
"""
Phase 1–2 adjacency collapse diagnosis (TR_GNN_DEBUGGING_PLAN.md).
Run before training fixes:  uv run python diagnose.py [--checkpoint path.pth]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from dataset_classes import ISO_NE
from debug_probes import (
    diagonal_ratio,
    graph_topology_metrics,
    plot_adjacency_heatmap,
    probe_gradients,
    probe_node_similarity,
    probe_raw_scores,
)
from helper_functions import build_correlation_priors
from models_with_temporal_graph import TR_GNN_MultiScale, TemporalGraphLearning

# Match main.py defaults
T_IN, T_OUT = 72, 240
HIDDEN_DIM, GCN_LAYER = 64, 5
KERNEL_SIZE, DILATION = 7, 3
CSV_PATH = "selected_data_ISONE.csv"
OUT_DIR = "diagnosis_output"


def load_val_batch(device: str, batch_size: int = 8):
    dataset = ISO_NE(
        csv_path=CSV_PATH,
        T_in=T_IN,
        T_out=T_OUT,
        lag_hours=[1, 12, 24, 168],
        rolling_windows=[12, 24],
    )
    total_len = len(dataset.df_numeric)
    train_split_idx = int(0.6 * total_len)
    val_split_idx = int(0.8 * total_len)
    scaler = StandardScaler()
    scaler.fit(dataset.df_numeric.iloc[:train_split_idx].values.astype("float32"))
    dataset.apply_scaler(scaler)
    val_start = train_split_idx - T_IN
    val_end = min(val_split_idx - T_IN - T_OUT, len(dataset))
    val_idx = range(val_start, val_end)
    loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    X, Y = next(iter(loader))
    feature_names = list(dataset.df_numeric.columns)
    return dataset.N, X.to(device), Y.to(device), feature_names, dataset


def inspect_graph_module_code() -> dict[str, bool]:
    import inspect

    init_src = inspect.getsource(TemporalGraphLearning.__init__)
    fwd_src = inspect.getsource(TemporalGraphLearning.forward)
    return {
        "has_A_plus_I_gated": "if self.add_identity" in fwd_src,
        "has_optional_lrelu": "if self.leaky_relu is not None" in fwd_src,
        "has_configurable_scale_exp": "scale_exp" in init_src,
        "has_corr_prior": "corr_prior" in init_src,
    }


def parse_fix_flags_from_checkpoint(path: str | None) -> tuple[bool, bool, bool, bool]:
    """Infer fix flags from run name suffix _fixXYZ (e.g. fix111 = all three on)."""
    if not path:
        return False, False, False, False
    base = os.path.basename(path)
    use_correlation_prior = "_corr" in base
    marker = "_fix"
    if marker not in base:
        return False, False, False, use_correlation_prior
    suffix = base.split(marker, 1)[1]
    digits = ""
    for ch in suffix:
        if ch.isdigit():
            digits += ch
        else:
            break
    if len(digits) < 2:
        return False, False, False, use_correlation_prior
    fix_no_identity = digits[0] == "1"
    fix_no_lrelu = digits[1] == "1"
    fix_soft_temperature = digits[2] == "1" if len(digits) >= 3 else False
    return fix_no_identity, fix_no_lrelu, fix_soft_temperature, use_correlation_prior


def interpret_diagonal_ratio(ratio: float) -> str:
    if ratio > 10:
        return "identity collapse (diag >> off-diag)"
    if ratio > 5:
        return "borderline diagonal dominance"
    if ratio > 1.2:
        return "selective structure (diag moderately > off-diag)"
    if ratio >= 0.85:
        return "near-uniform rows (diag ≈ off-diag; graph mixes all nodes equally)"
    return "off-diagonal-heavy (unusual for row-softmax)"


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="TR-GNN adjacency collapse diagnosis")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional .pth state_dict")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--fix-no-identity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Match main.py Fix 1 (no A+I). Default: parse from checkpoint _fix11 suffix.",
    )
    parser.add_argument(
        "--fix-no-lrelu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Match main.py Fix 3 (no LeakyReLU before softmax). Default: parse from checkpoint.",
    )
    parser.add_argument(
        "--fix-soft-temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Match main.py Fix 2 (scale d**-0.25). Default: parse from checkpoint _fix111 suffix.",
    )
    parser.add_argument(
        "--use-correlation-prior",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fix 5: bias graph with feature |corr| prior. Default: parse _corr from checkpoint name.",
    )
    parser.add_argument(
        "--corr-prior-strength",
        type=float,
        default=1.0,
        help="Logit bias strength for correlation prior (main.py default 1.0).",
    )
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUT_DIR, exist_ok=True)

    fix_no_identity, fix_no_lrelu, fix_soft_temperature, use_correlation_prior = (
        parse_fix_flags_from_checkpoint(args.checkpoint)
    )
    if args.fix_no_identity is not None:
        fix_no_identity = args.fix_no_identity
    if args.fix_no_lrelu is not None:
        fix_no_lrelu = args.fix_no_lrelu
    if args.fix_soft_temperature is not None:
        fix_soft_temperature = args.fix_soft_temperature
    if args.use_correlation_prior is not None:
        use_correlation_prior = args.use_correlation_prior

    print_section("0. Code inspection (TemporalGraphLearning)")
    flags = inspect_graph_module_code()
    for k, v in flags.items():
        status = "yes" if v else "no"
        print(f"  {k}: {status}")
    if flags["has_A_plus_I_gated"]:
        print("  → Fix 1/2/3 are flag-gated; pass matching flags when loading checkpoints (_fix111 = all on).")

    print_section("1. Data & model")
    scale_exp = -0.25 if fix_soft_temperature else -0.5
    print(
        f"  fix_no_identity={fix_no_identity}  fix_no_lrelu={fix_no_lrelu}  "
        f"fix_soft_temperature={fix_soft_temperature}  (scale_exp={scale_exp})  "
        f"use_correlation_prior={use_correlation_prior}"
    )
    N, X, Y, feature_names, dataset = load_val_batch(device)
    print(f"  Device: {device}  |  N={N}  |  batch X: {tuple(X.shape)}  Y: {tuple(Y.shape)}")

    corr_prior = None
    if use_correlation_prior:
        corr_prior, _ = build_correlation_priors(dataset.df_numeric)
        corr_prior = corr_prior.to(device)

    model = TR_GNN_MultiScale(
        N=N,
        T_in=T_IN,
        T_out=T_OUT,
        hidden_dim=HIDDEN_DIM,
        GCN_Layer=GCN_LAYER,
        kernel_size=KERNEL_SIZE,
        dilation=DILATION,
        fix_no_identity=fix_no_identity,
        fix_no_lrelu=fix_no_lrelu,
        fix_soft_temperature=fix_soft_temperature,
        corr_prior=corr_prior,
        corr_prior_strength=args.corr_prior_strength,
    ).to(device)

    label = "random_init"
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 1
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Note: missing keys (ok if adding corr prior): {missing[:5]}...")
        if unexpected:
            print(f"  Note: unexpected keys: {unexpected[:5]}...")
        label = os.path.basename(args.checkpoint)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print("  No checkpoint — reporting random-init forward (train collapse needs .pth).")

    model.eval()

    print_section("2. Probe 1 — raw attention scores")
    p1 = probe_raw_scores(model, X)
    s = p1["scores_before_relu"][0]
    Nn = s.shape[0]
    mask = ~torch.eye(Nn, dtype=torch.bool)
    off = s.flatten()[mask.flatten()]
    print(f"  Raw Q@K (batch 0): diag mean={s.diagonal().mean():.4f}  off-diag mean={off.mean():.4f}")
    print(f"                     diag std={s.diagonal().std():.4f}   off-diag std={off.std():.4f}")

    print_section("3. Probe 2 — TemporalConv embedding similarity (Cause A)")
    p2 = probe_node_similarity(model, X)
    sim = p2["cos_sim"]
    cross_sim = (sim.sum() - sim.diagonal().sum()) / (Nn * (Nn - 1))
    print(f"  Mean cross-node cosine sim: {cross_sim:.4f}")
    if cross_sim.item() > 0.85:
        print("  ❌ CAUSE A likely: embeddings too similar → Q@K dominated by self-similarity.")
    else:
        print("  ✅ Embeddings distinct enough; collapse likely inside graph learning.")

    print_section("4. Test B — identity + renorm (Cause B)")
    A_full = p1["A_model_output"]
    A_soft = p1["A_softmax_only"]
    A_id = p1["A_with_identity"]
    r_full = diagonal_ratio(A_full)
    r_soft = diagonal_ratio(A_soft)
    r_id = diagonal_ratio(A_id)
    print(f"  diagonal_ratio softmax only:     {r_soft:.2f}")
    print(f"  diagonal_ratio + I + renorm:     {r_id:.2f}")
    print(f"  diagonal_ratio model output:     {r_full:.2f}")
    if r_id > r_soft * 1.5:
        print("  ❌ CAUSE B: I+renorm amplifies diagonal dominance.")
    else:
        print("  → I+renorm is not the dominant amplifier at this stage.")

    print_section("5. Probe 4 — final A diagonal dominance")
    with torch.no_grad():
        _, A = model(X)
    ratio = diagonal_ratio(A)
    print(f"  diagonal_ratio: {ratio:.2f}  ({interpret_diagonal_ratio(ratio)})")

    heatmap_path = os.path.join(OUT_DIR, f"adjacency_{label}.png")
    plot_adjacency_heatmap(
        A,
        feature_names,
        "Learned Graph Adjacency Matrix (A)",
        heatmap_path,
        color_scale="auto",
    )
    print(f"  Saved heatmap (auto color scale): {heatmap_path}")

    rel_path = os.path.join(OUT_DIR, f"adjacency_{label}_relative.png")
    plot_adjacency_heatmap(
        A,
        feature_names,
        "Learned Graph Adjacency (row-relative, max=1 per source)",
        rel_path,
        color_scale="per_row",
    )
    print(f"  Saved row-relative heatmap: {rel_path}")

    if use_correlation_prior:
        _prior, _display = build_correlation_priors(dataset.df_numeric)
        prior_path = os.path.join(OUT_DIR, f"adjacency_{label}_correlation_prior.png")
        import numpy as np

        prior_t = torch.from_numpy(_display).unsqueeze(0).float()
        plot_adjacency_heatmap(
            prior_t,
            feature_names,
            "Feature correlation prior (display, diagonal=1)",
            prior_path,
            color_scale="unit",
        )
        print(f"  Saved correlation prior reference: {prior_path}")

    print_section("6. Test D — graph topology (off-diagonal edges)")
    try:
        metrics, _G = graph_topology_metrics(A)
        print(f"  {metrics}")
        if metrics["num_edges"] == 0:
            print("  → No off-diagonal edges above threshold (matches collapsed heatmap).")
    except ImportError as e:
        print(f"  Skipped: {e}")

    print_section("7. Probe 3 — gradient norms (graph_learn vs rest)")
    criterion = nn.MSELoss()
    grads = probe_gradients(model, X, Y, criterion)
    gl_grads = {k: v for k, v in grads.items() if k.startswith("graph_learn.") and v is not None}
    tc_grads = {k: v for k, v in grads.items() if k.startswith("temporal_conv.") and v is not None}
    if gl_grads:
        print(f"  graph_learn grad norm (sum): {sum(gl_grads.values()):.6f}")
        for k in sorted(gl_grads):
            print(f"    {k}: {gl_grads[k]:.6f}")
    if tc_grads:
        print(f"  temporal_conv grad norm (sum): {sum(tc_grads.values()):.6f}")

    print_section("Summary")
    print(f"  Model state: {label}")
    print(f"  diagonal_ratio = {ratio:.2f} — {interpret_diagonal_ratio(ratio)}")
    if ratio > 10 and fix_no_identity:
        print("  ⚠️  Ratio > 10 despite fix_no_identity=True — checkpoint may be from a different arch.")
    elif 0.85 <= ratio <= 1.2:
        print("  → Fixes removed I+renorm collapse; graph rows are ~uniform (~1/N per edge).")
        if not fix_soft_temperature:
            print("  → Next: Fix 2 (softer temperature, fix_soft_temperature) or correlation prior.")
        else:
            print("  → Fix 2 already on; try correlation prior (Fix 5) if ratio stays ~1.")
    elif 1.2 < ratio <= 4.0:
        print("  → Selective adjacency in target band; consider re-enabling sparse/entropy tuning carefully.")
    elif ratio > 10:
        print("  → Apply Fix 1 (+ Fix 3), retrain, and re-run with matching --fix-no-identity flags.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# debug_probes.py — Phase 1/2 diagnostics for TR-GNN adjacency collapse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    import networkx as nx

    _HAS_NX = True
except ImportError:
    _HAS_NX = False


def diagonal_ratio(A_tensor: torch.Tensor) -> float:
    """Mean(diag) / mean(off-diag). Healthy ≈ 1–5; collapsed >> 10."""
    B, N, _ = A_tensor.shape
    diag_mean = A_tensor.diagonal(dim1=-2, dim2=-1).mean().item()
    off_diag_sum = A_tensor.sum() - A_tensor.diagonal(dim1=-2, dim2=-1).sum()
    off_diag_mean = off_diag_sum / (B * N * (N - 1))
    return diag_mean / (off_diag_mean + 1e-9)


def _scores_from_graph_module(module, H: torch.Tensor) -> torch.Tensor:
    """Raw Q@K^T scaled scores, shape (B, N, N)."""
    Q = module.W_q(H)  # (B, N, d)
    K = module.W_k(H)  # (B, N, d)
    scale = module.scale
    if isinstance(scale, torch.nn.Parameter):
        scale = scale.item()
    return torch.matmul(Q, K.transpose(-2, -1)) * scale


def _adjacency_pipeline(module, scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reproduce TemporalGraphLearning stages for ablation."""
    out: dict[str, torch.Tensor] = {}
    if hasattr(module, "leaky_relu") and module.leaky_relu is not None:
        scores_lrelu = module.leaky_relu(scores)
    else:
        scores_lrelu = scores
    if hasattr(module, "_bias_scores_with_corr_prior"):
        scores_lrelu = module._bias_scores_with_corr_prior(scores_lrelu)
    out["scores_before_softmax"] = scores_lrelu
    A_soft = F.softmax(scores_lrelu, dim=-1)
    out["A_softmax_only"] = A_soft
    N = A_soft.size(-1)
    I = torch.eye(N, device=A_soft.device).unsqueeze(0)
    A_with_id = A_soft + I
    A_with_id = A_with_id / A_with_id.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    out["A_with_identity"] = A_with_id
    return out


def probe_raw_scores(model, X_batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture Q@K scores and adjacency variants via forward hook on graph_learn."""
    captured: dict[str, torch.Tensor] = {}
    gl = model.graph_learn

    def hook(_module, inputs, output):
        H = inputs[0]  # (B, N, d)
        scores = _scores_from_graph_module(gl, H)
        captured["scores_before_relu"] = scores.detach().cpu()
        variants = _adjacency_pipeline(gl, scores)
        for k, v in variants.items():
            captured[k] = v.detach().cpu()
        captured["A_model_output"] = output.detach().cpu()

    handle = gl.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(X_batch)
    finally:
        handle.remove()
    return captured


def probe_node_similarity(model, X_batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cosine similarity of TemporalConv embeddings H, first batch item."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        H = output.detach().cpu()  # (B, N, hidden_dim)
        h0 = H[0]
        h0_norm = F.normalize(h0, dim=-1)
        sim = torch.matmul(h0_norm, h0_norm.T)
        captured["H"] = H
        captured["cos_sim"] = sim

    handle = model.temporal_conv.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(X_batch)
    finally:
        handle.remove()
    return captured


def probe_gradients(
    model,
    X_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    criterion,
) -> dict[str, float | None]:
    """One backward step; L2 norm per parameter."""
    model.train()
    pred, _A = model(X_batch)
    loss = criterion(pred, Y_batch)
    loss.backward()
    grad_report: dict[str, float | None] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_report[name] = param.grad.norm().item()
        else:
            grad_report[name] = None
    model.zero_grad()
    return grad_report


def graph_topology_metrics(A_tensor: torch.Tensor, threshold: float = 0.01):
    """Average adjacency → directed graph metrics (off-diagonal edges only)."""
    if not _HAS_NX:
        raise ImportError("networkx required for graph_topology_metrics (uv add networkx)")
    A_mean = A_tensor.mean(0).cpu().numpy()
    A_thresh = (A_mean > threshold).astype(float)
    np.fill_diagonal(A_thresh, 0)
    G = nx.from_numpy_array(A_thresh, create_using=nx.DiGraph)
    metrics = {
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_weakly_connected": nx.is_weakly_connected(G) if G.number_of_nodes() else False,
        "avg_degree": sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1),
    }
    return metrics, G


def _resolve_heatmap_matrix(A_mean: np.ndarray, color_scale: str) -> tuple[np.ndarray, float, float, str]:
    """
    color_scale:
      auto    — vmin=0, vmax=data max (row-softmax weights are ~1/N; do not use 0–1 fixed)
      unit    — force 0–1 axis (only for matrices already in [0, 1], e.g. correlation display)
      per_row — each row divided by its max (relative influence; legacy-style contrast)
    """
    if color_scale == "per_row":
        row_max = A_mean.max(axis=1, keepdims=True)
        viz = A_mean / np.maximum(row_max, 1e-8)
        return viz, 0.0, 1.0, "relative row max (=1)"
    if color_scale == "unit":
        return A_mean, 0.0, 1.0, "weight"
    # auto
    hi = float(np.max(A_mean))
    hi = max(hi, 1e-6)
    return A_mean, 0.0, hi, "attention weight (row-stochastic)"


def plot_adjacency_heatmap(
    A_tensor: torch.Tensor,
    feature_names: list[str] | None,
    title: str,
    save_path: str,
    n_samples_avg: int = 10,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    color_scale: str = "auto",
) -> None:
    """Save mean adjacency heatmap. Default color_scale='auto' matches data range (~1/N)."""
    n = min(A_tensor.size(0), n_samples_avg)
    A_mean = A_tensor[:n].mean(0).cpu().numpy()
    nn = A_mean.shape[0]
    viz, lo, hi, cbar_label = _resolve_heatmap_matrix(A_mean, color_scale)
    if vmin is not None:
        lo = vmin
    if vmax is not None:
        hi = vmax
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(viz, cmap="viridis", aspect="auto", vmin=lo, vmax=hi)
    plt.colorbar(im, ax=ax, label=cbar_label)
    if feature_names and len(feature_names) == nn:
        ax.set_xticks(range(nn))
        ax.set_yticks(range(nn))
        ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
        ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Target Nodes (Receivers)")
    ax.set_ylabel("Source Nodes (Influencers)")
    if not title:
        title = "Learned Graph Adjacency Matrix (A)"
    ax.set_title(title)
    if annotate:
        mid = 0.5 * (lo + hi)
        for i in range(nn):
            for j in range(nn):
                val = viz[i, j]
                raw = A_mean[i, j]
                label = f"{raw:.2f}" if color_scale != "per_row" else f"{val:.2f}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if val < mid else "black",
                )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

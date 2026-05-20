from tqdm import tqdm
import torch
import torch.nn as nn
import os
import numpy as np
import time  # <--- Added import for timing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad(), tqdm(loader, desc="Validating", leave=False) as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            pred, _ = model(X)      # <-- unpack tuple, discard A
            loss = criterion(pred, Y)
            total_loss += loss.item()
    return total_loss / len(loader)

# Training function with TensorBoard logging
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-4,
    device='cuda',
    patience=10,
    scheduler_patience=4,
    scheduler_factor=0.5,
    save_path="ISO_NE_Small_Dataset_Run2",
    writer=None,
    weight_decay=1e-5,
    lambda_smooth=0.01,     # <-- L2 temporal consistency weight
    lambda_sparse=1e-4,     # <-- L1 sparsity weight (optional, kept from before)
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-6
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_history = []
    val_history = []
    start_total_time = time.time()
    best_model_time = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for X, Y in loop:
            X, Y = X.to(device), Y.to(device)

            pred, A = model(X)          # <-- unpack (Y_hat, A)

            # 1. Task loss
            mse_loss = criterion(pred, Y)

            # 2. Temporal consistency: L2 penalty between adjacent time window graphs
            #    A: (B, N, N) — consecutive samples = consecutive time windows
            #    Encourages smooth graph evolution without freezing it
            smooth_loss = nn.functional.mse_loss(A[:-1], A[1:].detach())

            # 3. Sparsity: L1 penalty to keep graph sparse (optional but useful)
            sparse_loss = torch.norm(A, p=1)

            loss = mse_loss + lambda_smooth * smooth_loss + lambda_sparse * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(
                mse=f"{mse_loss.item():.4f}",
                smooth=f"{smooth_loss.item():.4f}"  # visible in tqdm so you can monitor it
            )

        avg_train_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion, device)

        train_history.append(avg_train_loss)
        val_history.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            best_model_time = time.time() - start_total_time
            print(f"✅ New best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\n⛔ Early stopping triggered after {patience} epochs without improvement.")
            break

    total_duration = time.time() - start_total_time

    def format_duration(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{int(h)}h {int(m)}m {int(s)}s"
        return f"{int(m)}m {int(s)}s"

    print("\n" + "="*50)
    print("               TIMING REPORT               ")
    print("="*50)
    print(f"⏱️  Time to reach Best Model: {format_duration(best_model_time)}")
    print(f"⏱️  Total Training Duration:  {format_duration(total_duration)}")
    print("="*50 + "\n")

    print(f"Loading best model from {save_path} (Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(torch.load(save_path, map_location=device))

    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(save_path + "_learning_curve.png", dpi=200)
    plt.close()

    print("Training complete. TensorBoard logs saved.")
    return model


# Testing function with TensorBoard logging
def test_model(dataset, model, test_loader, device='cuda', writer=None):
    model.eval()
    preds, trues = [], []

    with torch.no_grad(), tqdm(test_loader, desc="Testing") as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            out, _ = model(X)           # <-- unpack tuple, discard A
            preds.append(out.cpu().numpy())
            trues.append(Y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    print(f"\nTest Results:\nMSE = {mse:.4f} | MAE = {mae:.4f} | R² = {r2:.4f}\n")

    if writer:
        writer.add_scalar('Test_Metrics/MSE', mse, 1)
        writer.add_scalar('Test_Metrics/MAE', mae, 1)
        writer.add_scalar('Test_Metrics/R2', r2, 1)
        print("Test metrics logged to TensorBoard.")

    return preds, trues

def get_cluster_prior(dataset, n_clusters=5):
    """
    Computes a cluster mask based on feature correlations.
    Returns a tensor (N, N) where A_ij = 1 if i and j are in the same cluster.
    """
    # 1. Get raw data from the dataset
    # We use the unscaled data to capture true correlations
    df = dataset.df_numeric
    
    # 2. Compute Correlation Matrix (N x N)
    corr_matrix = df.corr().fillna(0).values
    
    # 3. Perform Clustering (e.g., K-Means on the correlation features)
    # This groups features that behave similarly
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(corr_matrix)
    
    # 4. Create the Prior Adjacency Matrix
    N = len(labels)
    prior_adj = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                prior_adj[i, j] = 1.0  # Same cluster connection
            else:
                prior_adj[i, j] = 0.0  # Different cluster (weak connection)
                
    # Normalize or scale if needed, but 0/1 is fine for a bias
    return torch.FloatTensor(prior_adj)
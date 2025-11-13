from tqdm import tqdm
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Validation helper function
def validate(model, loader, criterion, device):
    """Helper function to validate the model."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad(), tqdm(loader, desc="Validating", leave=False) as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
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
    save_path="ISO_NE_Small_Dataset_Run2",  # Define log directory for TensorBoard
    writer=None,
    weight_decay=1e-5,
):
    """
    Trains the model with:
    - tqdm progress bars
    - early stopping
    - LR scheduler
    - best-model saving
    - TensorBoard logging 
    """

    # --- Setup ---
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
        min_lr=1e-6
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for X, Y in loop:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = criterion(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add this
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(batch_loss=loss.item())

        # --- Validation and Reporting ---
        avg_train_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log scalars to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        # --- Early Stopping and Model Saving ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\n⛔ Early stopping triggered after {patience} epochs without improvement.")
            break

    # --- Cleanup ---
    print(f"\nLoading best model from {save_path} (Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    print("Training complete. TensorBoard logs saved.")

    return model


# Testing function with TensorBoard logging
def test_model(dataset, model, test_loader, device='cuda', writer=None):
    """
    Evaluates the model on the test set and logs metrics to TensorBoard.
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad(), tqdm(test_loader, desc="Testing") as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            out = model(X)
            preds.append(out.cpu().numpy())
            trues.append(Y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # --- Calculate Metrics ---
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    

    print(f"\nTest Results:\nMSE = {mse:.4f} | MAE = {mae:.4f} | R² = {r2:.4f}\n")

    # --- TensorBoard Logging ---
    # Log metrics to TensorBoard if a writer is provided
    if writer:
        # Re-open the writer to append to the existing log directory
        
        # Log final test metrics. Using step 1 as it's a single event.
        writer.add_scalar('Test_Metrics/MSE', mse, 1)
        writer.add_scalar('Test_Metrics/MAE', mae, 1)
        writer.add_scalar('Test_Metrics/R2', r2, 1)
        
        # You can also use add_hparams to log hyperparameters and metrics together
        # Example (assuming you pass hparams as a dict):
        # hparams = {'lr': 1e-4, 'batch_size': 32} # Example hparams
        # metrics = {'hparam/Test_MSE': mse, 'hparam/Test_MAE': mae, 'hparam/Test_R2': r2}
        # writer.add_hparams(hparams, metrics)
        
        print(f"Test metrics logged to TensorBoard.")

    return preds, trues
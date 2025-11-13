import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
from torch.utils.tensorboard import SummaryWriter 
import json

class GraphLearning(nn.Module):
    def __init__(self, N, d):  # ✅ CORRECT - accept N and d
        super().__init__()
        self.v = nn.Parameter(torch.randn(N, d))
        self.W = nn.Parameter(torch.randn(d, d))

    def forward(self):
        sim = torch.sigmoid(self.v @ self.W @ self.v.T)
        A = sim - torch.diag_embed(torch.diagonal(sim))
        # Add normalization for stability
        A = F.softmax(A, dim=1)
        return A
    
    
class TemporalConv(nn.Module):
    def __init__(self, N, T_in, hidden_dim=64, kernel_size=7, dilation=3, dropout=0.2, layers=3):
        super().__init__()
        
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(layers)
        ])
        self.dilation = dilation
        
        # Calculate padding to maintain temporal length
        # For causal conv: padding = (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation
        
        self.convs = nn.ModuleList([
            nn.Conv1d(1, hidden_dim, kernel_size, dilation=dilation, padding=padding)
            for _ in range(N)
        ])
        
    def forward(self, x):  # x: (batch, N, T_in)
        outs = []
        for i, conv in enumerate(self.convs):
            h = conv(x[:, i:i+1, :])          # (B, hidden_dim, T_padded)
            h = h[..., :x.size(2)]            # Trim to original length (causal)
            h = F.relu(h)
            h = self.dropout(h)
            h = torch.mean(h, dim=2)          # Temporal pooling → (B, hidden_dim)
            outs.append(h)
        return torch.stack(outs, dim=1)       # (B, N, hidden_dim)
    

class DenselyResidualGCN(nn.Module):
    """
    Graph convolution with dense residual links + dropout between layers.
    """
    def __init__(self, in_dim, hidden_dim, layers=5, dropout=0.3):
        super().__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.gcn_layers = nn.ModuleList([
            nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(layers)
        ])

    def forward(self, X, A):  # X:(B,N,D_in), A:(N,N)
        H_prev = X
        H_all = [H_prev]
        for l in range(self.layers):
            agg = torch.matmul(A, H_prev)
            H_cur = F.relu(self.gcn_layers[l](agg))
            H_cur = self.dropout(H_cur)       # dropout per node representation
            # dense residual connection
            H_prev = H_cur + torch.sum(torch.stack(H_all), dim=0)
            H_all.append(H_prev)
        return H_prev  # (B,N,hidden_dim)


class LoadForecasting(nn.Module):
    """
    Final prediction head; applied dropout before first linear layer.
    """
    def __init__(self, d_model, T_out, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, X):
        X = torch.mean(X, dim=1)  # mean across nodes (B, d_model)
        return self.net(X)

class GraphAwareGRUForecasting(nn.Module):
    """
    Graph-aware GRU decoder for multi-step load forecasting.
    Combines graph context with temporal recurrence.
    """
    def __init__(self, d_model, T_out, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.T_out = T_out

        # Spatial mixing linear transformation
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # GRU decoder (temporal dimension)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Final projection from hidden state → predicted load
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, X, A):
        """
        X: (B, N, d_model)  - Node embeddings from GCN
        A: (N, N)           - Learned adjacency
        Returns:
            Y_hat: (B, T_out)
        """

        # --- 1. Graph aggregation ---
        graph_context = torch.matmul(A, X)             # (B, N, d_model)
        graph_context = F.relu(self.Wg(graph_context)) # optional nonlinearity
        graph_context = torch.mean(graph_context, dim=1, keepdim=True)  # (B, 1, d_model)

        # --- 2. Temporal decoding ---
        # Create a dummy sequence of length = T_out filled by graph representation
        decoder_input = graph_context.repeat(1, self.T_out, 1)  # (B, T_out, d_model)

        out_seq, _ = self.gru(decoder_input)           # (B, T_out, hidden_dim)
        out_seq = self.dropout(out_seq)

        # --- 3. Output projection ---
        Y_hat = self.fc_out(out_seq).squeeze(-1)       # (B, T_out)
        return Y_hat

class MultiScaleForecasting(nn.Module):
    def __init__(self, d_model, T_out, dropout=0.2):
        super().__init__()
        self.short = nn.Linear(d_model, T_out)                      # Direct path
        self.mid = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, T_out)
        )
        self.long = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, T_out)
        )
        self.alpha = nn.Parameter(torch.tensor(0.3))  # learnable weighted sum

    def forward(self, X):
        X = torch.mean(X, dim=1)                         # (B, d_model)
        y_short = self.short(X)
        y_mid = self.mid(X)
        y_long = self.long(X)
        return self.alpha * y_mid + (1 - self.alpha) * (y_short + y_long) / 2

class AttentionForecasting(nn.Module):
    def __init__(self, d_model, T_out, dropout=0.3):
        super().__init__()
        self.query = nn.Linear(d_model, 1)
        self.value = nn.Linear(d_model, 64)
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, X):
        attn = F.softmax(self.query(X), dim=1)          # (B, N, 1)
        context = torch.sum(attn * self.value(X), dim=1)  # weighted sum (B, 64)
        Y_hat = self.proj(context)
        return Y_hat
    
class GlobalLocalForecasting(nn.Module):
    def __init__(self, d_model, T_out, dropout=0.2):
        super().__init__()
        self.global_fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, T_out)
        )
        self.local_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, X):
        g_feat = torch.mean(X, dim=1)  # global trend
        l_feat = X - g_feat.unsqueeze(1)  # residual deviations
        global_pred = self.global_fc(g_feat)
        local_pred = torch.mean(self.local_fc(l_feat), dim=1)
        return global_pred + local_pred
    
class GLFN_TC_Linear(nn.Module):
    """
    Graph Learning + Temporal Convolution + Dense Residual GCN + Dropout regularization.
    """
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3, GCN_Layer=5):
        super().__init__()
        self.graph_learn = GraphLearning(N, d)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          dilation=3, dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = LoadForecasting(
            d_model=hidden_dim,
            T_out=T_out,
            dropout=dropout_forecast,
)

    def forward(self, X):  # X: (B, N, T_in)
        A = self.graph_learn()
        H = self.temporal_conv(X)
        H = self.dense_gcn(H, A)
        Y_hat = self.forecaster(H)
        return Y_hat
    
class GLFN_TC_GraphGRU(nn.Module):
    """
    Graph Learning + Temporal Convolution + Dense Residual GCN + Dropout regularization.
    """
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3, GCN_Layer=5):
        super().__init__()
        self.graph_learn = GraphLearning(N, d)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          dilation=3, dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = GraphAwareGRUForecasting(
            d_model=hidden_dim,
            T_out=T_out,
            dropout=dropout_forecast,
)

    def forward(self, X):  # X: (B, N, T_in)
        A = self.graph_learn()
        H = self.temporal_conv(X)
        H = self.dense_gcn(H, A)
        Y_hat = self.forecaster(H, A)
        return Y_hat
    
class GLFN_TC_MultiScale(nn.Module):
    """
    Graph Learning + Temporal Convolution + Dense Residual GCN + Dropout regularization.
    """
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3, GCN_Layer=5):
        super().__init__()
        self.graph_learn = GraphLearning(N, d)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          dilation=3, dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = MultiScaleForecasting(
            d_model=hidden_dim,
            T_out=T_out,
            dropout=dropout_forecast,
)

    def forward(self, X):  # X: (B, N, T_in)
        A = self.graph_learn()
        H = self.temporal_conv(X)
        H = self.dense_gcn(H, A)
        Y_hat = self.forecaster(H)
        return Y_hat
    
class GLFN_TC_Attention(nn.Module):
    """
    Graph Learning + Temporal Convolution + Dense Residual GCN + Dropout regularization.
    """
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3, GCN_Layer=5):
        super().__init__()
        self.graph_learn = GraphLearning(N, d)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          dilation=3, dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = AttentionForecasting(
            d_model=hidden_dim,
            T_out=T_out,
            dropout=dropout_forecast,
)

    def forward(self, X):  # X: (B, N, T_in)
        A = self.graph_learn()
        H = self.temporal_conv(X)
        H = self.dense_gcn(H, A)
        Y_hat = self.forecaster(H)
        return Y_hat

class GLFN_TC_GlobalLocal(nn.Module):
    """
    Graph Learning + Temporal Convolution + Dense Residual GCN + Dropout regularization.
    """
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3, GCN_Layer=5):
        super().__init__()
        self.graph_learn = GraphLearning(N, d)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          dilation=3, dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = GlobalLocalForecasting(
            d_model=hidden_dim,
            T_out=T_out,
            dropout=dropout_forecast,
)

    def forward(self, X):  # X: (B, N, T_in)
        A = self.graph_learn()
        H = self.temporal_conv(X)
        H = self.dense_gcn(H, A)
        Y_hat = self.forecaster(H)
        return Y_hat
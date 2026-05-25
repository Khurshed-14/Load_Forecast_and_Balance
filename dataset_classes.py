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

class ISO_NE(Dataset):
    def __init__(
        self,
        csv_path,
        T_in=72,
        T_out=240,
        use_cyclic_time=True,
        lag_hours=[1, 24],
        rolling_windows=[24] ## Added this line for rolling average features
    ):
        df = pd.read_csv(csv_path)
        df = df.dropna().reset_index(drop=True)

        # --- Parse datetime ---
        if "date" not in df.columns:
            raise ValueError("Dataset must contain 'date' column.")
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

        # --- Optional cyclic encodings ---
        if use_cyclic_time:
            days_in_month = df["date"].dt.days_in_month
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            df["day_sin"] = np.sin(2 * np.pi * df["day"] / days_in_month)
            df["day_cos"] = np.cos(2 * np.pi * df["day"] / days_in_month)
            df["weekday_sin"] = np.sin(2 * np.pi * df["date"].dt.weekday / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["date"].dt.weekday / 7)
            
        # Drop the original time columns if cyclic encodings were used
        df.drop(columns=["hour","month", "day", "weekday", "year"], inplace=True, errors='ignore')
        
        df['EMA_12'] = df['demand'].ewm(span=12, adjust=False).mean()
        df['EMA_24'] = df['demand'].ewm(span=24, adjust=False).mean()
        df['diff_1'] = df['demand'].diff().fillna(0)
    

        # --- Target column ---
        target_col = "demand"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")

        # --- Create lag features ---
        for lag in lag_hours:
            df[f"Lag_{lag}h"] = df[target_col].shift(lag)
            
        ## ADDED THIS SECTION FOR ROLLING AVERAGE FEATURES ##
        # --- Create rolling average features ---  <-- ADD THIS SECTION
        for window in rolling_windows:
            # Rolling average of the target
            df[f"Demand_Roll_Avg_{window}h"] = df[target_col].rolling(window=window, min_periods=1).mean().shift(1)
                
        df = df.dropna().reset_index(drop=True)
        
        self.timestamps = df["date"].reset_index(drop=True)

        # Keep only numeric data (scaling applied later)
        df_num = df.select_dtypes(include=[np.number])
        
        cols = list(df_num.columns)
        if target_col in cols:
            cols.remove(target_col) # Remove target from current spot
        else:
            raise ValueError(f"Target '{target_col}' is not numeric or missing!")
            
        # Reconstruct list: [Target, Feature1, Feature2, ...]
        new_order = [target_col] + cols 
        
        # Reindex DataFrame
        df_num = df_num[new_order]
        
        self.df_numeric = df_num  # store unscaled numeric values
        self.data = None          # placeholder for scaled tensor (set later)
        
        self.T_in = T_in
        self.T_out = T_out
        self.N = df_num.shape[1]
        self.target_idx = list(df_num.columns).index(target_col)
        self.feature_names = df_num.columns.tolist()
        self.csv_path = csv_path

        print(
            f"Loaded dataset with {self.N} features "
            f"(target={target_col}), total rows={len(df_num)}"
        )

    def apply_scaler(self, scaler: StandardScaler):
        """Apply fitted StandardScaler to numeric data and create tensor."""
        scaled = scaler.transform(self.df_numeric.values.astype(np.float32))
        self.data = torch.tensor(scaled)
        return self

    def __len__(self):
        return len(self.data) - self.T_in - self.T_out

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.T_in]  # (T_in, N)
        y = self.data[idx + self.T_in : idx + self.T_in + self.T_out, self.target_idx]
        return x.T, y
    
    def preview(self, n=5):
        """Preview first n samples from the dataset."""
        for i in range(n):
            x, y = self[i]
            print(f"Sample {i}:")
            print(f"  Input (x) shape: {x.shape}")
            print(f"  Target (y) shape: {y.shape}\n")
            
class AT(Dataset):
    def __init__(
        self,
        csv_path,
        T_in=72,
        T_out=240,
        use_cyclic_time=True,
        lag_hours=[1, 24],
        rolling_windows=[24] ## Added this line for rolling average features
    ):
        df = pd.read_csv(csv_path)
        df = df.dropna().reset_index(drop=True)

        # --- Parse datetime ---
        if "utc_timestamp" not in df.columns:
            raise ValueError("Dataset must contain 'date' column.")
        df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], format="%Y-%m-%dT%H:%M:%SZ")
        df.rename(columns={"AT_load_actual_entsoe_power_statistics": "demand"}, inplace=True)


        # --- Optional cyclic encodings ---
        if use_cyclic_time:
            df["hour"] = df["utc_timestamp"].dt.hour
            df["month"] = df["utc_timestamp"].dt.month
            df["day"] = df["utc_timestamp"].dt.day
            df["year"] = df["utc_timestamp"].dt.year
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

            days_in_month = df["utc_timestamp"].dt.days_in_month
            df["day_sin"] = np.sin(2 * np.pi * df["day"] / days_in_month)
            df["day_cos"] = np.cos(2 * np.pi * df["day"] / days_in_month)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            df["weekday_sin"] = np.sin(2 * np.pi * df["week"] / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["week"] / 7)
            
            # Drop the original time columns if cyclic encodings were used
            df.drop(columns=["mean", "std", "month", "day", "hour", "week", "year"], inplace=True)
            
        df['EMA_12'] = df['demand'].ewm(span=12, adjust=False).mean()
        df['EMA_24'] = df['demand'].ewm(span=24, adjust=False).mean()
        df['diff_1'] = df['demand'].diff().fillna(0)
        
        # --- Target column ---
        target_col = "demand"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")

        # --- Create lag features ---
        for lag in lag_hours:
            df[f"Lag_{lag}h"] = df[target_col].shift(lag)
            
        ## ADDED THIS SECTION FOR ROLLING AVERAGE FEATURES ##
        # --- Create rolling average features ---  <-- ADD THIS SECTION
        for window in rolling_windows:
            # Rolling average of the target
            df[f"Demand_Roll_Avg_{window}h"] = df[target_col].rolling(window=window, min_periods=1).mean().shift(1)
                
        df = df.dropna().reset_index(drop=True)
        
        self.timestamps = df["utc_timestamp"].reset_index(drop=True)

        # Keep only numeric data (scaling applied later)
        df_num = df.select_dtypes(include=[np.number])
        
        cols = list(df_num.columns)
        if target_col in cols:
            cols.remove(target_col) # Remove target from current spot
        else:
            raise ValueError(f"Target '{target_col}' is not numeric or missing!")
            
        # Reconstruct list: [Target, Feature1, Feature2, ...]
        new_order = [target_col] + cols 
        
        # Reindex DataFrame
        df_num = df_num[new_order]
        
        self.df_numeric = df_num  # store unscaled numeric values
        self.data = None          # placeholder for scaled tensor (set later)
        
        self.T_in = T_in
        self.T_out = T_out
        self.N = df_num.shape[1]
        self.target_idx = list(df_num.columns).index(target_col)
        self.feature_names = df_num.columns.tolist()
        self.csv_path = csv_path

        print(
            f"Loaded dataset with {self.N} features "
            f"(target={target_col}), total rows={len(df_num)}"
        )

    def apply_scaler(self, scaler: StandardScaler):
        """Apply fitted StandardScaler to numeric data and create tensor."""
        scaled = scaler.transform(self.df_numeric.values.astype(np.float32))
        self.data = torch.tensor(scaled)
        return self

    def __len__(self):
        return len(self.data) - self.T_in - self.T_out

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.T_in]  # (T_in, N)
        y = self.data[idx + self.T_in : idx + self.T_in + self.T_out, self.target_idx]
        return x.T, y
    
    def preview(self, n=5):
        """Preview first n samples from the dataset."""
        for i in range(n):
            x, y = self[i]
            print(f"Sample {i}:")
            print(f"  Input (x) shape: {x.shape}")
            print(f"  Target (y) shape: {y.shape}\n")
            
class SH_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        T_in=72,
        T_out=240,
        use_cyclic_time=True,
        lag_hours=[1, 24],
        rolling_windows=[24]
    ):
        df = pd.read_csv(csv_path)
        df = df.dropna().reset_index(drop=True)
 
        # --- Parse datetime ---
        if "time" not in df.columns:
            raise ValueError("Dataset must contain 'time' column.")
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
 
        # --- Drop redundant columns ---
        # tem:          0.99 corr with tembody; tembody has higher corr with load (0.41 vs 0.38)
        # day_of_year:  fully derivable from month + day
        # week_of_year: fully derivable from month + day
        drop_redundant = ["tem", "day_of_year", "week_of_year"]
        df.drop(columns=[c for c in drop_redundant if c in df.columns], inplace=True)
 
        # --- Cyclic encode wind direction ---
        # dir (1–9) is a compass direction — circular, not ordinal.
        # dir=1 and dir=9 are adjacent; plain integer encoding treats them as far apart.
        if "dir" in df.columns:
            df["dir_sin"] = np.sin(2 * np.pi * df["dir"] / 9)
            df["dir_cos"] = np.cos(2 * np.pi * df["dir"] / 9)
            df.drop(columns=["dir"], inplace=True)
 
        # --- Optional cyclic time encodings ---
        if use_cyclic_time:
            days_in_month = df["time"].dt.days_in_month
            df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
            df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            df["day_sin"]   = np.sin(2 * np.pi * df["day"]   / days_in_month)
            df["day_cos"]   = np.cos(2 * np.pi * df["day"]   / days_in_month)
            df.drop(columns=["hour", "month", "day", "year"], inplace=True, errors="ignore")
 
        # --- Target column ---
        target_col = "load"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")
 
        df["EMA_12"]  = df["load"].ewm(span=12, adjust=False).mean()
        df["EMA_24"]  = df["load"].ewm(span=24, adjust=False).mean()
        df["diff_1"]  = df["load"].diff().fillna(0)
 
        # --- Lag features ---
        for lag in lag_hours:
            df[f"Lag_{lag}h"] = df[target_col].shift(lag)
 
        # --- Rolling average features ---
        for window in rolling_windows:
            df[f"Demand_Roll_Avg_{window}h"] = (
                df[target_col].rolling(window=window, min_periods=1).mean().shift(1)
            )
 
        df = df.dropna().reset_index(drop=True)
 
        self.timestamps = df["time"].reset_index(drop=True)
 
        # Keep only numeric data (scaling applied later)
        df_num = df.select_dtypes(include=[np.number])
 
        cols = list(df_num.columns)
        if target_col in cols:
            cols.remove(target_col)
        else:
            raise ValueError(f"Target '{target_col}' is not numeric or missing!")
 
        # Reconstruct: [Target, Feature1, Feature2, ...]
        new_order = [target_col] + cols
        df_num = df_num[new_order]
 
        self.df_numeric    = df_num
        self.data          = None
        self.T_in          = T_in
        self.T_out         = T_out
        self.N             = df_num.shape[1]
        self.target_idx    = list(df_num.columns).index(target_col)
        self.feature_names = df_num.columns.tolist()
        self.csv_path      = csv_path
 
        print(
            f"Loaded dataset with {self.N} features "
            f"(target={target_col}), total rows={len(df_num)}"
        )
 
    def apply_scaler(self, scaler: StandardScaler):
        """Apply fitted StandardScaler to numeric data and create tensor."""
        scaled = scaler.transform(self.df_numeric.values.astype(np.float32))
        self.data = torch.tensor(scaled)
        return self
 
    def __len__(self):
        return len(self.data) - self.T_in - self.T_out
 
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.T_in]
        y = self.data[idx + self.T_in : idx + self.T_in + self.T_out, self.target_idx]
        return x.T, y
 
    def preview(self, n=5):
        """Preview first n samples from the dataset."""
        for i in range(n):
            x, y = self[i]
            print(f"Sample {i}:")
            print(f"  Input (x) shape: {x.shape}")
            print(f"  Target (y) shape: {y.shape}\n")
 
            

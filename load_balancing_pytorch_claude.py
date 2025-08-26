import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandapower as pp
import pandapower.networks as pn
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PowerSystemDataset(Dataset):
    """Custom Dataset for power system load balancing data"""
    def __init__(self, X, y_actions, y_balance):
        self.X = torch.FloatTensor(X)
        self.y_actions = torch.FloatTensor(y_actions)
        self.y_balance = torch.FloatTensor(y_balance)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_actions[idx], self.y_balance[idx]

class LoadBalancingLSTM(nn.Module):
    """PyTorch LSTM model for load balancing with dual outputs"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, action_dim=10, dropout=0.3):
        super(LoadBalancingLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Shared dense layers
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Balance classification head
        self.balance_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Balancing actions regression head
        self.action_regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Shared layer
        shared_features = self.shared_fc(last_output)
        
        # Dual outputs
        balance_output = self.balance_classifier(shared_features)
        action_output = self.action_regressor(shared_features)
        
        return balance_output, action_output

class PyTorchLoadBalancer:
    def __init__(self, system_type='IEEE39'):
        self.system_type = system_type
        self.net = None
        self.scaler_state = StandardScaler()
        self.scaler_actions = StandardScaler()
        self.model = None
        self.sequence_length = 12
        self.imbalance_threshold = 5.0
        self.device = device
        
    def load_power_system(self, system_data_path=None):
        """Load power system network"""
        if self.system_type == 'IEEE39':
            print("Loading IEEE 39 Bus System for Load Balancing...")
            self.net = pn.case39()
        elif self.system_type == 'WECC179':
            print("Loading WECC 179 Bus System for Load Balancing...")
            self.net = self._create_wecc179_synthetic()
        
        # Extract system parameters
        self.n_buses = len(self.net.bus)
        self.n_generators = len(self.net.gen)
        self.n_loads = len(self.net.load)
        
        # Get generator limits
        self.gen_limits = {
            'pmax': self.net.gen['max_p_mw'].values,
            'pmin': self.net.gen['min_p_mw'].values,
            'qmax': self.net.gen['max_q_mvar'].values,
            'qmin': self.net.gen['min_q_mvar'].values
        }
        
        print(f"System loaded: {self.n_buses} buses, {self.n_generators} generators")
        return self.net
    
    def _create_wecc179_synthetic(self):
        """Create synthetic WECC179 for demonstration"""
        net = pp.create_empty_network()
        
        # Add buses, generators, loads
        for i in range(179):
            pp.create_bus(net, vn_kv=138, name=f"Bus_{i+1}")
            
            if i < 50:  # Add generators
                pp.create_gen(net, bus=i, p_mw=100, max_p_mw=200, min_p_mw=10)
            
            if i % 2 == 0:  # Add loads
                pp.create_load(net, bus=i, p_mw=50, q_mvar=15)
        
        # Add transmission lines
        for i in range(0, 178, 3):
            pp.create_line(net, from_bus=i, to_bus=i+1, length_km=50, 
                          std_type="243-AL1/39-ST1A 110.0")
        
        return net
    
    def generate_imbalanced_scenarios(self, num_scenarios=10000):
        """Generate training data with various imbalanced scenarios"""
        print(f"Generating {num_scenarios} imbalanced scenarios...")
        
        scenarios = []
        balancing_actions = []
        imbalance_labels = []
        
        np.random.seed(42)
        
        for scenario_id in range(num_scenarios):
            # Create base operating point
            base_loads = self.net.load['p_mw'].values.copy()
            base_gens = self.net.gen['p_mw'].values.copy()
            
            # Introduce random imbalances
            imbalance_type = np.random.choice(['load_increase', 'load_decrease', 
                                             'gen_outage', 'line_outage', 'balanced'])
            
            scenario_data = self._create_imbalance_scenario(
                base_loads, base_gens, imbalance_type
            )
            
            # Calculate required balancing actions
            actions = self._calculate_balancing_actions(scenario_data)
            
            # Determine if system is balanced
            total_imbalance = abs(scenario_data['total_load'] - scenario_data['total_gen'])
            is_balanced = 1 if total_imbalance <= self.imbalance_threshold else 0
            
            scenarios.append(scenario_data)
            balancing_actions.append(actions)
            imbalance_labels.append(is_balanced)
        
        return scenarios, balancing_actions, imbalance_labels
    
    def _create_imbalance_scenario(self, base_loads, base_gens, imbalance_type):
        """Create specific imbalance scenario"""
        loads = base_loads.copy()
        gens = base_gens.copy()
        voltages = np.ones(self.n_buses)
        frequencies = np.ones(self.n_buses) * 50.0
        
        if imbalance_type == 'load_increase':
            affected_loads = np.random.choice(len(loads), size=np.random.randint(1, 5))
            for load_idx in affected_loads:
                increase = np.random.uniform(10, 50)
                loads[load_idx] += increase
        
        elif imbalance_type == 'load_decrease':
            affected_loads = np.random.choice(len(loads), size=np.random.randint(1, 3))
            for load_idx in affected_loads:
                decrease = np.random.uniform(5, 30)
                loads[load_idx] = max(0, loads[load_idx] - decrease)
        
        elif imbalance_type == 'gen_outage':
            outaged_gen = np.random.choice(len(gens))
            outaged_power = gens[outaged_gen]
            gens[outaged_gen] = 0
            
            frequency_drop = min(2.0, outaged_power / 1000)
            frequencies -= frequency_drop
        
        elif imbalance_type == 'line_outage':
            voltage_drop = np.random.uniform(0.02, 0.08)
            affected_buses = np.random.choice(self.n_buses, size=np.random.randint(3, 8))
            voltages[affected_buses] -= voltage_drop
        
        # Calculate power imbalance
        total_load = np.sum(loads)
        total_gen = np.sum(gens)
        power_imbalance = total_gen - total_load
        
        scenario_data = {
            'loads': loads,
            'generations': gens,
            'voltages': voltages,
            'frequencies': frequencies,
            'total_load': total_load,
            'total_gen': total_gen,
            'power_imbalance': power_imbalance,
            'imbalance_type': imbalance_type
        }
        
        return scenario_data
    
    def _calculate_balancing_actions(self, scenario_data):
        """Calculate optimal balancing actions"""
        power_imbalance = scenario_data['power_imbalance']
        gens = scenario_data['generations']
        
        gen_adjustments = np.zeros(self.n_generators)
        load_shedding = np.zeros(self.n_loads)
        
        if abs(power_imbalance) <= self.imbalance_threshold:
            return {
                'gen_adjustments': gen_adjustments,
                'load_shedding': load_shedding,
                'action_type': 'none'
            }
        
        elif power_imbalance < -self.imbalance_threshold:
            # Generation deficit
            deficit = abs(power_imbalance)
            available_capacity = self.gen_limits['pmax'] - gens
            available_capacity[available_capacity < 0] = 0
            
            if np.sum(available_capacity) >= deficit:
                # Increase generation
                total_available = np.sum(available_capacity)
                if total_available > 0:
                    for i, capacity in enumerate(available_capacity):
                        if capacity > 0:
                            gen_adjustments[i] = (capacity / total_available) * deficit
                action_type = 'increase_generation'
            else:
                # Load shedding needed
                remaining_deficit = deficit - np.sum(available_capacity)
                gen_adjustments = available_capacity
                
                total_load = np.sum(scenario_data['loads'])
                if total_load > 0:
                    for i, load in enumerate(scenario_data['loads']):
                        load_shedding[i] = (load / total_load) * remaining_deficit
                action_type = 'load_shedding'
        
        else:
            # Generation excess
            excess = power_imbalance
            total_gen = np.sum(gens)
            if total_gen > 0:
                for i, gen in enumerate(gens):
                    if gen > self.gen_limits['pmin'][i]:
                        max_decrease = gen - self.gen_limits['pmin'][i]
                        gen_adjustments[i] = -min(max_decrease, (gen / total_gen) * excess)
            action_type = 'decrease_generation'
        
        return {
            'gen_adjustments': gen_adjustments,
            'load_shedding': load_shedding,
            'action_type': action_type
        }
    
    def prepare_training_data(self, scenarios, actions, labels):
        """Prepare PyTorch training data"""
        print("Preparing PyTorch training data for load balancing...")
        
        # Convert scenarios to feature vectors
        X_features = []
        y_actions = []
        y_balance = []
        
        for i, scenario in enumerate(scenarios):
            features = np.concatenate([
                scenario['loads'],
                scenario['generations'], 
                scenario['voltages'],
                scenario['frequencies'],
                [scenario['power_imbalance'], scenario['total_load'], scenario['total_gen']]
            ])
            
            X_features.append(features)
            
            action_vector = np.concatenate([
                actions[i]['gen_adjustments'],
                actions[i]['load_shedding']
            ])
            y_actions.append(action_vector)
            y_balance.append(labels[i])
        
        X_features = np.array(X_features)
        y_actions = np.array(y_actions)
        y_balance = np.array(y_balance)
        
        # Create sequences
        X_sequences = []
        y_action_sequences = []
        y_balance_sequences = []
        
        for i in range(self.sequence_length, len(X_features)):
            X_sequences.append(X_features[i-self.sequence_length:i])
            y_action_sequences.append(y_actions[i])
            y_balance_sequences.append(y_balance[i])
        
        X_sequences = np.array(X_sequences)
        y_action_sequences = np.array(y_action_sequences)
        y_balance_sequences = np.array(y_balance_sequences)
        
        # Scale data
        n_samples, n_timesteps, n_features = X_sequences.shape
        X_reshaped = X_sequences.reshape(-1, n_features)
        X_scaled = self.scaler_state.fit_transform(X_reshaped)
        X_sequences_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        y_actions_scaled = self.scaler_actions.fit_transform(y_action_sequences)
        
        # Split data
        train_size = int(0.7 * len(X_sequences_scaled))
        val_size = int(0.15 * len(X_sequences_scaled))
        
        train_data = {
            'X': X_sequences_scaled[:train_size],
            'y_actions': y_actions_scaled[:train_size],
            'y_balance': y_balance_sequences[:train_size]
        }
        
        val_data = {
            'X': X_sequences_scaled[train_size:train_size+val_size],
            'y_actions': y_actions_scaled[train_size:train_size+val_size],
            'y_balance': y_balance_sequences[train_size:train_size+val_size]
        }
        
        test_data = {
            'X': X_sequences_scaled[train_size+val_size:],
            'y_actions': y_actions_scaled[train_size+val_size:],
            'y_balance': y_balance_sequences[train_size+val_size:]
        }
        
        print(f"Training sequences: {train_data['X'].shape}")
        print(f"Features per timestep: {train_data['X'].shape[2]}")
        print(f"Action dimension: {train_data['y_actions'].shape[1]}")
        
        return train_data, val_data, test_data
    
    def create_data_loaders(self, train_data, val_data, test_data, batch_size=32):
        """Create PyTorch DataLoaders"""
        train_dataset = PowerSystemDataset(
            train_data['X'], train_data['y_actions'], train_data['y_balance']
        )
        val_dataset = PowerSystemDataset(
            val_data['X'], val_data['y_actions'], val_data['y_balance']
        )
        test_dataset = PowerSystemDataset(
            test_data['X'], test_data['y_actions'], test_data['y_balance']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, epochs=100, lr=0.001):
        """Train the PyTorch LSTM model"""
        print("Training PyTorch Load Balancing LSTM Model...")
        
        # Get input dimensions from first batch
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[2]  # Features per timestep
        action_dim = sample_batch[1].shape[1]  # Action dimension
        
        # Initialize model
        self.model = LoadBalancingLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            action_dim=action_dim,
            dropout=0.3
        ).to(self.device)
        
        print(f"Model initialized with input_size={input_size}, action_dim={action_dim}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss functions and optimizer
        balance_criterion = nn.BCELoss()
        action_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_balance_acc': [], 'val_balance_acc': [],
            'train_action_mae': [], 'val_action_mae': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            train_balance_correct = 0
            train_balance_total = 0
            train_action_maes = []
            
            for batch_X, batch_y_actions, batch_y_balance in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y_actions = batch_y_actions.to(self.device)
                batch_y_balance = batch_y_balance.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                
                # Forward pass
                balance_pred, action_pred = self.model(batch_X)
                
                # Calculate losses
                balance_loss = balance_criterion(balance_pred, batch_y_balance)
                action_loss = action_criterion(action_pred, batch_y_actions)
                
                # Combined loss with weights
                total_loss = balance_loss + 10.0 * action_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
                
                # Calculate accuracy for balance classification
                balance_pred_binary = (balance_pred > 0.5).float()
                train_balance_correct += (balance_pred_binary == batch_y_balance).sum().item()
                train_balance_total += batch_y_balance.size(0)
                
                # Calculate MAE for actions
                action_mae = F.l1_loss(action_pred, batch_y_actions).item()
                train_action_maes.append(action_mae)
            
            # Validation phase
            self.model.eval()
            val_losses = []
            val_balance_correct = 0
            val_balance_total = 0
            val_action_maes = []
            
            with torch.no_grad():
                for batch_X, batch_y_actions, batch_y_balance in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y_actions = batch_y_actions.to(self.device)
                    batch_y_balance = batch_y_balance.to(self.device).unsqueeze(1)
                    
                    balance_pred, action_pred = self.model(batch_X)
                    
                    balance_loss = balance_criterion(balance_pred, batch_y_balance)
                    action_loss = action_criterion(action_pred, batch_y_actions)
                    total_loss = balance_loss + 10.0 * action_loss
                    
                    val_losses.append(total_loss.item())
                    
                    balance_pred_binary = (balance_pred > 0.5).float()
                    val_balance_correct += (balance_pred_binary == batch_y_balance).sum().item()
                    val_balance_total += batch_y_balance.size(0)
                    
                    action_mae = F.l1_loss(action_pred, batch_y_actions).item()
                    val_action_maes.append(action_mae)
            
            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_balance_acc = train_balance_correct / train_balance_total
            val_balance_acc = val_balance_correct / val_balance_total
            train_action_mae = np.mean(train_action_maes)
            val_action_mae = np.mean(val_action_maes)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_balance_acc'].append(train_balance_acc)
            history['val_balance_acc'].append(val_balance_acc)
            history['train_action_mae'].append(train_action_mae)
            history['val_action_mae'].append(val_action_mae)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_load_balancer.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"  Train Balance Acc: {train_balance_acc:.4f}, Val Balance Acc: {val_balance_acc:.4f}")
                print(f"  Train Action MAE: {train_action_mae:.4f}, Val Action MAE: {val_action_mae:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_load_balancer.pth'))
        
        return history
    
    def evaluate_model(self, test_loader):
        """Evaluate the trained PyTorch model"""
        print("Evaluating PyTorch Load Balancing Model...")
        
        self.model.eval()
        
        all_balance_preds = []
        all_balance_true = []
        all_action_preds = []
        all_action_true = []
        
        with torch.no_grad():
            for batch_X, batch_y_actions, batch_y_balance in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y_actions = batch_y_actions.to(self.device)
                batch_y_balance = batch_y_balance.to(self.device)
                
                balance_pred, action_pred = self.model(batch_X)
                
                # Move predictions to CPU for evaluation
                balance_pred = balance_pred.cpu().numpy()
                action_pred = action_pred.cpu().numpy()
                batch_y_actions = batch_y_actions.cpu().numpy()
                batch_y_balance = batch_y_balance.cpu().numpy()
                
                all_balance_preds.extend(balance_pred.flatten())
                all_balance_true.extend(batch_y_balance)
                all_action_preds.extend(action_pred)
                all_action_true.extend(batch_y_actions)
        
        # Convert to numpy arrays
        balance_preds = np.array(all_balance_preds)
        balance_true = np.array(all_balance_true)
        action_preds = np.array(all_action_preds)
        action_true = np.array(all_action_true)
        
        # Evaluate balance classification
        balance_pred_binary = (balance_preds > 0.5).astype(int)
        balance_accuracy = accuracy_score(balance_true, balance_pred_binary)
        
        print(f"Balance Classification Accuracy: {balance_accuracy:.4f}")
        print("\nBalance Classification Report:")
        print(classification_report(balance_true, balance_pred_binary))
        
        # Evaluate action prediction
        action_preds_orig = self.scaler_actions.inverse_transform(action_preds)
        action_true_orig = self.scaler_actions.inverse_transform(action_true)
        
        action_mae = mean_absolute_error(action_true_orig, action_preds_orig)
        action_rmse = np.sqrt(mean_squared_error(action_true_orig, action_preds_orig))
        
        print(f"\nBalancing Actions Prediction:")
        print(f"  MAE: {action_mae:.4f}")
        print(f"  RMSE: {action_rmse:.4f}")
        
        return {
            'balance_accuracy': balance_accuracy,
            'action_mae': action_mae,
            'action_rmse': action_rmse,
            'predictions': {
                'balance': balance_pred_binary,
                'actions': action_preds_orig
            },
            'actual': {
                'balance': balance_true,
                'actions': action_true_orig
            }
        }
    
    def real_time_load_balancing(self, current_state):
        """Perform real-time load balancing using PyTorch model"""
        self.model.eval()
        
        # Prepare state vector
        state_vector = np.concatenate([
            current_state['loads'],
            current_state['generations'],
            current_state['voltages'], 
            current_state['frequencies'],
            [current_state['power_imbalance'], 
             current_state['total_load'], 
             current_state['total_gen']]
        ])
        
        # Create sequence
        state_sequence = np.array([state_vector] * self.sequence_length)
        state_sequence = state_sequence.reshape(1, self.sequence_length, -1)
        
        # Scale input
        state_reshaped = state_sequence.reshape(-1, state_sequence.shape[2])
        state_scaled = self.scaler_state.transform(state_reshaped)
        state_sequence_scaled = state_scaled.reshape(state_sequence.shape)
        
        # Convert to PyTorch tensor
        state_tensor = torch.FloatTensor(state_sequence_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            balance_pred, action_pred = self.model(state_tensor)
            
            balance_prob = balance_pred.cpu().numpy()[0][0]
            actions_scaled = action_pred.cpu().numpy()[0]
        
        # Inverse transform actions
        actions = self.scaler_actions.inverse_transform(actions_scaled.reshape(1, -1))[0]
        
        # Parse actions
        n_gen_actions = self.n_generators
        gen_adjustments = actions[:n_gen_actions]
        load_shedding = actions[n_gen_actions:]
        
        is_balanced = balance_prob > 0.5
        
        return {
            'is_balanced': is_balanced,
            'balance_probability': balance_prob,
            'generator_adjustments': gen_adjustments,
            'load_shedding': load_shedding,
            'total_gen_adjustment': np.sum(gen_adjustments),
            'total_load_shedding': np.sum(load_shedding)
        }

def plot_training_history(history):
    """Plot PyTorch training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Balance Accuracy
    axes[0, 1].plot(history['train_balance_acc'], label='Train Balance Acc')
    axes[0, 1].plot(history['val_balance_acc'], label='Val Balance Acc')
    axes[0, 1].set_title('Balance Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Action MAE
    axes[0, 2].plot(history['train_action_mae'], label='Train Action MAE')
    axes[0, 2].plot(history['val_action_mae'], label='Val Action MAE')
    axes[0, 2].set_title('Action Prediction MAE')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Clear unused subplots
    for i in range(1, 2):
        for j in range(3):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def main_pytorch_load_balancing():
    """Main function for PyTorch load balancing training"""
    # Initialize load balancer
    balancer = PyTorchLoadBalancer(system_type='IEEE39')
    
    # Step 1: Load power system
    print("=== Step 1: Loading Power System ===")
    net = balancer.load_power_system()
    
    # Step 2: Generate imbalanced scenarios
    print("\n=== Step 2: Generating Imbalanced Scenarios ===")
    scenarios, actions, labels = balancer.generate_imbalanced_scenarios(num_scenarios=8000)
    
    # Step 3: Prepare training data
    print("\n=== Step 3: Preparing Training Data ===")
    train_data, val_data, test_data = balancer.prepare_training_data(scenarios, actions, labels)
    
    # Step 4: Create data loaders
    print("\n=== Step 4: Creating PyTorch DataLoaders ===")
    train_loader, val_loader, test_loader = balancer.create_data_loaders(
        train_data, val_data, test_data, batch_size=32
    )
    
    # Step 5: Train model
    print("\n=== Step 5: Training PyTorch Load Balancing Model ===")
    history = balancer.train_model(train_loader, val_loader, epochs=100, lr=0.001)
    
    # Step 6: Evaluate model
    print("\n=== Step 6: Evaluating Model ===")
    results = balancer.evaluate_model(test_loader)
    
    # Step 7: Plot training history
    print("\n=== Step 7: Plotting Training History ===")
    plot_training_history(history)
    
    # Step 8: Demo real-time balancing
    print("\n=== Step 8: Real-Time Load Balancing Demo ===")
    
    # Create test imbalanced state
    test_state = {
        'loads': balancer.net.load['p_mw'].values * 1.2,  # 20% load increase
        'generations': balancer.net.gen['p_mw'].values,
        'voltages': np.ones(balancer.n_buses) * 0.98,  # Slight voltage drop
        'frequencies': np.ones(balancer.n_buses) * 49.8,  # Frequency drop
        'power_imbalance': -50,  # Generation deficit
        'total_load': np.sum(balancer.net.load['p_mw'].values * 1.2),
        'total_gen': np.sum(balancer.net.gen['p_mw'].values)
    }
    
    balancing_action = balancer.real_time_load_balancing(test_state)
    
    print(f"System Balance Status: {'Balanced' if balancing_action['is_balanced'] else 'Imbalanced'}")
    print(f"Balance Probability: {balancing_action['balance_probability']:.3f}")
    print(f"Required Generation Adjustment: {balancing_action['total_gen_adjustment']:.2f} MW")
    print(f"Required Load Shedding: {balancing_action['total_load_shedding']:.2f} MW")
    
    # Show individual generator adjustments
    print(f"\nDetailed Generator Adjustments:")
    for i, adj in enumerate(balancing_action['generator_adjustments']):
        if abs(adj) > 0.1:  # Only show significant adjustments
            print(f"  Generator {i+1}: {adj:+.2f} MW")
    
    # Show load shedding if any
    if balancing_action['total_load_shedding'] > 0.1:
        print(f"\nLoad Shedding Required:")
        for i, shed in enumerate(balancing_action['load_shedding']):
            if abs(shed) > 0.1:
                print(f"  Load {i+1}: {shed:.2f} MW")
    
    print("\n=== PyTorch Load Balancing Training Completed! ===")
    
    return balancer, history, results

def demo_multiple_scenarios():
    """Demonstrate load balancing on multiple scenarios"""
    print("\n=== Multi-Scenario Load Balancing Demo ===")
    
    # Load the trained balancer (assuming it's already trained)
    balancer = PyTorchLoadBalancer(system_type='IEEE39')
    balancer.load_power_system()
    
    # You would load the trained model here:
    # balancer.model = LoadBalancingLSTM(...)
    # balancer.model.load_state_dict(torch.load('best_load_balancer.pth'))
    
    scenarios = [
        {
            'name': 'Normal Operation',
            'state': {
                'loads': balancer.net.load['p_mw'].values,
                'generations': balancer.net.gen['p_mw'].values,
                'voltages': np.ones(balancer.n_buses),
                'frequencies': np.ones(balancer.n_buses) * 50.0,
                'power_imbalance': 0,
                'total_load': np.sum(balancer.net.load['p_mw'].values),
                'total_gen': np.sum(balancer.net.gen['p_mw'].values)
            }
        },
        {
            'name': 'High Load Demand',
            'state': {
                'loads': balancer.net.load['p_mw'].values * 1.3,
                'generations': balancer.net.gen['p_mw'].values,
                'voltages': np.ones(balancer.n_buses) * 0.97,
                'frequencies': np.ones(balancer.n_buses) * 49.7,
                'power_imbalance': -100,
                'total_load': np.sum(balancer.net.load['p_mw'].values * 1.3),
                'total_gen': np.sum(balancer.net.gen['p_mw'].values)
            }
        },
        {
            'name': 'Generator Outage',
            'state': {
                'loads': balancer.net.load['p_mw'].values,
                'generations': balancer.net.gen['p_mw'].values * 0.8,  # 20% generation loss
                'voltages': np.ones(balancer.n_buses) * 0.95,
                'frequencies': np.ones(balancer.n_buses) * 49.5,
                'power_imbalance': -80,
                'total_load': np.sum(balancer.net.load['p_mw'].values),
                'total_gen': np.sum(balancer.net.gen['p_mw'].values * 0.8)
            }
        },
        {
            'name': 'Low Load Period',
            'state': {
                'loads': balancer.net.load['p_mw'].values * 0.6,
                'generations': balancer.net.gen['p_mw'].values,
                'voltages': np.ones(balancer.n_buses) * 1.02,
                'frequencies': np.ones(balancer.n_buses) * 50.2,
                'power_imbalance': 150,
                'total_load': np.sum(balancer.net.load['p_mw'].values * 0.6),
                'total_gen': np.sum(balancer.net.gen['p_mw'].values)
            }
        }
    ]
    
    print("Note: This demo requires a trained model. Run main_pytorch_load_balancing() first.")
    print("Scenario demonstrations:")
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Total Load: {scenario['state']['total_load']:.2f} MW")
        print(f"Total Generation: {scenario['state']['total_gen']:.2f} MW")
        print(f"Power Imbalance: {scenario['state']['power_imbalance']:.2f} MW")
        print(f"Average Frequency: {np.mean(scenario['state']['frequencies']):.2f} Hz")
        print(f"Average Voltage: {np.mean(scenario['state']['voltages']):.3f} pu")
        
        # Uncomment this when model is trained:
        # action = balancer.real_time_load_balancing(scenario['state'])
        # print(f"Recommended Action: {action}")

# Additional utility functions for PyTorch implementation

def save_model_and_scalers(balancer, model_path='load_balancer_model.pth'):
    """Save the trained model and scalers"""
    torch.save({
        'model_state_dict': balancer.model.state_dict(),
        'scaler_state': balancer.scaler_state,
        'scaler_actions': balancer.scaler_actions,
        'n_buses': balancer.n_buses,
        'n_generators': balancer.n_generators,
        'n_loads': balancer.n_loads,
        'sequence_length': balancer.sequence_length,
        'imbalance_threshold': balancer.imbalance_threshold
    }, model_path)
    print(f"Model and scalers saved to {model_path}")

def load_model_and_scalers(balancer, model_path='load_balancer_model.pth'):
    """Load the trained model and scalers"""
    checkpoint = torch.load(model_path, map_location=balancer.device)
    
    # Restore balancer parameters
    balancer.scaler_state = checkpoint['scaler_state']
    balancer.scaler_actions = checkpoint['scaler_actions']
    balancer.n_buses = checkpoint['n_buses']
    balancer.n_generators = checkpoint['n_generators']
    balancer.n_loads = checkpoint['n_loads']
    balancer.sequence_length = checkpoint['sequence_length']
    balancer.imbalance_threshold = checkpoint['imbalance_threshold']
    
    # Load model state
    balancer.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model and scalers loaded from {model_path}")

def analyze_model_performance(results):
    """Analyze and print detailed model performance"""
    print("\n=== Detailed Performance Analysis ===")
    
    balance_preds = results['predictions']['balance']
    balance_actual = results['actual']['balance']
    action_preds = results['predictions']['actions']
    action_actual = results['actual']['actions']
    
    # Balance classification analysis
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(balance_actual, balance_preds)
    print(f"\nBalance Classification Confusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Balanced  Imbalanced")
    print(f"Balanced      {cm[1,1]:4d}      {cm[1,0]:4d}")
    print(f"Imbalanced    {cm[0,1]:4d}      {cm[0,0]:4d}")
    
    # Action prediction analysis
    action_errors = np.abs(action_preds - action_actual)
    print(f"\nAction Prediction Analysis:")
    print(f"Mean Absolute Error: {np.mean(action_errors):.4f}")
    print(f"Median Absolute Error: {np.median(action_errors):.4f}")
    print(f"95th Percentile Error: {np.percentile(action_errors, 95):.4f}")
    print(f"Max Error: {np.max(action_errors):.4f}")
    
    # Per-action analysis (first few actions)
    print(f"\nPer-Action MAE (first 10 actions):")
    for i in range(min(10, action_preds.shape[1])):
        mae = np.mean(np.abs(action_preds[:, i] - action_actual[:, i]))
        print(f"Action {i+1}: {mae:.4f}")

if __name__ == "__main__":
    # Run the main training and evaluation
    balancer, training_history, evaluation_results = main_pytorch_load_balancing()
    
    # Save the trained model
    save_model_and_scalers(balancer)
    
    # Analyze performance
    analyze_model_performance(evaluation_results)
    
    # Demonstrate multiple scenarios
    demo_multiple_scenarios()
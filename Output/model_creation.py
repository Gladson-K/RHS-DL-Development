import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import h5py
import zipfile
import os
import joblib

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define FCNN Model with custom test and reconstruction methods
class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128, 64], output_dim=25601):
        super(FCNN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Input layer
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        prev_dim = hidden_dims[0]
        
        # Hidden layers
        for dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        # Additional dense layer
        layers.append(nn.Linear(prev_dim, prev_dim // 2))
        layers.append(nn.PReLU())
        layers.append(nn.Dropout(0.1))
        prev_dim = prev_dim // 2
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def reconstruct_M(self, output, batch_size):
        """Reconstruct complex-valued M matrix from FCNN output"""
        M_real = output[:, :12800].reshape(batch_size, 20, 64, 10)
        M_imag = output[:, 12800:25600].reshape(batch_size, 20, 64, 10)
        M_complex = torch.complex(M_real, M_imag)
        sum_rate = output[:, 25600]
        return M_complex, sum_rate
    
    def test(self, test_loader, device="cpu"):
        """Evaluate model and compute separate metrics for M_real, M_imag, and sum_rate"""
        self.eval()
        self.to(device)
        total_mse_M_real = 0
        total_mae_M_real = 0
        total_mse_M_imag = 0
        total_mae_M_imag = 0
        total_mse_sum_rate = 0
        total_mae_sum_rate = 0
        total_r2_M_real = 0
        total_r2_M_imag = 0
        total_r2_sum_rate = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = self(batch_x)
                
                # Split output and target
                M_real_pred = output[:, :12800]
                M_imag_pred = output[:, 12800:25600]
                sum_rate_pred = output[:, 25600]
                
                M_real_true = batch_y[:, :12800]
                M_imag_true = batch_y[:, 12800:25600]
                sum_rate_true = batch_y[:, 25600]
                
                # Metrics for M_real
                mse_M_real = F.mse_loss(M_real_pred, M_real_true, reduction='sum').item()
                mae_M_real = F.l1_loss(M_real_pred, M_real_true, reduction='sum').item()
                ss_tot_M_real = torch.sum((M_real_true - M_real_true.mean(dim=0))**2)
                ss_res_M_real = torch.sum((M_real_true - M_real_pred)**2)
                r2_M_real = 1 - (ss_res_M_real / (ss_tot_M_real + 1e-9))
                
                total_mse_M_real += mse_M_real
                total_mae_M_real += mae_M_real
                total_r2_M_real += r2_M_real.item()
                
                # Metrics for M_imag
                mse_M_imag = F.mse_loss(M_imag_pred, M_imag_true, reduction='sum').item()
                mae_M_imag = F.l1_loss(M_imag_pred, M_imag_true, reduction='sum').item()
                ss_tot_M_imag = torch.sum((M_imag_true - M_imag_true.mean(dim=0))**2)
                ss_res_M_imag = torch.sum((M_imag_true - M_imag_pred)**2)
                r2_M_imag = 1 - (ss_res_M_imag / (ss_tot_M_imag + 1e-9))
                
                total_mse_M_imag += mse_M_imag
                total_mae_M_imag += mae_M_imag
                total_r2_M_imag += r2_M_imag.item()
                
                # Metrics for sum_rate
                mse_sum_rate = F.mse_loss(sum_rate_pred, sum_rate_true, reduction='sum').item()
                mae_sum_rate = F.l1_loss(sum_rate_pred, sum_rate_true, reduction='sum').item()
                ss_tot_sum_rate = torch.sum((sum_rate_true - sum_rate_true.mean())**2)
                ss_res_sum_rate = torch.sum((sum_rate_true - sum_rate_pred)**2)
                r2_sum_rate = 1 - (ss_res_sum_rate / (ss_tot_sum_rate + 1e-9))
                
                total_mse_sum_rate += mse_sum_rate
                total_mae_sum_rate += mae_sum_rate
                total_r2_sum_rate += r2_sum_rate.item()
        
        n_samples = len(test_loader.dataset)
        n_elements_M = 12800
        
        avg_mse_M_real = total_mse_M_real / (n_samples * n_elements_M)
        avg_mae_M_real = total_mae_M_real / (n_samples * n_elements_M)
        avg_r2_M_real = total_r2_M_real / len(test_loader)
        
        avg_mse_M_imag = total_mse_M_imag / (n_samples * n_elements_M)
        avg_mae_M_imag = total_mae_M_imag / (n_samples * n_elements_M)
        avg_r2_M_imag = total_r2_M_imag / len(test_loader)
        
        avg_mse_sum_rate = total_mse_sum_rate / n_samples
        avg_mae_sum_rate = total_mae_sum_rate / n_samples
        avg_r2_sum_rate = total_r2_sum_rate / len(test_loader)
        
        return {
            'M_real': {'mse': avg_mse_M_real, 'mae': avg_mae_M_real, 'r2': avg_r2_M_real},
            'M_imag': {'mse': avg_mse_M_imag, 'mae': avg_mae_M_imag, 'r2': avg_r2_M_imag},
            'sum_rate': {'mse': avg_mse_sum_rate, 'mae': avg_mae_sum_rate, 'r2': avg_r2_sum_rate}
        }

# Extract and process dataset
def extract_and_process_dataset(zip_path="D://Output//64x64_MIMO_beamforming_dataset.zip", h5_name="64x64_MIMO_beamforming_dataset.h5"):
    extract_dir = "temp_h5_extract"
    os.makedirs(extract_dir, exist_ok=True)
    h5_path = os.path.join(extract_dir, h5_name)
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extract(h5_name, extract_dir)
        print(f"Extracted {h5_name} to {extract_dir}")
    
    with h5py.File(h5_path, 'r') as f:
        metadata = f['metadata'][:]
        M = f['M'][:]  # (25000, 20, 64, MAX_RFNO) complex64
        rate = metadata[:, 12]
        RFno = metadata[:, 9].astype(int)  # Number of RF chains per sample
        test_size = metadata[:, 8].astype(int)  # Test size per sample
        
        num_samples = metadata.shape[0]
        desired_rfno = 10  # Use maximum RFno
        desired_test_size = 20  # MAX_TEST_SIZE
        elements_per_sample = desired_test_size * 64 * desired_rfno  # 12800 for RFno=10
        
        y = np.zeros((num_samples, elements_per_sample * 2 + 1), dtype=np.float32)
        for i in range(num_samples):
            # Extract M with actual test_size and RFno
            M_sample = M[i, :test_size[i], :, :RFno[i]]  # (test_size[i], 64, RFno[i])
            
            # Adjust test_size first to ensure consistent axis=0
            if test_size[i] < desired_test_size:
                pad_shape = (desired_test_size - test_size[i], 64, RFno[i])
                M_sample = np.concatenate([M_sample, np.zeros(pad_shape, dtype=M_sample.dtype)], axis=0)
            elif test_size[i] > desired_test_size:
                M_sample = M_sample[:desired_test_size, :, :]
            
            # Now pad RFno to match desired_rfno
            if RFno[i] < desired_rfno:
                pad_shape = (desired_test_size, 64, desired_rfno - RFno[i])
                M_sample = np.concatenate([M_sample, np.zeros(pad_shape, dtype=M_sample.dtype)], axis=2)
            elif RFno[i] > desired_rfno:
                M_sample = M_sample[:, :, :desired_rfno]  # Truncate (shouldn't occur with MAX_RFNO=10)
            
            # Verify shape
            assert M_sample.shape == (desired_test_size, 64, desired_rfno), \
                f"M_sample shape {M_sample.shape} does not match ({desired_test_size}, 64, {desired_rfno})"
            
            M_real = M_sample.real.flatten()  # (12800,)
            M_imag = M_sample.imag.flatten()  # (12800,)
            y[i, :elements_per_sample] = M_real
            y[i, elements_per_sample:elements_per_sample*2] = M_imag
            y[i, elements_per_sample*2] = rate[i]
        
        feature_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        X = metadata[:, feature_indices]  # (25000, 10)
    
    os.remove(h5_path)
    os.rmdir(extract_dir)
    print(f"Cleaned up {h5_path} and {extract_dir}")
    
    return X, y

# Train and evaluate with cross-validation
def train_evaluate_fcnn(model, X, y, k_folds=5, learning_rates=[0.001, 0.01], optimizers=["adam", "sgd", "RMSProp"], epochs=100):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []
    best_score = float('inf')
    best_config = None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for optimizer_name in optimizers:
        for lr in learning_rates:
            fold_mse = []
            print(f"\nTesting {optimizer_name} with lr={lr}")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train_fold = X_scaled[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X_scaled[val_idx]
                y_val_fold = y[val_idx]
                
                train_dataset = TensorDataset(
                    torch.tensor(X_train_fold, dtype=torch.float32),
                    torch.tensor(y_train_fold, dtype=torch.float32)
                )
                val_dataset = TensorDataset(
                    torch.tensor(X_val_fold, dtype=torch.float32),
                    torch.tensor(y_val_fold, dtype=torch.float32)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                model_fold = FCNN(input_dim=X.shape[1], output_dim=25601).to(device)
                
                if optimizer_name == "adam":
                    optimizer = Adam(model_fold.parameters(), lr=lr, weight_decay=1e-4)
                elif optimizer_name == "sgd":
                    optimizer = SGD(model_fold.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                elif optimizer_name == "RMSProp":
                    optimizer_name = RMSprop(best_model.parameters(), lr=best_lr, alpha=0.9, weight_decay=1e-4)
                criterion = nn.MSELoss()
                model_fold.train()
                for epoch in range(epochs):
                    total_loss = 0
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        output = model_fold(batch_x)
                        loss = criterion(output, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_fold.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item()
                
                metrics = model_fold.test(val_loader, device=device)
                fold_mse.append(metrics['sum_rate']['mse'])
                print(f"Fold {fold+1}, Sum Rate MSE: {metrics['sum_rate']['mse']:.6f}")
            
            avg_mse = np.mean(fold_mse)
            results.append({
                'optimizer': optimizer_name,
                'lr': lr,
                'mse': avg_mse
            })
            print(f"Average Sum Rate MSE: {avg_mse:.6f}")
            
            if avg_mse < best_score:
                best_score = avg_mse
                best_config = {'optimizer': optimizer_name, 'lr': lr}
    
    return results, best_config

# Main execution
if __name__ == "__main__":
    print("Extracting and processing dataset...")
    X, y = extract_and_process_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    fcnn_model = FCNN(input_dim=X_train.shape[1], output_dim=25601)
    results, best_config = train_evaluate_fcnn(fcnn_model, X_train, y_train, k_folds=5, 
                                             learning_rates=[0.001, 0.01],
                                             optimizers=["adam", "sgd", "RMSProp"], epochs=100)
    
    print("\nCross-Validation Results:")
    for result in results:
        print(f"Optimizer: {result['optimizer']}, LR: {result['lr']}, Avg Sum Rate MSE: {result['mse']:.6f}")
    print(f"\nBest Configuration: Optimizer={best_config['optimizer']}, LR={best_config['lr']}")
    
    # Retrain best model
    best_model = FCNN(input_dim=X_train.shape[1], output_dim=25601).to(device)
    best_optimizer = best_config['optimizer']
    best_lr = best_config['lr']
    model_filename = f'best_fcnn_{best_optimizer}_lr{best_lr}_final.pkl'
    
    if best_optimizer == "adam":
        optimizer = Adam(best_model.parameters(), lr=best_lr, weight_decay=1e-4)
    elif best_optimizer == "sgd":
        optimizer = SGD(best_model.parameters(), lr=best_lr, momentum=0.9, weight_decay=1e-4)
    elif best_optimizer == "RMSProp":
        optimizer = RMSprop(best_model.parameters(), lr=best_lr, alpha=0.9, weight_decay=1e-4)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    full_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.float32))
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.MSELoss()
    best_model.train()
    for epoch in range(300):
        total_loss = 0
        for batch_x, batch_y in full_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = best_model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(full_loader):.6f}")
    
    joblib.dump(best_model, model_filename)
    joblib.dump(scaler, 'scaler_final.pkl')
    print(f"Model saved as '{model_filename}'")
    
    # Evaluate on test set and reconstruct M
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load('scaler_final.pkl')
    
    X_test_scaled = loaded_scaler.transform(X_test)
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                               torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_metrics = loaded_model.test(test_loader, device=device)
    
    print("\nFinal Test Metrics:")
    print("M_real:")
    print(f"  MSE: {test_metrics['M_real']['mse']:.6f}")
    print(f"  MAE: {test_metrics['M_real']['mae']:.6f}")
    print(f"  R²: {test_metrics['M_real']['r2']:.6f}")
    print("M_imag:")
    print(f"  MSE: {test_metrics['M_imag']['mse']:.6f}")
    print(f"  MAE: {test_metrics['M_imag']['mae']:.6f}")
    print(f"  R²: {test_metrics['M_imag']['r2']:.6f}")
    print("Sum Rate:")
    print(f"  MSE: {test_metrics['sum_rate']['mse']:.6f}")
    print(f"  MAE: {test_metrics['sum_rate']['mae']:.6f}")
    print(f"  R²: {test_metrics['sum_rate']['r2']:.6f}")
    
    # Demonstrate reconstruction on a single batch
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            output = loaded_model(batch_x)
            M_complex, sum_rate = loaded_model.reconstruct_M(output, batch_x.size(0))
            print(f"\nReconstructed M shape: {M_complex.shape}")  # Should be [batch_size, 20, 64, 10]
            print(f"Reconstructed M dtype: {M_complex.dtype}")  # Should be torch.complex64
            print(f"Sum Rate shape: {sum_rate.shape}")  # Should be [batch_size]
            break
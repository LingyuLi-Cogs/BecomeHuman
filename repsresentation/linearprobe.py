import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import random
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import gc
import pickle
import Levenshtein
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Linear Probe Training for Representational Analysis')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing H5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--mode', type=str, choices=['ref', 'log', 'levenshtein'], default='ref',
                       help='Distance calculation mode')
    parser.add_argument('--reference_year', type=int, default=2025,
                       help='Reference year for ref mode')
    parser.add_argument('--year_start', type=int, default=1525,
                       help='Start year for year range')
    parser.add_argument('--year_end', type=int, default=2525,
                       help='End year for year range')
    
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

def calculate_ref_log_distance(p, ref_year):
    i, j = p
    val_i = np.log(np.abs(ref_year - i) + 1e-9)
    val_j = np.log(np.abs(ref_year - j) + 1e-9)
    i_on_same_side = (i > ref_year and j > ref_year) or \
                     (i < ref_year and j < ref_year) or \
                     (i == j)
    if i_on_same_side:
        return np.abs(val_i - val_j)
    else:
        return val_i + val_j

def calculate_log_distance(p):
    i, j = p
    return np.abs(np.log(i) - np.log(j))

def calculate_levenshtein(p):
    i, j = p
    return Levenshtein.distance(str(i), str(j))

def detect_model_parameters(data_dir):
    print("=" * 50)
    print("Auto-detecting model parameters from H5 files...")
    
    if not os.path.exists(data_dir) or not any(f.endswith('.h5') for f in os.listdir(data_dir)):
        raise FileNotFoundError(f"Error: Data directory '{data_dir}' does not exist or contains no .h5 files.")
    
    try:
        all_h5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        
        with h5py.File(all_h5_files[0], 'r') as f:
            layer_keys = [k for k in f.keys() if k.startswith('layer_')]
            if not layer_keys:
                raise ValueError(f"Error: No 'layer_X' datasets found in H5 file '{all_h5_files[0]}'.")
            
            layers_to_probe = sorted([int(k.split('_')[1]) for k in layer_keys])
            first_layer_data = f[layer_keys[0]]
            hidden_size = first_layer_data.shape[1]
        
        print(f"   - Detected layers ({len(layers_to_probe)} total): {layers_to_probe}")
        print(f"   - Detected hidden size: {hidden_size}")
        
        return all_h5_files, layers_to_probe, hidden_size
        
    except Exception as e:
        print(f"Error: Unable to auto-detect parameters. Please check H5 files.")
        raise e
    finally:
        print("=" * 50)

def generate_year_pairs_and_targets(args):
    print("Generating year pairs...")
    years = list(range(args.year_start, args.year_end))
    all_pairs = list(itertools.combinations(years, 2))
    self_pairs = [(y, y) for y in years]
    all_pairs.extend(self_pairs)
    print(f"Year pairs generated. Total: {len(all_pairs)}")
    
    print(f"Calculating {args.mode} distances...")
    if args.mode == 'ref':
        y_targets = np.array([calculate_ref_log_distance(p, args.reference_year) for p in tqdm(all_pairs)], dtype=np.float32)
    elif args.mode == 'log':
        y_targets = np.array([calculate_log_distance(p) for p in tqdm(all_pairs)], dtype=np.float32)
    elif args.mode == 'levenshtein':
        y_targets = np.array([calculate_levenshtein(p) for p in tqdm(all_pairs)], dtype=np.float32)
    
    return all_pairs, y_targets

def split_files(all_files, args):
    print("Splitting H5 files...")
    random.seed(args.random_seed)
    random.shuffle(all_files)
    
    n_total = len(all_files)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val :]
    
    print(f"Total files: {n_total}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    return train_files, val_files, test_files

class ResidualDataset(Dataset):
    def __init__(self, file_paths, layer_idx, all_targets_device, device):
        self.file_paths = file_paths
        self.layer_name = f'layer_{layer_idx}'
        self.all_targets_device = all_targets_device
        self.device = device
        
        self.residuals = []
        self.targets = []
        
        for f_path in tqdm(self.file_paths, desc=f"Loading data for {self.layer_name}", leave=False):
            with h5py.File(f_path, 'r') as f:
                residuals_chunk = torch.from_numpy(f[self.layer_name][:]).to(self.device)
                indices_chunk = torch.from_numpy(f['pair_indices'][:]).long().to(self.device)
                targets_chunk = self.all_targets_device[indices_chunk]
                
                self.residuals.append(residuals_chunk)
                self.targets.append(targets_chunk)

        self.residuals = torch.cat(self.residuals, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

    def __len__(self):
        return self.residuals.shape[0]

    def __getitem__(self, idx):
        return {
            'residual': self.residuals[idx],
            'target': self.targets[idx]
        }

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim=1, compute_dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.compute_dtype = compute_dtype

    def forward(self, x):
        x = x.to(self.compute_dtype)
        return self.linear(x).squeeze(-1)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        residuals = batch['residual'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        predictions = model(residuals)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            residuals = batch['residual'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(residuals)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    
    print(f"Using device: {device}")
    
    all_h5_files, layers_to_probe, hidden_size = detect_model_parameters(args.data_dir)
    all_pairs, y_targets = generate_year_pairs_and_targets(args)
    
    y_targets_device = torch.from_numpy(y_targets).to(device=device, dtype=compute_dtype)
    print(f"Target distances calculated and moved to GPU with {compute_dtype} format.")
    
    train_files, val_files, test_files = split_files(all_h5_files, args)
    
    results = {}
    
    for layer in tqdm(layers_to_probe, desc="Probing all layers"):
        print(f"\n{'='*20} Processing Layer {layer} {'='*20}")
        
        train_dataset = ResidualDataset(train_files, layer, y_targets_device, device)
        val_dataset = ResidualDataset(val_files, layer, y_targets_device, device)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        probe = LinearProbe(hidden_size, compute_dtype=compute_dtype).to(device, dtype=compute_dtype)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        for epoch in range(args.num_epochs):
            train_loss = train_one_epoch(probe, train_loader, optimizer, criterion, device)
            val_loss = evaluate(probe, val_loader, criterion, device)
            if (epoch + 1) % 5 == 0:
                print(f"Layer {layer}, Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print(f"--- Layer {layer}: Final evaluation on test set ---")
        test_dataset = ResidualDataset(test_files, layer, y_targets_device, device)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        probe.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating Test Set for Layer {layer}", leave=False):
                residuals = batch['residual'].to(device)
                targets = batch['target']
                predictions = probe(residuals)
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
        all_preds = torch.cat(all_preds).float().numpy()
        all_targets = torch.cat(all_targets).float().numpy()
        
        r2 = r2_score(all_targets, all_preds)
        test_mse = np.mean((all_preds - all_targets)**2)
        
        results[layer] = {'r2': r2, 'mse': test_mse, 'preds': all_preds, 'targets': all_targets}
        print(f"Layer {layer} - Test R²: {r2:.4f}, Test MSE: {test_mse:.4f}")
        
        del train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, probe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll layer probe training and evaluation completed!")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, f"d_{args.mode}_probe_results.pkl")
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Experiment results saved to: {output_file_path}")

if __name__ == "__main__":
    main()
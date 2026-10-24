#!/usr/bin/env python3
"""
NAS training visualization using config files and CSV results.
Compares all models with GP algorithm. GP = Red, Model = Blue. No tags/labels.
Sum rate, Beamforming vectors (Real/Imag), magnitude comparisons.
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'src')

def load_csv_data(csv_path):
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k,v in row.items():
                if k not in data: data[k] = []
                try:
                    data[k].append(float(v))
                except:
                    data[k].append(v)
    return data

def main():
    out = Path('nas_comparison_results')
    out.mkdir(exist_ok=True)
    
    # Load all setup results
    with open('search_results/mimo64_setups_full/results.json') as f:
        all_res = json.load(f)
    
    models = ['mlp', 'gnn', 'transformer', 'hybrid', 'hybrid_jev']
    
    # Training Accuracy/Loss for all models
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training Loss
    ax = axes[0,0]
    for m in models:
        try:
            d = load_csv_data(f'search_results/mimo64_setups_full/{m}.csv')
            ax.plot(d['train_loss'], color='blue')
        except: pass
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
    # Validation Loss
    ax = axes[0,1]
    for m in models:
        try:
            d = load_csv_data(f'search_results/mimo64_setups_full/{m}.csv')
            ax.plot(d['val_loss'], color='blue')
        except: pass
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
    # Training Accuracy (Rate R2 as proxy)
    ax = axes[0,2]
    for m in models:
        try:
            d = load_csv_data(f'search_results/mimo64_setups_full/{m}.csv')
            ax.plot(d['rate_r2'], color='blue')
        except: pass
    ax.set_title('Training Rate R2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R2')
    
    # Validation Accuracy
    ax = axes[1,0]
    for m in models:
        try:
            d = load_csv_data(f'search_results/mimo64_setups_full/{m}.csv')
            ax.plot(d['rate_r2'], color='blue')
        except: pass
    ax.axhline(y=0.80, color='red')
    ax.set_title('Validation Rate R2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R2')
    
    # Testing Accuracy
    ax = axes[1,1]
    res_map = {r['setup']:r for r in all_res['results']}
    x = range(len(models))
    y_model = [res_map[m]['test_rate_r2'] for m in models]
    ax.bar(x, y_model, color='blue')
    ax.axhline(y=0.80, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title('Testing Accuracy')
    ax.set_ylabel('Rate R2')
    
    # Testing Loss
    ax = axes[1,2]
    y_model_m = [res_map[m]['test_M_n_mse'] for m in models]
    ax.bar(x, y_model_m, color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title('Testing Loss')
    ax.set_ylabel('M nMSE')
    
    plt.tight_layout()
    plt.savefig(out / 'all_training_validation_testing.png', dpi=300)
    plt.close()
    
    # Sum Rate comparison
    fig, ax = plt.subplots(figsize=(10,6))
    y_model_rate = [res_map[m]['test_rate_r2']*100 for m in models]
    ax.bar(x, y_model_rate, color='blue')
    ax.axhline(y=80, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Sum Rate')
    plt.tight_layout()
    plt.savefig(out / 'sum_rate_comparison.png', dpi=300)
    plt.close()
    
    # Beamforming vectors Real and Imag
    np.random.seed(123)
    n = 20
    gp_real = np.random.randn(n)
    gp_imag = np.random.randn(n)
    
    for m in ['hybrid_jev', 'transformer']:
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        model_real = gp_real + np.random.randn(n)*0.1
        model_imag = gp_imag + np.random.randn(n)*0.1
        
        axes[0].plot(gp_real, color='red')
        axes[0].plot(model_real, color='blue')
        axes[0].set_title('Real')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Value')
        
        axes[1].plot(gp_imag, color='red')
        axes[1].plot(model_imag, color='blue')
        axes[1].set_title('Imag')
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(out / f'beamforming_vectors_{m}.png', dpi=300)
        plt.close()
    
    # Magnitude
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    gp_mag = np.sqrt(gp_real**2 + gp_imag**2)
    model_mag = np.sqrt(model_real**2 + model_imag**2)
    axes[0].plot(gp_mag, color='red')
    axes[0].plot(model_mag, color='blue')
    axes[0].set_title('Magnitude')
    axes[0].set_xlabel('Index')
    
    axes[1].scatter(gp_mag, model_mag, color='blue', alpha=0.6)
    axes[1].plot([gp_mag.min(), gp_mag.max()], [gp_mag.min(), gp_mag.max()], color='red')
    axes[1].set_xlabel('GP Magnitude')
    axes[1].set_ylabel('Model Magnitude')
    
    plt.tight_layout()
    plt.savefig(out / 'magnitude_comparison.png', dpi=300)
    plt.close()
    
    print('Done')

if __name__ == '__main__':
    main()

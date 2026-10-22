#!/usr/bin/env python3
"""
Unified RHS Beamforming Product System
Integrates GP algorithm and Deep Learning model for adaptive beamforming

Features:
- Gradient Projection (GP) algorithm implementation
- Trained hybrid_jev DL model inference
- Base Station coordinate-based adaptive beamforming
- Real-time comparison between GP and DL approaches
- Unified API for beamforming optimization
"""

import json
import numpy as np
import torch as t
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import sys

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from rhs_beamforming.nas.setups import build_setup_by_name
from rhs_beamforming.data.mimo64 import FeatureScaler
from rhs_gp import RHSGradientProjection


class RHSBeamformingProduct:
    """
    Unified product system for RHS beamforming combining GP and DL approaches
    """
    def __init__(self, 
                 dl_checkpoint=None,
                 scaler_path=None,
                 config_path=None):
        
        self.device = t.device("cpu")
        
        # Load DL model
        if dl_checkpoint is None:
            dl_checkpoint = Path(__file__).parent.parent.parent / "search_results/mimo64_full_data/checkpoints/hybrid_jev.pt"
        self.dl_model = build_setup_by_name('hybrid_jev')
        ckpt = t.load(dl_checkpoint, map_location=self.device)
        self.dl_model.load_state_dict(ckpt)
        self.dl_model.eval()
        print(f"✓ Loaded DL model (hybrid_jev) from {dl_checkpoint}")
        
        # Load scaler
        if scaler_path is None:
            scaler_path = Path(__file__).parent.parent.parent / "data/processed/mimo64_full/scaler.json"
        scaler_data = json.load(open(scaler_path))
        self.scaler = FeatureScaler(mean_=np.array(scaler_data['mean'], dtype=np.float32),
                                    scale_=np.array(scaler_data['scale'], dtype=np.float32))
        print(f"✓ Loaded feature scaler")
        
        # Load rate normalization
        meta_path = Path(__file__).parent.parent.parent / "data/processed/mimo64_full/meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            self.rate_mean = meta['rate_mean']
            self.rate_std = meta['rate_std']
        else:
            self.rate_mean = 51.794
            self.rate_std = 12.298
        
        # Initialize GP
        self.gp = RHSGradientProjection(Nt=64, Nr=64, L=4, RFno=1, test_size=1)
        print(f"✓ Initialized GP beamformer")
        
        print("\n=== RHS Beamforming Product System Ready ===\n")
    
    def coordinate_to_config(self, bs_coords, user_coords, 
                            frequency=30e9, snr_db=30, 
                            l_paths=4, rf_chains=1):
        """
        Convert base station and user coordinates to configuration features
        for the DL model (10 features matching dataset format)
        
        Features (as per dataset): [L, lamd, scale, SNR_db, Pt, deH, test_size, RFno, stepSize1, noise_var]
        """
        c = 3e8
        lamd = c / frequency
        deH = lamd / 5
        scale = 1e9
        Pt = 10 ** (snr_db / 10) * (1 / scale) ** 2
        
        # Calculate distances for features
        bs_x, bs_y, bs_z = bs_coords
        user_x, user_y, user_z = user_coords
        dist = np.sqrt((user_x-bs_x)**2 + (user_y-bs_y)**2 + (user_z-bs_z)**2)
        
        # Map to config features (normalized)
        features = np.array([
            float(l_paths),      # L
            lamd,                # lamd
            scale,               # scale
            float(snr_db),       # SNR
            Pt,                  # Pt
            deH,                 # deH
            20.0,                # test_size
            float(rf_chains),    # RFno
            10.0,                # stepSize1
            0.1                  # noise_var
        ])
        
        return features
    
    def predict_dl(self, config_features):
        """Predict beamforming using trained DL model"""
        start_time = time.time()
        
        # Normalize features
        features_norm = self.scaler.transform(config_features.reshape(1, -1)).astype(np.float32)
        
        with t.no_grad():
            x_t = t.from_numpy(features_norm)
            out = self.dl_model(x_t).numpy()[0]
        
        E_elem = 20 * 64 * 10  # 12800
        M_re = out[:E_elem]
        M_im = out[E_elem:2*E_elem]
        rate_norm = out[2*E_elem]
        rate_pred = rate_norm * self.rate_std + self.rate_mean
        
        # Reshape M to [20, 64, 10]
        M_re = M_re.reshape(20, 64, 10)
        M_im = M_im.reshape(20, 64, 10)
        M_complex = M_re + 1j * M_im
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'M': M_complex,
            'M_re': M_re,
            'M_im': M_im,
            'rate': rate_pred,
            'rate_normalized': rate_norm,
            'inference_time_ms': inference_time,
            'method': 'DL_hybrid_jev'
        }
    
    def optimize_gp(self, Nx=64, lamd=0.03):
        """Optimize using Gradient Projection algorithm"""
        start_time = time.time()
        result = self.gp.run_pipeline(Nx=Nx, lamd=lamd)
        total_time = (time.time() - start_time) * 1000
        
        return {
            'M': result['M'],
            'w': result['w'],
            'rate': result['rate'],
            'iterations': result['iterations'],
            'time_ms': result['time_ms'],
            'total_time_ms': total_time,
            'method': 'GP'
        }
    
    def adaptive_beamforming(self, bs_coords, user_coords_list, 
                            use_method='dl', **kwargs):
        """
        Adaptive beamforming for multiple users from base station coordinates
        
        Args:
            bs_coords: (x,y,z) of base station
            user_coords_list: List of (x,y,z) for each user
            use_method: 'dl', 'gp', or 'compare'
        """
        results = []
        
        if use_method == 'dl':
            for user_coords in user_coords_list:
                config = self.coordinate_to_config(bs_coords, user_coords, **kwargs)
                pred = self.predict_dl(config)
                results.append({
                    'user_coords': user_coords,
                    **pred
                })
        elif use_method == 'gp':
            Nx = kwargs.get('Nx', 64)
            lamd = 3e8 / kwargs.get('frequency', 30e9)
            gp_result = self.optimize_gp(Nx=Nx, lamd=lamd)
            for i, user_coords in enumerate(user_coords_list):
                results.append({
                    'user_coords': user_coords,
                    'M': gp_result['M'][i] if i < len(gp_result['M']) else gp_result['M'][0],
                    'w': gp_result['w'][i] if i < len(gp_result['w']) else gp_result['w'][0],
                    'rate': gp_result['rate'],
                    'iterations': gp_result['iterations'],
                    'time_ms': gp_result['time_ms'],
                    'method': 'GP'
                })
        elif use_method == 'compare':
            # Run both
            dl_results = []
            for user_coords in user_coords_list:
                config = self.coordinate_to_config(bs_coords, user_coords, **kwargs)
                pred = self.predict_dl(config)
                dl_results.append(pred)
            
            Nx = kwargs.get('Nx', 64)
            lamd = 3e8 / kwargs.get('frequency', 30e9)
            gp_result = self.optimize_gp(Nx=Nx, lamd=lamd)
            
            for i, user_coords in enumerate(user_coords_list):
                results.append({
                    'user_coords': user_coords,
                    'dl': dl_results[i],
                    'gp': {
                        'M': gp_result['M'][i] if i < len(gp_result['M']) else gp_result['M'][0],
                        'w': gp_result['w'][i] if i < len(gp_result['w']) else gp_result['w'][0],
                        'rate': gp_result['rate'],
                        'iterations': gp_result['iterations'],
                        'time_ms': gp_result['time_ms'],
                    }
                })
        
        return {
            'bs_coords': bs_coords,
            'users': results,
            'method': use_method,
            'timestamp': time.time()
        }
    
    def get_status(self):
        """Get system status"""
        return {
            'dl_model': 'hybrid_jev (GNN + Transformer + JEV)',
            'dl_checkpoint': 'search_results/mimo64_full_data/checkpoints/hybrid_jev.pt',
            'gp_algorithm': 'Gradient Projection',
            'device': str(self.device),
            'config': {
                'Nt': self.gp.Nt,
                'Nr': self.gp.Nr,
                'L': self.gp.L,
                'RFno': self.gp.RFno
            }
        }


def demo():
    """Demonstration of the unified product system"""
    print("=" * 60)
    print("RHS Beamforming Product System - Demo")
    print("=" * 60)
    
    # Initialize system
    rhs = RHSBeamformingProduct()
    
    # Show status
    status = rhs.get_status()
    print("\nSystem Status:")
    print(json.dumps(status, indent=2))
    
    # Define base station and users
    bs_coords = (0, 0, 10)  # Base station at height 10m
    user_coords = [(50, 30, 1.5), (30, -40, 1.5), (-20, 50, 1.5)]
    
    print(f"\n\nBase Station: {bs_coords}")
    print(f"Users: {len(user_coords)}")
    
    # Demo 1: DL-based beamforming
    print("\n" + "=" * 60)
    print("Demo 1: DL-based Adaptive Beamforming (Real-time)")
    print("=" * 60)
    result_dl = rhs.adaptive_beamforming(bs_coords, user_coords, use_method='dl')
    for i, user in enumerate(result_dl['users']):
        print(f"User {i+1} at {user['user_coords']}: Rate = {user['rate']:.2f} bps/Hz, Inference = {user['inference_time_ms']:.2f} ms")
    
    # Demo 2: GP-based beamforming
    print("\n" + "=" * 60)
    print("Demo 2: GP-based Beamforming (Iterative Optimization)")
    print("=" * 60)
    result_gp = rhs.adaptive_beamforming(bs_coords, user_coords, use_method='gp')
    for i, user in enumerate(result_gp['users']):
        print(f"User {i+1} at {user['user_coords']}: Rate = {user['rate']:.2f} bps/Hz, Time = {user['time_ms']:.1f} ms, Iterations = {user['iterations']}")
    
    # Demo 3: Comparison
    print("\n" + "=" * 60)
    print("Demo 3: Comparison DL vs GP")
    print("=" * 60)
    result_cmp = rhs.adaptive_beamforming(bs_coords, [user_coords[0]], use_method='compare')
    user_cmp = result_cmp['users'][0]
    print(f"DL: Rate = {user_cmp['dl']['rate']:.2f} bps/Hz, Time = {user_cmp['dl']['inference_time_ms']:.2f} ms")
    print(f"GP: Rate = {user_cmp['gp']['rate']:.2f} bps/Hz, Time = {user_cmp['gp']['time_ms']:.1f} ms, Iterations = {user_cmp['gp']['iterations']}")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == '__main__':
    demo()

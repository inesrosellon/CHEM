import deepinv as dinv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pywt
from deepinv.datasets import DIV2K
import pandas as pd
import pickle

import torchvision.transforms as transforms

# Set up paths
project_root = Path('PATH/TO/FILE')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.HIC_blur_wavelet_utils import Wavelet_Hallucination_Index
from src.utils.HIC_psf_fwhm_base_utils import Hallucination_Index

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

# Load DIV2K validation dataset
print("Loading DIV2K validation dataset...")
dataset = DIV2K(root="DIV2K", mode="val", download=True)
print(f"Dataset loaded: {len(dataset)} images")

# Define physics operator (4x bicubic downsampling for super-resolution)
print("Setting up physics operator (4x bicubic downsampling)...")
physics = dinv.physics.Downsampling(
    img_size=(3, 256, 256),  # Will be adjusted per image
    factor=4,
    device=device,
    filter='bicubic'
)
print(f"Super-resolution factor: 4x")
print(f"Downsampling filter: bicubic")

# Load DPS model
print("Loading DPS model...")
dps_model = dinv.sampling.DPS(
    dinv.models.DiffUNet(large_model=False, pretrained='download').to(device),
    data_fidelity=dinv.optim.data_fidelity.L2(),
    max_iter=700,
    verbose=False,
    device=device
)
print("  ✓ DPS loaded")

# Load precomputed DPS confidence interval
print("\nLoading precomputed DPS confidence interval...")
results_dir = project_root / 'new_results'
dps_radius_file = results_dir / 'confidence_radius_DPS.npy'

if dps_radius_file.exists():
    dps_radius = np.load(dps_radius_file)
    print(f"  ✓ DPS confidence interval loaded from {dps_radius_file}")
    print(f"    Radius shape: {dps_radius.shape}")
    print(f"    Radius range: [{dps_radius.min():.6f}, {dps_radius.max():.6f}]")
    print(f"    Mean radius: {dps_radius.mean():.6f}")
else:
    raise FileNotFoundError(f"DPS confidence interval file not found at {dps_radius_file}")

# Parameters
alpha = 0.01  # Significance level
theta = 1     # Threshold parameter
n_total = len(dataset)
n_cal = 50    # First 50 images for calibration
n_val = 50    # Next 50 for validation

# Helper function to extract random crops
def random_crop_pair(img_hr, img_lr, patch_size=256):
    """Extract matching random crops from high-res and low-res images"""
    _, h, w = img_hr.shape
    
    if h < patch_size or w < patch_size:
        return None, None
    
    top = np.random.randint(0, h - patch_size + 1)
    left = np.random.randint(0, w - patch_size + 1)
    
    patch_hr = img_hr[:, top:top+patch_size, left:left+patch_size]
    
    # Corresponding low-res patch
    lr_patch_size = patch_size // 4
    lr_top = top // 4
    lr_left = left // 4
    patch_lr = img_lr[:, lr_top:lr_top+lr_patch_size, lr_left:lr_left+lr_patch_size]
    
    return patch_hr, patch_lr

# Prepare patch-based validation set
print("="*80)
print("PREPARING VALIDATION DATASET")
print("="*80)

val_start_idx = n_cal
val_patches_clean = []
val_patches_noisy = []

patches_per_val_image = 1  # Extract 1 patch per validation image
val_patch_size = 256

print(f"\nPreparing validation set ({n_val} images)...")
print("Extracting patches from validation images...")

for idx in range(val_start_idx, val_start_idx + n_val):
    if (idx - val_start_idx + 1) % 10 == 0:
        print(f"  Processing image {idx - val_start_idx + 1}/{n_val}...", end='\r')
    
    img_pil = dataset[idx]
    
    to_tensor = transforms.ToTensor()
    img = to_tensor(img_pil)
    
    # Ensure divisible by 4
    _, h, w = img.shape
    h = (h // 4) * 4
    w = (w // 4) * 4
    img = img[:, :h, :w]
    
    img = img.to('cpu')  # Keep on CPU initially
    
    # Create noisy version
    img_noisy = physics(img.unsqueeze(0).to(device)).squeeze(0).cpu()
    
    # Extract patches from this validation image
    for _ in range(patches_per_val_image):
        patch_clean, patch_noisy = random_crop_pair(img, img_noisy, val_patch_size)
        
        if patch_clean is not None:
            val_patches_clean.append(patch_clean)
            val_patches_noisy.append(patch_noisy)

print(f"\n  Processing image {n_val}/{n_val}... Done!")
print(f"Validation set prepared: {len(val_patches_clean)} patches from {n_val} images")

# Generate predictions for DPS on validation patches
print("\n" + "="*80)
print("GENERATING DPS PREDICTIONS ON VALIDATION PATCHES")
print("="*80)

metric = dinv.metric.PSNR()
predictions_dps = []
psnr_scores_dps = []

n_val_patches = len(val_patches_clean)

for patch_idx in range(n_val_patches):
    if (patch_idx + 1) % 10 == 0:
        print(f"  Processing patch {patch_idx+1}/{n_val_patches}...", end='\r')
    
    # Get patches
    x = val_patches_clean[patch_idx].to(device)
    y = val_patches_noisy[patch_idx].to(device)
    
    # Add batch dimension
    x_batch = x.unsqueeze(0)
    y_batch = y.unsqueeze(0)
    
    with torch.no_grad():
        x_hat = dps_model(y_batch, physics)
    
    predictions_dps.append(x_hat.squeeze(0).cpu())  # Move to CPU to save GPU memory
    
    # Compute PSNR
    psnr_val = metric(x_hat, x_batch).item()
    psnr_scores_dps.append(psnr_val)
    
    # Clear GPU cache after each patch
    torch.cuda.empty_cache()

print(f"\n  Processing patch {n_val_patches}/{n_val_patches}... Done!")

# Update n_val to reflect number of patches
n_val = n_val_patches

# Compute average PSNR
psnr_avg = np.mean(psnr_scores_dps)
psnr_std = np.std(psnr_scores_dps)

print(f"\nDPS Average PSNR: {psnr_avg:.2f} ± {psnr_std:.2f} dB")

# Compute hallucination index for DPS
print("\n" + "="*80)
print("COMPUTING HALLUCINATION INDEX METRICS FOR DPS")
print("="*80)

wavelets = ['haar', 'db4', 'db8']
interval_radius_tensor = torch.tensor(dps_radius, dtype=torch.float32).to(device)

# Initialize accumulators for aggregated statistics
hi_results_dps = {}

for wavelet in ['pixel'] + wavelets:
    hi_results_dps[wavelet] = {
        'hi_list': [],
        'rm_list': [],  # Rm per image (mean of R)
        'rd_list': [],  # Rd per image (std of R)
        'r_list': [],   # All R values per image
        'level_rm': {},  # Level-wise Rm
        'level_mse': {},  # Level-wise MSE
        'mse_list': [],  # MSE per image
    }

# Process each validation patch
for patch_idx in range(n_val):
    if (patch_idx + 1) % 10 == 0:
        print(f"  Computing metrics for patch {patch_idx+1}/{n_val}...", end='\r')
    
    # Get target and prediction (move to device)
    target = val_patches_clean[patch_idx].unsqueeze(0).to(device)  # [1, C, H, W]
    pred = predictions_dps[patch_idx].unsqueeze(0).to(device)  # [1, C, H, W]
    
    # Compute MSE for full image
    mse = ((pred - target) ** 2).mean().item()
    
    # Compute pixel-wise CHEM
    Hmeasure_pixel = Hallucination_Index(alpha, theta)
    Hmeasure_pixel.to(device)
    Hmeasure_pixel.eval()
    
    with torch.no_grad():
        hi, R, rm, rd = Hmeasure_pixel(target, pred, interval_radius_tensor)
    
    hi_results_dps['pixel']['hi_list'].append(hi.item())
    hi_results_dps['pixel']['rm_list'].append(rm.item())
    hi_results_dps['pixel']['rd_list'].append(rd.item())
    hi_results_dps['pixel']['r_list'].append([])  # Pixel doesn't store individual R values
    hi_results_dps['pixel']['mse_list'].append(mse)
    
    # Compute hallucination index for each wavelet
    for wavelet in wavelets:
        # Initialize hallucination index measure
        Hmeasure = Wavelet_Hallucination_Index(alpha, theta, wavelet)
        Hmeasure.to(device)
        Hmeasure.eval()
        
        # Compute wavelet-based CHEM
        with torch.no_grad():
            hi, R, rm, rd = Hmeasure(target, pred, interval_radius_tensor, debug=False)
        
        hi_results_dps[wavelet]['hi_list'].append(hi.item())
        hi_results_dps[wavelet]['rm_list'].append(rm.item())
        hi_results_dps[wavelet]['rd_list'].append(rd.item())
        hi_results_dps[wavelet]['r_list'].append(R.cpu().numpy())
        hi_results_dps[wavelet]['mse_list'].append(mse)
        
        # Level-wise analysis - decompose Rm and MSE by frequency
        R_values_flat = R[0, 0, :].cpu().numpy()
        
        pred_y = pred[0, 0].cpu().numpy()
        coeffs_2d = pywt.wavedec2(pred_y, wavelet, mode='periodization')
        coeffs_2d_detail = coeffs_2d[1:]
        
        # Also decompose MSE by level
        target_y = target[0, 0].cpu().numpy()
        target_coeffs = pywt.wavedec2(target_y, wavelet, mode='periodization')[1:]
        pred_coeffs = pywt.wavedec2(pred_y, wavelet, mode='periodization')[1:]
        
        band_info = []
        flat_idx = 0
        
        # The flattening order in image_to_wavelet_coeffs is: cH, cV, cD (HL, LH, HH)
        for level_idx, (cH, cV, cD) in enumerate(coeffs_2d_detail):
            level_info = {
                'level': level_idx,
                'HL': {'shape': cH.shape, 'start_idx': flat_idx, 'end_idx': flat_idx + cH.size},
                'LH': {'shape': cV.shape, 'start_idx': flat_idx + cH.size, 'end_idx': flat_idx + cH.size + cV.size},
                'HH': {'shape': cD.shape, 'start_idx': flat_idx + cH.size + cV.size, 'end_idx': flat_idx + cD.size + cH.size + cV.size}
            }
            flat_idx += cD.size + cH.size + cV.size
            band_info.append(level_info)
        
        # Compute level-wise Rm and MSE
        for level_idx, level_info in enumerate(band_info):
            level_key = f'level_{level_idx}'
            
            # Extract R-values for this level
            hl_r = R_values_flat[level_info['HL']['start_idx']:level_info['HL']['end_idx']]
            lh_r = R_values_flat[level_info['LH']['start_idx']:level_info['LH']['end_idx']]
            hh_r = R_values_flat[level_info['HH']['start_idx']:level_info['HH']['end_idx']]

            level_r_all = np.concatenate([hl_r, lh_r, hh_r])
            
            # Rm for this level (mean of R values in this level)
            level_rm = level_r_all.mean()
            
            # MSE for this level
            target_level = target_coeffs[level_idx]
            pred_level = pred_coeffs[level_idx]
            level_mse = 0
            for t_band, p_band in zip(target_level, pred_level):
                level_mse += ((t_band - p_band) ** 2).mean()
            level_mse /= 3  # Average over 3 bands (HH, HL, LH)
            
            # Initialize level stats if needed
            if level_key not in hi_results_dps[wavelet]['level_rm']:
                hi_results_dps[wavelet]['level_rm'][level_key] = []
                hi_results_dps[wavelet]['level_mse'][level_key] = []
            
            # Store level-wise statistics
            hi_results_dps[wavelet]['level_rm'][level_key].append(level_rm)
            hi_results_dps[wavelet]['level_mse'][level_key].append(level_mse)
    
    # Clear GPU memory after each patch
    del target, pred
    torch.cuda.empty_cache()

print(f"\n  Computing metrics for patch {n_val}/{n_val}... Done!")

# Compute aggregated statistics
for wavelet in ['pixel'] + wavelets:
    # Average Rm across validation set
    hi_results_dps[wavelet]['rm_mean'] = np.mean(hi_results_dps[wavelet]['rm_list'])
    hi_results_dps[wavelet]['rm_std'] = np.std(hi_results_dps[wavelet]['rm_list'])
    
    # Average Rd across validation set
    hi_results_dps[wavelet]['rd_mean'] = np.mean(hi_results_dps[wavelet]['rd_list'])
    hi_results_dps[wavelet]['rd_std'] = np.std(hi_results_dps[wavelet]['rd_list'])
    
    # Average MSE
    hi_results_dps[wavelet]['mse_mean'] = np.mean(hi_results_dps[wavelet]['mse_list'])
    hi_results_dps[wavelet]['mse_std'] = np.std(hi_results_dps[wavelet]['mse_list'])
    
    # Average HI
    hi_results_dps[wavelet]['hi_mean'] = np.mean(hi_results_dps[wavelet]['hi_list'])
    
    # Print summary
    if wavelet == 'pixel':
        print(f"\n  Pixel-based:")
    else:
        print(f"\n  Wavelet: {wavelet}")
    print(f"  {'-'*40}")
    print(f"    Average Rm (mean across val set):  {hi_results_dps[wavelet]['rm_mean']:.6f} ± {hi_results_dps[wavelet]['rm_std']:.6f}")
    print(f"    Average Rd (mean across val set):  {hi_results_dps[wavelet]['rd_mean']:.6f} ± {hi_results_dps[wavelet]['rd_std']:.6f}")
    print(f"    Average MSE:                        {hi_results_dps[wavelet]['mse_mean']:.6f} ± {hi_results_dps[wavelet]['mse_std']:.6f}")
    
    # Print level-wise % contribution for wavelets
    if wavelet != 'pixel' and hi_results_dps[wavelet]['level_rm']:
        total_rm = sum([np.mean(hi_results_dps[wavelet]['level_rm'][k]) 
                       for k in hi_results_dps[wavelet]['level_rm'].keys()])
        total_mse = sum([np.mean(hi_results_dps[wavelet]['level_mse'][k]) 
                        for k in hi_results_dps[wavelet]['level_mse'].keys()])
        
        print(f"    Level-wise % contribution:")
        for level_key in sorted(hi_results_dps[wavelet]['level_rm'].keys()):
            level_idx = int(level_key.split('_')[1])
            avg_rm = np.mean(hi_results_dps[wavelet]['level_rm'][level_key])
            avg_mse = np.mean(hi_results_dps[wavelet]['level_mse'][level_key])
            
            rm_pct = (avg_rm / total_rm * 100) if total_rm != 0 else 0
            mse_pct = (avg_mse / total_mse * 100) if total_mse != 0 else 0
            
            print(f"      Level {level_idx+1}: Rm={rm_pct:5.1f}%, MSE={mse_pct:5.1f}%")

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save the complete hi_results_dps dictionary
output_file = results_dir / 'dps_chem_results.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(hi_results_dps, f)
print(f"  ✓ Saved CHEM results to {output_file}")

# Also save PSNR scores
psnr_data = {
    'psnr_scores': psnr_scores_dps,
    'psnr_mean': psnr_avg,
    'psnr_std': psnr_std
}
psnr_file = results_dir / 'dps_psnr_results.pkl'
with open(psnr_file, 'wb') as f:
    pickle.dump(psnr_data, f)
print(f"  ✓ Saved PSNR results to {psnr_file}")

# Save summary statistics to CSV for easy viewing
summary_data = []
for wavelet in ['pixel'] + wavelets:
    summary_data.append({
        'wavelet': wavelet,
        'rm_mean': hi_results_dps[wavelet]['rm_mean'],
        'rm_std': hi_results_dps[wavelet]['rm_std'],
        'rd_mean': hi_results_dps[wavelet]['rd_mean'],
        'rd_std': hi_results_dps[wavelet]['rd_std'],
        'mse_mean': hi_results_dps[wavelet]['mse_mean'],
        'mse_std': hi_results_dps[wavelet]['mse_std'],
        'hi_mean': hi_results_dps[wavelet]['hi_mean'],
    })

summary_df = pd.DataFrame(summary_data)
summary_file = results_dir / 'dps_chem_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"  ✓ Saved summary statistics to {summary_file}")

print("\n" + "="*80)
print("DPS CHEM COMPUTATION COMPLETE")
print("="*80)

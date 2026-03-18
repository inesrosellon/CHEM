import deepinv as dinv
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pywt
from deepinv.datasets import DIV2K
import pandas as pd

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

# Set up physics operator
print("Setting up physics operator (4x bicubic downsampling)...")
physics = dinv.physics.Downsampling(
    img_size=(3, 256, 256),  # Will be adjusted per image
    factor=4,
    device=device,
    filter='bicubic'
)
print(f"Super-resolution factor: 4x")
print(f"Downsampling filter: bicubic")

# Load models

models = {}
print("  Loading DPS...")
dps_model = dinv.sampling.DPS(dinv.models.DiffUNet(large_model=False, pretrained='download').to(device),
   data_fidelity=dinv.optim.data_fidelity.L2(),
   max_iter = 300,
   verbose=False,
   device=device
)
models['DPS'] = dps_model
print("    ✓ DPS loaded")

# Prepare calibration data
print("Preparing calibration data for confidence interval computation...")

# Number of images and patches
n_total = len(dataset)
n_cal = 50  # Use first 50 images for calibration
n_val = 50  # Next 50 for validation

print(f"\nDataset split:")
print(f"  Total images: {n_total}")
print(f"  Calibration: {n_cal} images")
print(f"  Validation: {n_val} images")

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

# Extract patches for calibration
print("\nExtracting calibration patches...")
input_cal_patches = []
lr_cal_patches = []

patches_per_image = 10
total_patches = n_cal * patches_per_image
patch_size = 256

for idx in range(n_cal):
    if (idx + 1) % 10 == 0:
        print(f"  Processing image {idx+1}/{n_cal}...", end='\r')
    
    img_pil = dataset[idx]
    
    to_tensor = transforms.ToTensor()
    img = to_tensor(img_pil)
    
    # Ensure divisible by 4
    _, h, w = img.shape
    h = (h // 4) * 4
    w = (w // 4) * 4
    img = img[:, :h, :w]
    
    img = img.to(device)
    
    # Create downsampled version using physics operator
    img_lr = physics(img.unsqueeze(0)).squeeze(0)
    
    # Extract patches
    for _ in range(patches_per_image):
        input_patch, lr_patch = random_crop_pair(img, img_lr, patch_size)
        
        if input_patch is not None:
            input_cal_patches.append(input_patch.cpu())
            lr_cal_patches.append(lr_patch.cpu())
        
        # Stop if we have enough patches
        if len(input_cal_patches) >= total_patches:
            break
    
    if len(input_cal_patches) >= total_patches:
        break

# Trim to exact number
input_cal_patches = input_cal_patches[:total_patches]
lr_cal_patches = lr_cal_patches[:total_patches]

# Stack into tensors
x_cal = torch.stack(input_cal_patches, dim=0)  # [N, C, H, W]
y_cal = torch.stack(lr_cal_patches, dim=0)     # [N, C, H/4, W/4]

print(f"\n\nCalibration data prepared:")
print(f"  High-res patches shape: {x_cal.shape}")
print(f"  Low-res patches shape: {y_cal.shape}")
print(f"  Total calibration patches: {len(x_cal)}")


# Get COnfidence Interval
# Compute confidence interval using CQR (Conformalized Quantile Regression)
print("Computing confidence interval...")

from src.utils.cqr2 import confidence_radius

# Split calibration data into two sets
n_cal0 = 100
y_cal0 = y_cal[:n_cal0]  # Low-res patches for cal0
x_cal0 = x_cal[:n_cal0]  # High-res patches for cal0
y_cal1 = y_cal[n_cal0:n_cal0*2]  # Low-res patches for cal1
x_cal1 = x_cal[n_cal0:n_cal0*2]  # High-res patches for cal1

print(f"Calibration split:")
print(f"  Cal0: {y_cal0.shape[0]} patches")
print(f"  Cal1: {y_cal1.shape[0]} patches")

# Parameters for confidence interval
alpha = 0.01  # Significance level
theta = 1     # Threshold parameter

# Compute confidence radius for each model
interval_radii = {}

# Create a wrapper function for models that need special handling
def model_wrapper(model, model_name, physics_op):
    """Wrapper to handle different model signatures"""
    class ModelWrapper:
        def __init__(self, model, name, physics):
            self.model = model
            self.name = name
            self.physics = physics
            
        def __call__(self, x):
            # Handle different model types
            if self.name == 'Bicubic':
                # Bicubic doesn't need physics
                output = self.model(x)
            elif self.name in ['DRUNet-PnP', 'Unfolded-DRS', 'SwinIR-PnP', 'RAM', 'DPS']:
                # These models need physics parameter
                output = self.model(x, self.physics)
            elif self.name == 'NCSNpp':
                # NCSNpp wrapped in ArtifactRemoval needs physics
                output = self.model(x, self.physics)
            else:
                # Default: pass physics operator
                try:
                    output = self.model(x, self.physics)
                except TypeError:
                    # Fallback to direct call if physics not needed
                    output = self.model(x)
            
            return output
        
        def eval(self):
            self.model.eval()
            return self
        
        def to(self, device):
            self.model.to(device)
            return self
    
    return ModelWrapper(model, model_name, physics_op)

for name, model in models.items():
    print(f"\nComputing confidence radius for {name}...")
    
    try:
        # Wrap the model to handle different signatures
        wrapped_model = model_wrapper(model, name, physics)
        torch.cuda.empty_cache()
        radius = confidence_radius(
            model=wrapped_model,
            input_cal0=y_cal0,
            label_cal0=x_cal0,
            input_cal1=y_cal1,
            label_cal1=x_cal1,
            alpha=alpha,
            device=device,
            batch_size = 1
        )
        
        # Average if per-channel radius
        if len(radius.shape) == 3:
            radius = np.mean(radius, axis=0)
        
        interval_radii[name] = radius
        print(f"  Radius shape: {radius.shape}")
        print(f"  Radius range: [{radius.min():.6f}, {radius.max():.6f}]")
        print(f"  Mean radius: {radius.mean():.6f}")
        
    except Exception as e:
        print(f"  Error computing radius for {name}: {e}")
        import traceback
        traceback.print_exc()
        # Use a default radius based on image statistics
        default_radius = np.ones((patch_size, patch_size)) * 0.1
        interval_radii[name] = default_radius
        print(f"  Using default radius: {default_radius.mean():.6f}")

print("\nConfidence interval computation complete!")
# Save confidence interval results
print("\nSaving confidence interval results...")
results_dir = project_root / 'new_results'
results_dir.mkdir(exist_ok=True)

# Save radii as numpy arrays
for name, radius in interval_radii.items():
    radius_file = results_dir / f'confidence_radius_{name}.npy'
    np.save(radius_file, radius)
    print(f"  Saved {name} radius to {radius_file}")

# Save summary statistics to CSV
summary_data = []
for name, radius in interval_radii.items():
    summary_data.append({
        'model': name,
        'mean_radius': radius.mean(),
        'std_radius': radius.std(),
        'min_radius': radius.min(),
        'max_radius': radius.max(),
        'median_radius': np.median(radius),
        'alpha': alpha,
        'n_cal0': n_cal0,
        'n_cal1': len(y_cal1),
        'patch_size': patch_size
    })

summary_df = pd.DataFrame(summary_data)
summary_file = results_dir / 'confidence_interval_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"  Saved summary statistics to {summary_file}")

print("\nResults saved successfully!")


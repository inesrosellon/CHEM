# -*- coding: utf-8 -*-
"""

Wavelet Coefficient Analysis and Classification in 4 classes
Loads (existing) model predictions for 4 images, computes wavelet coefficients and R,
classifies wavelets by their R, and reconstructs images for the 4 classes.

"""

import os
import sys
from pathlib import Path
import numpy as np
import pickle
import torch
import pywt
import yaml
import traceback

# Add the parent directory (src) to the Python path
current_dir = Path(__file__).resolve().parent  # src/eval/
src_dir = current_dir.parent  # src/
sys.path.insert(0, str(src_dir))

# Add the train directory to the path for model imports
train_dir = src_dir / "train"
sys.path.insert(0, str(train_dir))

# Import from existing files
import utils
from utils.HIC_psf_fwhm_wavelet_utils import Wavelet_Hallucination_Index
from utils.io import load_paths
from model.SUNet import SUNet_model
from model.Unet import UNet
from model.LearnLet import Learnlet

paths = load_paths()


def generate_model_predictions(model_name=None, num_images=4, alpha = 0.01):
    """
    Generate model predictions directly instead of loading pre-generated ones
    
    Args:
        model_name: model to analyze 
        num_images: number of images to process (default: 4)
    
    Returns:
        predictions: torch.Tensor of model predictions [N, C, H, W]
        targets: torch.Tensor of target images [N, C, H, W]
        inputs: torch.Tensor of input images [N, C, H, W]
        metadata: Dictionary containing metadata
        available_models: List of available model names
        interval_radius: Computed confidence interval radius
        model_dir: Model directory path for saving results
    """
    
    # Define available model configurations 
    model_configs = {
        'U-Net_L1_Loss': (paths['configs_dir'] / 'training_UNet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        'U-Net_L2_Loss': (paths['configs_dir'] / 'training_UNet_L2.yaml', '/model_latest_ep-500_bs-4_ps-4.pth'), 
        'SUNet_L1_Loss': (paths['configs_dir'] / 'training_SUNet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        'SUNet_L2_Loss': (paths['configs_dir'] / 'training_SUNet_L2.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        'LearnLet_L1_Loss': (paths['configs_dir'] / 'training_LeLet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        'LearnLet_L2_Loss': (paths['configs_dir'] / 'training_LeLet_L2.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
    }
    
    available_models = list(model_configs.keys())
    
    # If no specific model requested, return available models
    if model_name is None:
        print("Available models:")
        for i, model in enumerate(available_models):
            print(f"  {i}: {model}")
        metadata = {'model_keys': available_models}
        return None, None, None, metadata, available_models, None, None
    
    # Validate model name
    if model_name not in available_models:
        print(f"Model '{model_name}' not found.")
        print(f"Available models: {available_models}")
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    print(f"Generating predictions for model: {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    f = open(paths['data_file'], 'rb')
    dico = pickle.load(f)
    f.close()
    
    # Extract data
    y_test = dico['inputs_tikho_laplacian']  # Input images
    x_test = dico['targets']                 # Target images
    noisy = dico['noisy']                    # Noisy images
    
    # Select images 
    hardcoded_indices = [1283, 899, 1004, 1656]
    #hardcoded_indices = [1004]  
    if len(hardcoded_indices) >= num_images:
        print(f"Using hardcoded indices (standard case)...")
        selected_indices = np.array(hardcoded_indices[:num_images])
        print(f"Selected hardcoded indices: {selected_indices}")
        selection_method = "hardcoded_standard"
        
    else:
        # Fallback to random selection
        print("No hardcoded indices or hallucination scores, falling back to random selection...")
        np.random.seed(42)
        torch.manual_seed(42)
        
        total_images = y_test.shape[0]
        selected_indices = np.random.choice(total_images, num_images, replace=False)
        print(f"Selected random indices: {selected_indices}")
        selection_method = "random_fallback"
    
    # Extract selected images
    selected_inputs = y_test[selected_indices]
    selected_targets = x_test[selected_indices]
    selected_noisy = noisy[selected_indices]
    
    # Normalize images 
    # Normalize targets
    x_norm = selected_targets - np.mean(selected_targets, axis=(1,2), keepdims=True)
    norm_fact = np.max(x_norm, axis=(1,2), keepdims=True) 
    x_norm /= norm_fact
    
    # Normalize & scale inputs using same normalization factor
    y_norm = selected_inputs - np.mean(selected_inputs, axis=(1,2), keepdims=True)
    y_norm /= norm_fact
    
    # Normalize noisy images
    noisy_norm = selected_noisy - np.mean(selected_noisy, axis=(1,2), keepdims=True)
    noisy_norm /= norm_fact
    
    # Convert to NCHW format and torch tensors
    y_norm = np.expand_dims(y_norm, 1).astype(np.float32)
    x_norm = np.expand_dims(x_norm, 1).astype(np.float32)
    noisy_norm = np.expand_dims(noisy_norm, 1).astype(np.float32)
    
    inputs_tensor = torch.tensor(y_norm).to(device)
    targets_tensor = torch.tensor(x_norm).to(device)
    noisy_tensor = torch.tensor(noisy_norm).to(device)
    
    # Load and run the specific model
    yaml_path, model_path = model_configs[model_name]
    
    print(f"Loading model from: {yaml_path}")
    
    with open(yaml_path, 'r') as config:
        opt = yaml.safe_load(config)
        
    Train = opt['TRAINING']
    SUNet = opt['SWINUNET']
    
    # Build model directory path
    model_dir = os.path.join(Train['SAVE_DIR'], SUNet['MODEL_NAME'], Train['LOSS'])
    if not os.path.isabs(model_dir):
        project_root = paths['configs_dir'].parent  # Get project root
        model_dir = str(project_root / model_dir)
    
    full_model_path = os.path.join(model_dir, model_path.lstrip('/'))
    
    print(f"Model directory: {model_dir}")
    print(f"Full model path: {full_model_path}")
    
    # Check if model file exists
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found: {full_model_path}")
    
    # Build Model
    print(f"Building model: {SUNet['MODEL_NAME']}")
    if SUNet['MODEL_NAME'] == 'SUNet':
        model = SUNet_model(opt)
    elif SUNet['MODEL_NAME'] == 'U-Net':
        model = UNet(1)
    elif SUNet['MODEL_NAME'] == 'LearnLet':
        model = Learnlet(n_scales=5, kernel_size=5, filters=64, exact_rec=True, thresh='hard')
    else:
        raise ValueError(f"Unknown model name: {SUNet['MODEL_NAME']}")
    
    model.to(device)
    
    # Load model weights
    def load_checkpoint(model, weights):
        checkpoint = torch.load(weights, map_location=device)
        
        # Try different possible keys for the model state dict
        possible_keys = ['state_dict', 'model', 'model_state_dict', 'net']
        
        state_dict = None
        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        
        if state_dict is None:
            
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
    
    print(f"Loading weights from: {full_model_path}")
    load_checkpoint(model, full_model_path)

    
    print("\n=== Step 1.5: Computing confidence interval ===")
        
    # Load calibration dataset using paths dictionary
    x_train = np.load(paths["x_train_file"])
    y_train = np.load(paths["y_train_file"])
        
    # Use subset for calibration 
    x_cal = x_train[10000:, :, :, :]
    y_cal = y_train[10000:, :, :, :]
    del x_train, y_train
        
        # Normalize targets 
    x_cal = x_cal - np.mean(x_cal, axis=(1,2), keepdims=True)
    norm_fact = np.max(x_cal, axis=(1,2), keepdims=True) 

    x_cal /= norm_fact
        
        # Normalize & scale tikho inputs
    y_cal = y_cal - np.mean(y_cal, axis=(1,2), keepdims=True)
    y_cal /= norm_fact
        
        # NCHW convention
    x_cal = np.moveaxis(x_cal, -1, 1)
    y_cal = np.moveaxis(y_cal, -1, 1)
        
        # Convert to torch tensor
    x_cal = torch.tensor(x_cal).float()
    y_cal = torch.tensor(y_cal).float()
        
        # Split calibration data in half 
    n_obj = x_cal.size()[0]
    n_cal0 = np.int16(n_obj*0.5)
        
    y_cal0 = y_cal[0:n_cal0, :, :, :]
    x_cal0 = x_cal[0:n_cal0, :, :, :]
        
    y_cal1 = y_cal[n_cal0:, :, :, :]
    x_cal1 = x_cal[n_cal0:, :, :, :]
        
    print('Confidence interval evaluation...')
        
    # Compute confidence interval using utils function 
    confidence_interval = utils.confidence_radius(model=model, input_cal0=y_cal0, label_cal0=x_cal0, 
                                                      input_cal1=y_cal1, label_cal1=x_cal1, alpha=alpha, device=device)
        
        # Clean up memory
    del x_cal, x_cal0, x_cal1, y_cal, y_cal0, y_cal1
        
    print(f"Confidence interval computed: {confidence_interval}")
        
       
    interval_radius = torch.tensor(confidence_interval, dtype=torch.float).to(device)
    
    # Generate predictions
    print("Generating predictions...")
    model.eval()
    with torch.no_grad():
        predictions = model(inputs_tensor)
    
    print(f"Generated predictions shape: {predictions.shape}")
    
    # Create metadata
    metadata = {
        'num_images': num_images,
        'selection_method': selection_method,
        'selected_indices': selected_indices.tolist(),
        'model_name': model_name,
        'input_shape': inputs_tensor.shape,
        'device_used': str(device)
    }
    
    
    print(f"Loaded data shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets_tensor.shape}")
    print(f"  Inputs: {inputs_tensor.shape}")
    
    # Save generated data for later use
    print("Saving generated data for future use...")

    output_dir = os.path.join(model_dir, 'wavelet_model_predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data
    np.save(os.path.join(output_dir, f'predictions_{model_name}.npy'), predictions.cpu().numpy())
    np.save(os.path.join(output_dir, 'original_targets.npy'), targets_tensor.cpu().numpy())
    np.save(os.path.join(output_dir, 'original_inputs.npy'), inputs_tensor.cpu().numpy())
    np.save(os.path.join(output_dir, 'original_noisy.npy'), noisy_tensor.cpu().numpy())
    np.save(os.path.join(output_dir, 'selected_indices.npy'), selected_indices)
    
    # Save metadata
    with open(os.path.join(output_dir, f'metadata_{model_name}.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Also save a general metadata file for compatibility
    general_metadata = {
        'model_keys': [model_name],
        'num_images': num_images,
        'selection_method': selection_method,
        'selected_indices': selected_indices.tolist(),
        'device_used': str(device)
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(general_metadata, f)
    
    print(f"Data saved to '{output_dir}' directory:")
    print(f"  - predictions_{model_name}.npy")
    print(f"  - original_targets.npy")
    print(f"  - original_inputs.npy") 
    print(f"  - original_noisy.npy")
    print(f"  - selected_indices.npy")
    print(f"  - metadata_{model_name}.pkl")
    print(f"  - metadata.pkl")
    
    del model
    torch.cuda.empty_cache()
    
    return predictions, targets_tensor, inputs_tensor, metadata, available_models, interval_radius, model_dir

def compute_wavelet_coefficients_and_hi_for_predictions(predictions, targets, wavelets, alpha, theta, device, interval_radius):
    """
    Compute wavelet coefficients for model predictions
    
    Args:
        predictions: model predictions [N, C, H, W]
        targets: ground truth targets [N, C, H, W] 
        wavelets: list of wavelets to test
        alpha, theta: HI parameters
        device: torch device
        interval_radius: confidence interval radius (computed from calibration data)
    
    Returns:
        coeffs_dict: dictionary with wavelet coefficients for each wavelet
        rm_dict: dictionary with Rm values for each wavelet
    """
    
    print("Computing wavelet coefficients for predictions...")
    
    coeffs_dict = {}
    rm_dict = {}
    
    for wavelet in wavelets:
        print(f"Processing wavelet: {wavelet}")
        
        # Initialize HI measure
        hi_measure = Wavelet_Hallucination_Index(alpha, theta, wavelet)
        hi_measure.to(device)
        hi_measure.eval()
        
        predictions_gpu = predictions.to(device)
        targets_gpu = targets.to(device)
        
        with torch.no_grad():
            # Compute Hallucination Index, R, Rm, and Rd 
            R, Rm, Rd = hi_measure(targets_gpu, predictions_gpu, interval_radius)
            
            # Also compute the wavelet coefficients for reconstruction
            pred_coeffs = hi_measure.image_to_wavelet_coeffs(predictions_gpu)
            target_coeffs = hi_measure.image_to_wavelet_coeffs(targets_gpu)
            print (f"shape pred_coeffs: {pred_coeffs.shape}, shape target_coeffs: {target_coeffs.shape}")
            # Convert the flattened coefficients back to the original structure
            coeffs_list = []
            for b in range(predictions.shape[0]):
                batch_coeffs = []
                for c in range(predictions.shape[1]):
                    # Get the image for this batch and channel
                    img = predictions[b, c].cpu().numpy()
                    # Compute 2D wavelet decomposition
                    coeffs_2d = pywt.wavedec2(img, wavelet, mode='periodization')
                    
                    coeffs_2d = coeffs_2d[1:]

                    batch_coeffs.append(coeffs_2d)
                coeffs_list.append(batch_coeffs)
            
            # Store results
            coeffs_dict[wavelet] = {
                'R': R.cpu(),
                'Rm': Rm.cpu(), 
                'Rd': Rd.cpu(),
                'coeffs': coeffs_list, 
                'pred_coeffs_flat': pred_coeffs.cpu(),  
                'target_coeffs_flat': target_coeffs.cpu()  
            }
            
            # Compute average values for classification
            rm_values = torch.mean(R, dim=(1, 2)).cpu().numpy()
            rm_dict[wavelet] = rm_values  # Per-image Rm values
    
    return coeffs_dict, rm_dict


def classify_wavelets_by_coefficient_hi(R_tensor, n_classes=4, thresholds=None):
    """
    Classify wavelet coefficients into classes based on their R values (coefficient-level Hallucination)
    Uses meaningful thresholds based on R value distribution, not equal-sized groups
    
    Args:
        R_tensor: R values tensor [N, C, total_coeffs] from Wavelet_Hallucination_Index
        n_classes: number of classes to create
    
    Returns:
        coefficient_classes: dictionary mapping class indices to coefficient indices
        class_thresholds: threshold values for each class  
        classification_stats: detailed statistics for classification
    """
    
    print("Classifying wavelet coefficients by their R values...")
    
    R_flat = R_tensor.flatten().numpy()
    
    R_flat = R_flat[~np.isnan(R_flat)]
    R_flat = R_flat[~np.isinf(R_flat)]
    
    print(f"Total coefficients to classify: {len(R_flat)}")
    print(f"R value range: [{R_flat.min():.6f}, {R_flat.max():.6f}]")
    print(f"R value statistics:")
    print(f"  Mean: {np.mean(R_flat):.6f}")
    print(f"  Std:  {np.std(R_flat):.6f}")
    print(f"  Median: {np.median(R_flat):.6f}")
    
    
    if thresholds is None:
        mean_R = np.mean(R_flat)
        std_R = np.std(R_flat)
    
        thresholds = [
            0,    
            0.1,    
            0.5   
        ]
    else:
        # When using fixed thresholds, still compute R stats for metadata
        mean_R = np.mean(R_flat)
        std_R = np.std(R_flat)
    
    # Ensure thresholds are within valid range
    thresholds = [max(R_flat.min(), min(R_flat.max(), t)) for t in thresholds]

    print(f"Classification thresholds (based on statistical distribution):")
    for i, thresh in enumerate(thresholds):
        print(f"  Threshold {i+1}: {thresh:.6f}")
    
    # Classify each coefficient
    coefficient_classes = {i: [] for i in range(n_classes)}
    
    # Get original tensor shape for indexing
    batch_size, n_channels, n_coeffs = R_tensor.shape
    
    for b in range(batch_size):
        for c in range(n_channels):
            for coeff_idx in range(n_coeffs):
                r_value = R_tensor[b, c, coeff_idx].item()
                
                # Skip invalid values
                if np.isnan(r_value) or np.isinf(r_value):
                    continue
                
                # Determine class based on thresholds
                class_idx = 0
                for threshold in thresholds:
                    if r_value > threshold:
                        class_idx += 1
                    else:
                        break
                
                # Store global coefficient index
                global_coeff_idx = b * n_channels * n_coeffs + c * n_coeffs + coeff_idx
                coefficient_classes[class_idx].append(global_coeff_idx)
    
    # Classification statistics
    classification_stats = {
        'n_coeffs_per_class': {i: len(coefficient_classes[i]) for i in range(n_classes)},
        'class_thresholds': thresholds,
        'total_coeffs': len(R_flat),
        'R_stats': {
            'mean': float(mean_R),
            'std': float(std_R),
            'min': float(np.min(R_flat)),
            'max': float(np.max(R_flat)),
            'median': float(np.median(R_flat))
        }
    }
    
    # Print class distribution
    total_classified = sum(len(coefficient_classes[i]) for i in range(n_classes))
    print(f"\nClass distribution (total classified: {total_classified}):")
    for i in range(n_classes):
        count = len(coefficient_classes[i])
        percentage = (count / total_classified) * 100 if total_classified > 0 else 0
        if i == 0:
            desc = "No/Minimal Hallucination"
        elif i == 1:
            desc = "Low Hallucination" 
        elif i == 2:
            desc = "Medium Hallucination"
        else:
            desc = "High Hallucination"
        print(f"  Class {i} ({desc}): {count} coefficients ({percentage:.1f}%)")
    
    return coefficient_classes, thresholds, classification_stats

def reconstruct_filtered_predictions(predictions, targets, coeffs_dict, coefficient_classes, classification_stats, wavelet='db4', mode='periodization'):
    """
    Reconstruct predictions with filtering based on coefficient classification
    
    Args:
        predictions: original model predictions [N, C, H, W]
        targets: ground truth targets [N, C, H, W]
        coeffs_dict: wavelet coefficients dictionary  
        coefficient_classes: classification of coefficients into classes
        classification_stats: statistics from classification
        wavelet: wavelet to use for reconstruction
        mode: wavelet mode
    
    Returns:
        reconstructed_predictions: dictionary with reconstructed predictions for each class
    """
    
    print("Reconstructing filtered predictions based on coefficient classification...")
    
    n_images, n_channels, height, width = predictions.shape
    n_classes = len(coefficient_classes)
    
    reconstructed_predictions = {}
    
    # Get R tensor for this wavelet
    R_tensor = coeffs_dict[wavelet]['R']  # [N, C, total_coeffs]
    batch_size, _, n_coeffs = R_tensor.shape
    
    for class_idx in range(n_classes): # for each class
        coeffs_in_class = coefficient_classes[class_idx]
        
        if not coeffs_in_class:
            print(f"No coefficients in class {class_idx}, skipping...")
            continue
        
        print(f"Processing class {class_idx} with {len(coeffs_in_class)} coefficients")
        
        class_reconstructions = []
        
        for img_idx in range(n_images): # for each image
            prediction = predictions[img_idx]
            
            # Convert prediction to numpy for wavelet processing
            prediction_np = prediction.squeeze().cpu().numpy()  # Remove channel dimension
            
            # Compute wavelet coefficients
            coeffs = pywt.wavedec2(prediction_np, wavelet, mode=mode)
            
            # Apply class-specific filtering: keep only coefficients in this class
            filtered_coeffs = apply_coefficient_class_filtering(coeffs, coefficient_classes[class_idx], 
                                                               R_tensor[img_idx, 0], n_coeffs, class_idx)
            
            # Reconstruct prediction
            reconstructed_np = pywt.waverec2(filtered_coeffs, wavelet, mode=mode)
            
            # Convert back to tensor and ensure correct shape
            reconstructed_tensor = torch.tensor(reconstructed_np).float()
            if len(reconstructed_tensor.shape) == 2:
                reconstructed_tensor = reconstructed_tensor.unsqueeze(0)  # Add channel dimension
            
            class_reconstructions.append(reconstructed_tensor)
        
        # Stack all reconstructions for this class
        reconstructed_predictions[class_idx] = torch.stack(class_reconstructions)
        print(f"  Class {class_idx} reconstructions shape: {reconstructed_predictions[class_idx].shape}")
    
    return reconstructed_predictions


def apply_coefficient_class_filtering(coeffs, class_coefficient_indices, R_values, total_coeffs, class_idx):
    """
    Apply filtering based on coefficient class: keep only coefficients belonging to this class
    
    Args:
        coeffs: wavelet coefficients 
        class_coefficient_indices: global indices of coefficients in this class
        R_values: R values for this image [total_coeffs]
        total_coeffs: total number of coefficients
        class_idx: current class index
    
    Returns:
        filtered_coeffs: filtered wavelet coefficients
    """
    
    # Convert global indices to local coefficient positions
    local_coefficient_indices = set()
    for global_idx in class_coefficient_indices:
        # Extract local coefficient index (within this image)
        local_idx = global_idx % total_coeffs
        local_coefficient_indices.add(local_idx)
    
    # Flatten all coefficients to match R_values indexing
    all_coeffs_flat = []
    coeff_positions = []  # Track which level and position each coefficient comes from
    
    # Process approximation coefficients
    cA = coeffs[0]
    cA_flat = cA.flatten()
    for i in range(len(cA_flat)):
        all_coeffs_flat.append(cA_flat[i])
        coeff_positions.append(('approx', 0, i))
    
    # Process detail coefficients
    for level_idx, detail_tuple in enumerate(coeffs[1:]):
        for band_idx, detail_band in enumerate(detail_tuple):
            detail_flat = detail_band.flatten()
            for i in range(len(detail_flat)):
                all_coeffs_flat.append(detail_flat[i])
                coeff_positions.append(('detail', level_idx, band_idx, i))
    
    # Create mask for coefficients in this class
    coefficient_mask = np.zeros(len(all_coeffs_flat), dtype=bool)
    for local_idx in local_coefficient_indices:
        if local_idx < len(coefficient_mask):
            # only coefficients belonging to this class get marked with true
            coefficient_mask[local_idx] = True
    
    # Apply filtering
    filtered_coeffs = []
    coeff_idx = 0
    
    # Filter approximation coefficients
    cA = coeffs[0]
    cA_filtered = np.zeros_like(cA)
    for i in range(cA.shape[0]):
        for j in range(cA.shape[1]):
            if coeff_idx < len(coefficient_mask) and coefficient_mask[coeff_idx]:
                cA_filtered[i, j] = cA[i, j]
            coeff_idx += 1
    filtered_coeffs.append(cA_filtered)
    
    # Filter detail coefficients
    for detail_tuple in coeffs[1:]: # loop through levels
        filtered_tuple = []
        for detail_band in detail_tuple:
            detail_filtered = np.zeros_like(detail_band)
            for i in range(detail_band.shape[0]):
                for j in range(detail_band.shape[1]):
                    if coeff_idx < len(coefficient_mask) and coefficient_mask[coeff_idx]:
                        detail_filtered[i, j] = detail_band[i, j]
                    coeff_idx += 1
            filtered_tuple.append(detail_filtered)
        filtered_coeffs.append(tuple(filtered_tuple))
    
    return filtered_coeffs

def save_analysis_results(predictions, targets, reconstructed_predictions, wavelet_classes, 
                         classification_stats, rm_dict, model_name, metadata, coefficient_classes,
                         model_dir, wavelet='db4'):
    """Save all analysis results """
    
    # Create wavelet-specific subdirectory inside model_dir (same as test_HIC_fwhm_wavelet.py)
    save_dir = os.path.join(model_dir, f'wavelet_{wavelet}_classification')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save numerical results
    results = {
        'model_name': model_name,
        'metadata': metadata,
        'wavelet_classes': wavelet_classes,
        'coefficient_classes': coefficient_classes,
        'classification_stats': classification_stats,
        'rm_dict': rm_dict,
        'predictions_shape': predictions.shape,
        'targets_shape': targets.shape,
        'reconstructed_shapes': {k: v.shape for k, v in reconstructed_predictions.items()}
    }
    
    with open(os.path.join(save_dir, f'{model_name}_analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save reconstructed predictions as numpy arrays
    for class_idx, recon_preds in reconstructed_predictions.items():
        np.save(os.path.join(save_dir, f'{model_name}_class_{class_idx}_reconstructed_high.npy'), 
                recon_preds.cpu().numpy())
    
    print(f"Analysis results saved to '{save_dir}' directory:")
    print(f"  - {model_name}_analysis_results.pkl")
    print(f"  - {model_name}_class_*_reconstructed.npy (for each class)")



def main(model_name=None):
    """
    Main function to run the complete analysis on model predictions
    """
    
    print("=== Wavelet Classification Analysis for Model Predictions ===")
    
    # Configuration
    wavelets = ['db8']  # wavelet to test
    n_classes = 4
    alpha = 0.01
    theta = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Testing wavelets: {wavelets}")
    
    try:
        # Step 1: Generate model predictions
        print("\n=== Step 1: Loading model predictions ===")
        predictions, targets, inputs, metadata, available_models, interval_radius, model_dir = generate_model_predictions(
            model_name, num_images=4, alpha = 0.01)

        # If model_name is None, just return the available models
        if model_name is None:
            return available_models
        
        # Step 2: Compute wavelet coefficients for predictions
        print(f"\n=== Step 2: Computing wavelet coefficients for {model_name} ===")
        coeffs_dict, rm_dict = compute_wavelet_coefficients_and_hi_for_predictions(
            predictions, targets, wavelets, alpha, theta, device, interval_radius)
        
        # Step 3: Classify coefficients by their R values (coefficient-level HI)
        print("\n=== Step 3: Classifying coefficients by their R values ===")
        # Use the first wavelet's R tensor for classification
        main_wavelet = wavelets[0]
        R_tensor = coeffs_dict[main_wavelet]['R']
        coefficient_classes, class_thresholds, classification_stats = classify_wavelets_by_coefficient_hi(
            R_tensor, n_classes)
        
        # Step 4: Reconstruct filtered predictions
        print("\n=== Step 4: Reconstructing filtered predictions ===")
        reconstructed_predictions = reconstruct_filtered_predictions(
            predictions, targets, coeffs_dict, coefficient_classes, classification_stats, main_wavelet)
        
        # Step 5: Create simple wavelet_classes dict for visualization compatibility
        wavelet_classes = {i: [main_wavelet] for i in range(n_classes) if i in coefficient_classes and coefficient_classes[i]}
        
        # Update classification_stats for visualization compatibility
        classification_stats['wavelet_avg_rm'] = {main_wavelet: np.mean(rm_dict[main_wavelet])}
        
        # Step 6: Save results
        print("\n=== Step 6: Saving results ===")
        save_analysis_results(predictions, targets, reconstructed_predictions, wavelet_classes, 
                             classification_stats, rm_dict, model_name, metadata, coefficient_classes,
                             model_dir, main_wavelet)
        
        print(f"\nAnalysis completed successfully for model: {model_name}")
        
        return {
            'predictions': predictions,
            'targets': targets,
            'reconstructed_predictions': reconstructed_predictions,
            'wavelet_classes': wavelet_classes,
            'coefficient_classes': coefficient_classes,
            'classification_stats': classification_stats
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main('U-Net_L1_Loss')      
    results = main('U-Net_L2_Loss')      
    results = main('SUNet_L1_Loss')      
    results = main('SUNet_L2_Loss')      
    results = main('LearnLet_L1_Loss')   
    results = main('LearnLet_L2_Loss')  

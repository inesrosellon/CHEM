import os
import numpy as np
import torch
import pyshearlab
import sys
from pathlib import Path
import pickle


from collections import OrderedDict
import yaml

# Add directorries to path
current_dir = Path(__file__).resolve().parent  # src/eval/
src_dir = current_dir.parent  # src/
sys.path.insert(0, str(src_dir))

train_dir = src_dir / "train"
sys.path.insert(0, str(train_dir))

import utils
from utils.HIC_psf_fwhm_shearlet_utils import Shearlet_Hallucination_Index
from utils.io import load_paths

from model.SUNet import SUNet_model
from model.Unet import UNet
from model.LearnLet import Learnlet

paths = load_paths()

def generate_model_predictions(model_name = None, num_images=4, alpha=0.01):
     # Configuration 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model configurations 
    model_configs = {
        'U-Net_L1_Loss': (paths['configs_dir'] / 'training_UNet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        'U-Net_L2_Loss': (paths['configs_dir'] / 'training_UNet_L2.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'), 
        'SUNet_L1_Loss': (paths['configs_dir'] / 'training_SUNet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        'SUNet_L2_Loss': (paths['configs_dir'] / 'training_SUNet_L2.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        'LearnLet_L1_Loss': (paths['configs_dir'] / 'training_LeLet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        'LearnLet_L2_Loss': (paths['configs_dir'] / 'training_LeLet_L2.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
    }
    available_models = list(model_configs.keys())
    
     # Load test data for FWHM analysis 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading test dataset...")
    f = open(paths['data_file'], 'rb')
    dico = pickle.load(f)
    f.close()
    
    y_test = dico['inputs_tikho_laplacian']  # Input images
    x_test = dico['targets']                 # Target images
    noisy = dico['noisy']                    # Noisy images
    
    # Select images 
    hardcoded_indices = [1283, 899, 1004, 1656] 

    print(f"Using hardcoded indices ...")
    selected_indices = np.array(hardcoded_indices[:num_images])
    print(f"Selected hardcoded indices: {selected_indices}")
    selection_method = "hardcoded_standard"
    
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
    
    with open(yaml_path, 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    SUNet = opt['SWINUNET']
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
    
    def load_checkpoint(model, weights):
        checkpoint = torch.load(weights, map_location=device)
        
        # Try different possible keys for the model state dict
        possible_keys = ['state_dict', 'model', 'model_state_dict', 'net']
        
        state_dict = None
        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        
       
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        print("Model checkpoint loaded successfully")
    
    load_checkpoint(model, full_model_path)

    print("Loading calibration data...")
   # Load calibration dataset using paths dictionary
    x_train = np.load(paths["x_train_file"])
    y_train = np.load(paths["y_train_file"])
    
    x_cal = x_train[10000:, :, :, :]
    y_cal = y_train[10000:, :, :, :]
    del x_train, y_train
    
    # Normalize calibration data
    x_cal = x_cal - np.mean(x_cal, axis=(1,2), keepdims=True)
    norm_fact = np.max(x_cal, axis=(1,2), keepdims=True) 
    x_cal /= norm_fact
    y_cal = y_cal - np.mean(y_cal, axis=(1,2), keepdims=True)
    y_cal /= norm_fact
    
    x_cal = np.moveaxis(x_cal, -1, 1)
    y_cal = np.moveaxis(y_cal, -1, 1)
    
    x_cal = torch.tensor(x_cal).float()
    y_cal = torch.tensor(y_cal).float()
    
    n_obj = x_cal.size()[0]
    n_cal0 = np.int16(n_obj*0.5)
    
    y_cal0 = y_cal[0:n_cal0, :, :, :]
    x_cal0 = x_cal[0:n_cal0, :, :, :]
    y_cal1 = y_cal[n_cal0:, :, :, :]
    x_cal1 = x_cal[n_cal0:, :, :, :]
    
    interval_radius = utils.confidence_radius(
        model=model, input_cal0=y_cal0, label_cal0=x_cal0, 
        input_cal1=y_cal1, label_cal1=x_cal1, alpha=alpha, device=device
    )
    
    # Convert interval_radius to tensor on correct device
    if isinstance(interval_radius, np.ndarray):
        interval_radius = torch.tensor(interval_radius, dtype=torch.float).to(device)
    else:
        interval_radius = interval_radius.to(device)
    
    del x_cal, x_cal0, x_cal1, y_cal, y_cal0, y_cal1
    
   #####################################################################################
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
    output_dir = os.path.join(model_dir, 'model_predictions')
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
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return predictions, targets_tensor, inputs_tensor, metadata, available_models, interval_radius, model_dir

def compute_shearlet_coefficients_and_hi_for_predictions(predictions, targets, scales, alpha, theta, device, interval_radius):
    """
    Compute shearlet coefficients for model predictions
    
    Args:
        predictions: model predictions [N, C, H, W]
        targets: ground truth targets [N, C, H, W] 
        scales: number of shearlet scales
        alpha, theta: CHEM parameters
        device: torch device
        interval_radius: confidence interval
    
    Returns:
        coeffs_dict: dictionary with shearlet coefficients
        rm_dict: dictionary with CHEM values 
    """
    
    print("Computing shearlet coefficients for predictions...")
    
    coeffs_dict = {}
    rm_dict = {}
    
    print(f"Processing shearlet scales: {scales}")
    
    
    
    # Create shearlet HI computer
    hi_measure = Shearlet_Hallucination_Index(alpha=alpha, theta=theta, scales=scales)
    hi_measure.to(device)
    hi_measure.eval()

    # Move data to device
    predictions = predictions.to(device)
    targets = targets.to(device)
    interval_radius = interval_radius.to(device)
    
    with torch.no_grad():
        # Compute hallucination indices and related metrics
        R, Rm, Rd = hi_measure(targets, predictions, interval_radius)

        pred_coeffs = hi_measure.image_to_shearlet_coeffs(predictions).cpu()
        target_coeffs = hi_measure.image_to_shearlet_coeffs(targets).cpu()

        # Store results
        scale_key = f"scales_{scales}"
        coeffs_dict[scale_key] = {
            'Rm': Rm.cpu().numpy(),
            'Rd': Rd.cpu().numpy(),
            'R': R.cpu(),
            'target_coeffs': target_coeffs,
            'pred_coeffs': pred_coeffs
        }
        
        # Store mean R for compatibility
        rm_values = torch.mean(R, dim=(1, 2)).cpu().numpy()
        rm_dict[scale_key] = rm_values

        print(f"  Mean R per image: {rm_dict[scale_key]}")
    
    return coeffs_dict, rm_dict


def classify_shearlets_by_coefficient_hi(R, n_classes=4, thresholds=None):
    """
    Classify shearlet coefficients into classes based on their R values (coefficient-level Hallucination)
    
    Args:
        R: R values tensor [N, C, total_coeffs] from Shearlet_Hallucination_Index
        n_classes: number of classes to create
    
    Returns:
        coefficient_classes: dictionary mapping class indices to coefficient indices
        class_thresholds: threshold values for each class  
        classification_stats: detailed statistics for classification
    """
    print("Classifying shearlet coefficients by their R values...")
    
    
    R_flat = R.flatten().cpu().numpy()
    
    R_flat = R_flat[~np.isnan(R_flat)]
    R_flat = R_flat[~np.isinf(R_flat)]
    
    print(f"Total coefficients to classify: {len(R_flat)}")
    print(f"R value range: [{R_flat.min():.6f}, {R_flat.max():.6f}]")
    print(f"R value statistics:")
    print(f"  Mean: {np.mean(R_flat):.6f}")
    print(f"  Std:  {np.std(R_flat):.6f}")
    print(f"  Median: {np.median(R_flat):.6f}")
    
    
    if thresholds is None:
        thresholds = [
            0,    
            0.1,                   
            0.5                    
        ]
   

    print(f"Classification thresholds:")
    for i, thresh in enumerate(thresholds):
        print(f"  Threshold {i+1}: {thresh:.6f}")
    
    # Classify each coefficient
    coefficient_classes = {i: [] for i in range(n_classes)}
    
    # Get original tensor shape for indexing
    batch_size, n_channels, n_coeffs = R.shape
    
    for b in range(batch_size):
        for c in range(n_channels):
            for coeff_idx in range(n_coeffs):
                R_val = R[b, c, coeff_idx].item()
                
                # Skip invalid values
                if np.isnan(R_val) or np.isinf(R_val):
                    continue
                
                # Determine class based on thresholds
                class_idx = 0
                for threshold in thresholds:
                    if R_val > threshold:
                        class_idx += 1
                    else:
                        break
                
                # Store global coefficient index
                global_idx = b * n_channels * n_coeffs + c * n_coeffs + coeff_idx
                coefficient_classes[class_idx].append(global_idx)
    
    # Calculate classification statistics
    n_coeffs_per_class = {i: len(coefficient_classes[i]) for i in range(n_classes)}
    total_coeffs = sum(n_coeffs_per_class.values())
    percentages = {i: (n_coeffs_per_class[i] / total_coeffs * 100) if total_coeffs > 0 else 0 
                   for i in range(n_classes)}
    
    print(f"Classification results:")
    class_names = ['No/Minimal Hallucination', 'Low Hallucination', 'Medium Hallucination', 'High Hallucination']
    for i in range(n_classes):
        count = len(coefficient_classes[i])
        percentage = percentages[i]
        print(f"  Class {i} ({class_names[i]}): {count} coefficients ({percentage:.1f}%)")
    
    classification_stats = {
        'n_coeffs_per_class': n_coeffs_per_class,
        'percentages_per_class': percentages,
        'class_thresholds': thresholds,  
        'total_coeffs': total_coeffs,
        'R_stats': {  
            'mean': np.mean(R_flat),
            'std': np.std(R_flat),
            'min': R_flat.min(),
            'max': R_flat.max(),
            'median': np.median(R_flat)
        }
    }
    
    return coefficient_classes, thresholds, classification_stats


def reconstruct_shearlet_filtered_predictions(predictions, targets, R, coefficient_classes, 
                                             classification_stats, scales=3):
    """
    Reconstruct predictions using only shearlet coefficients from specific classes
    
    Args:
        predictions: prediction tensor [N, C, H, W]
        targets: target tensor [N, C, H, W] 
        R: R values tensor [N, C, total_coeffs]
        coefficient_classes: dict mapping class -> list of coefficient indices
        classification_stats: classification statistics
        scales: number of shearlet scales
        
    Returns:
        reconstructed_predictions: dict mapping class -> reconstructed images tensor
    """
    if pyshearlab is None:
        raise ImportError("pyshearlab is required for shearlet reconstruction")
        
    print("Reconstructing shearlet-filtered predictions for each class...")
    
    n_images, n_channels, height, width = predictions.shape
    n_classes = len(coefficient_classes)
    reconstructed_predictions = {}
    
    # Get shearlet system for reconstruction
    shearletSystem = pyshearlab.SLgetShearletSystem2D(0, height, width, scales)
    
    for class_idx, coeff_indices in coefficient_classes.items():
        if len(coeff_indices) == 0:
            print(f"  Class {class_idx}: No coefficients - skipping")
            continue
            
        print(f"  Class {class_idx}: Reconstructing with {len(coeff_indices)} coefficients")
        
        class_reconstructions = []
        
        for b in range(n_images):
            for c in range(n_channels):
                # Get prediction coefficients for this image and channel
                pred_np = predictions[b, c].cpu().numpy()
                coeffs = pyshearlab.SLsheardec2D(pred_np, shearletSystem)
                
                
                coeffs_normalized = coeffs.copy()
                for j in range(coeffs_normalized.shape[2]):
                    coeffs_normalized[:, :, j] /= shearletSystem["RMS"][j]
               
                filtered_coeffs = coeffs.copy()
                
                filtered_coeffs[:, :, :-1] = 0  
                
                n_bands_excl_lowfreq = coeffs.shape[2] - 1  
                coeffs_per_band = coeffs.shape[0] * coeffs.shape[1]
                total_coeffs_excl_lowfreq = n_bands_excl_lowfreq * coeffs_per_band
                
                for global_idx in coeff_indices:
                    # Convert global index to batch, channel, and coefficient indices

                    batch_idx = global_idx // (n_channels * total_coeffs_excl_lowfreq)
                    remaining = global_idx % (n_channels * total_coeffs_excl_lowfreq)
                    channel_idx = remaining // total_coeffs_excl_lowfreq
                    coeff_idx = remaining % total_coeffs_excl_lowfreq

                    if batch_idx == b and channel_idx == c:
                        band_idx = coeff_idx // coeffs_per_band 
                        coeff_in_band = coeff_idx % coeffs_per_band  
                        row_idx = coeff_in_band // coeffs.shape[1]
                        col_idx = coeff_in_band % coeffs.shape[1]

                        # Bounds checking and coefficient restoration
                        if (band_idx < n_bands_excl_lowfreq and 
                            row_idx < coeffs.shape[0] and 
                            col_idx < coeffs.shape[1]):
                            # Restore the original coefficient value
                            filtered_coeffs[row_idx, col_idx, band_idx] = coeffs[row_idx, col_idx, band_idx]

                # Reconstruct using filtered coefficients
                reconstructed = pyshearlab.SLshearrec2D(filtered_coeffs, shearletSystem)
                class_reconstructions.append(torch.tensor(reconstructed).float().unsqueeze(0).unsqueeze(0))
        
        if class_reconstructions:
            reconstructed_predictions[class_idx] = torch.cat(class_reconstructions, dim=0)
    
   
    del shearletSystem
    
    print(f"Successfully reconstructed {len(reconstructed_predictions)} classes")
    return reconstructed_predictions

def save_analysis_results(predictions, targets, reconstructed_predictions, coefficient_classes, 
                         classification_stats, rm_dict, model_name, metadata, scales,
                         model_dir):
    """
    Save all analysis results for later use
    """
    
    
    save_dir = os.path.join(model_dir, f'shearlet_{scales}_classification')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save main results
    results = {
        'model_name': model_name,
        'metadata': metadata,
        'scales': scales,
        'coefficient_classes': coefficient_classes,
        'classification_stats': classification_stats,
        'rm_dict': rm_dict,
        'predictions_shape': predictions.cpu().numpy().shape,
        'targets_shape': targets.cpu().numpy().shape,
        'reconstructed_predictions_shape': {k: v.cpu().numpy().shape for k, v in reconstructed_predictions.items()},
    }
    
    results_file = os.path.join(save_dir, f'{model_name}_shearlet_analysis_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save individual class reconstructions as separate .npy files
    for class_idx, class_data in reconstructed_predictions.items():
        class_file = os.path.join(save_dir, f'{model_name}_class_{class_idx}_reconstructed.npy')
        np.save(class_file, class_data.cpu().numpy())
        print(f"  Saved class {class_idx} reconstruction: {class_file}")
    
    print(f"Analysis results saved: {results_file}")
    print(f"Analysis results saved to '{save_dir}' directory:")
    print(f"  - {model_name}_shearlet_analysis_results.pkl")
    print(f"  - {model_name}_class_*_reconstructed.npy (for each class)")
    return results_file


def main(model_name=None):
    """
    Main analysis function
    """
    if pyshearlab is None:
        print("Error: pyshearlab not available. Please install it for shearlet analysis.")
        return
    
    print("=== Shearlet Coefficient Classification Analysis ===")
    
    # Configuration
    
    scales = 3
    n_classes = 4
    alpha = 0.01
    theta = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    
    # Step 1: Load model predictions
    print("\n=== Step 1: Loading model predictions ===")
    predictions, targets, inputs, metadata, available_models, interval_radius, model_dir = generate_model_predictions(model_name, num_images=4, alpha=alpha)
    
    if predictions is None:
        print("No predictions loaded. Available models:", available_models)
        return available_models

   # Step 2: Compute shearlet coefficients for predictions
    print(f"\n=== Step 2: Computing shearlet coefficients for {model_name} ===")
    coeffs_dict, rm_dict = compute_shearlet_coefficients_and_hi_for_predictions(
        predictions, targets, scales, alpha, theta, device, interval_radius
    )
    
    # Step 3: Classify coefficients by their R values (coefficient-level HI)
    print("\n=== Step 3: Classifying coefficients by their R values ===")
    scale_key = f"scales_{scales}"
    R = coeffs_dict[scale_key]['R']
    
    # Classify coefficients
    coefficient_classes, class_thresholds, classification_stats = classify_shearlets_by_coefficient_hi(R, n_classes)
    
    # Step 4: Reconstruct filtered predictions
    print("\n=== Step 4: Reconstructing filtered predictions ===")
    reconstructed_predictions = reconstruct_shearlet_filtered_predictions(
        predictions, targets, R, coefficient_classes, classification_stats, scales
    )
    
    # Step 5: Save results
    save_analysis_results(
        predictions, targets, reconstructed_predictions, coefficient_classes,
        classification_stats, rm_dict, model_name, metadata, scales, model_dir
    )
    
    print("=== Shearlet Analysis Complete ===")
    return {
        'predictions': predictions,
        'targets': targets,
        'reconstructed_predictions': reconstructed_predictions,
        'coefficient_classes': coefficient_classes,
        'classification_stats': classification_stats
    }


def process_all_models():
    """Process all models and create combined data for plotting"""
    print("=== Processing All Models for Shearlet Analysis ===")
    
    model_configs = [
        'U-Net_L1_Loss',
        'U-Net_L2_Loss', 
        'SUNet_L1_Loss',
        'SUNet_L2_Loss',
        'LearnLet_L1_Loss',
        'LearnLet_L2_Loss'
    ]
    
    all_results = {}
    
    for model_name in model_configs:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        try:
            results = main(model_name)
            if results:
                all_results[model_name] = results
                print(f"Successfully processed {model_name}")
            else:
                print(f"Failed to process {model_name}")
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            continue
    
    print(f"\n Completed processing {len(all_results)}/{len(model_configs)} models")
    return all_results

if __name__ == "__main__":
    all_results = process_all_models()
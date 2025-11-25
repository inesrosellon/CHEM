# -*- coding: utf-8 -*-
"""

"""

import os
import sys
from pathlib import Path

# Add the parent directory (src) to the Python path
current_dir = Path(__file__).resolve().parent  # src/eval/
src_dir = current_dir.parent  # src/
sys.path.insert(0, str(src_dir))

# Add the train directory to the path for model imports
train_dir = src_dir / "train"
sys.path.insert(0, str(train_dir))
import numpy as np
import pickle
import torch
import pywt
from collections import OrderedDict
import yaml
import utils
from model.SUNet import SUNet_model
from model.Unet import UNet
from model.LearnLet import Learnlet
import time
from utils.io import load_paths

paths = load_paths()

def exp_Wavelet_Hallucination(yaml_path, trained_model_path, wavelet='haar'):

    # Load model
    with open(yaml_path, 'r') as config:
        opt = yaml.safe_load(config)
        
    Train = opt['TRAINING']
    SUNet = opt['SWINUNET']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("^^^ device:", device)
    print(f"^^^ Using wavelet: {wavelet}")
    
    alpha = 0.01
    theta = 1
    
    model_dir = os.path.join(Train['SAVE_DIR'], SUNet['MODEL_NAME'], Train['LOSS'])
    # Convert to absolute path
    if not os.path.isabs(model_dir):
        project_root = paths['configs_dir'].parent  # Get project root
        model_dir = str(project_root / model_dir)
    
    ## Build Model
    print('==> Load the model')
    
    print('^^^ ', SUNet['MODEL_NAME'])
    print('^^^ ', Train['LOSS'])
    if SUNet['MODEL_NAME'] == 'SUNet':
        model = SUNet_model(opt)
    elif SUNet['MODEL_NAME'] == 'U-Net':
        model = UNet(1)
    elif SUNet['MODEL_NAME'] == 'LearnLet':
        model = Learnlet(n_scales=5, kernel_size=5, filters=64, exact_rec=True, thresh='hard')
    
    model.to(device)
    
    def load_checkpoint(model, weights):
        checkpoint = torch.load(weights)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    
    print(model_dir)
    model_path = os.path.join(model_dir, trained_model_path.lstrip('/'))
    load_checkpoint(model, model_path)
    
    ## calibration dataset
    x_train = np.load(paths["x_train_file"])
    y_train = np.load(paths["y_train_file"])
    
    
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
    
    n_obj = x_cal.size()[0]
    n_cal0 = np.int16(n_obj*0.5)
    
    y_cal0 = y_cal[0:n_cal0, :, :, :]
    x_cal0 = x_cal[0:n_cal0, :, :, :]
    
    y_cal1 = y_cal[n_cal0:, :, :, :]
    x_cal1 = x_cal[n_cal0:, :, :, :]
    
    print('Confidence interval evaluation...')
    
    confidence_interval = utils.confidence_radius(model=model, input_cal0=y_cal0, label_cal0=x_cal0, 
                                                  input_cal1=y_cal1, label_cal1=x_cal1, alpha = alpha, device = device)

    del x_cal, x_cal0, x_cal1, y_cal, y_cal0, y_cal1
    
    ## test dataset
    f = open(paths['data_file'], 'rb')
    dico = pickle.load(f)
    f.close()
    
    # Norm
    noise_sigma_orig = dico['noisemap']
    y_test = dico['inputs_tikho_laplacian']
    x_test = dico['targets']
    noisy = dico['noisy']
    psf = dico['psf']
    
    
    print('Wavelet Hallucination procedure with FWHM variation...')
    
    fwhm_level = [10, 20, 30, 40, 50, 60, 70]
    noisy_tensor = np.expand_dims(noisy, 1)
    x_test_tensor = np.expand_dims(x_test, 1)
    
    noisy_tensor = torch.tensor(noisy_tensor).float()
    x_test_tensor = torch.tensor(x_test_tensor).float()
    
    MSE_x, RMean, RStd = utils.Wavelet_Hallucination(model=model, input_=noisy_tensor, label_=x_test_tensor, alpha=alpha, 
                                                   interval_radius=confidence_interval, fwhm_level=fwhm_level, 
                                                   psf=psf, theta=theta, device=device, wavelet=wavelet)

    # Create wavelet FWHM-specific directory
    wavelet_fwhm_dir = os.path.join(model_dir, f'wavelet_{wavelet}_fwhm')
    os.makedirs(wavelet_fwhm_dir, exist_ok=True)
    
    np.save(os.path.join(wavelet_fwhm_dir, "MSE_x.npy"), MSE_x)
    np.save(os.path.join(wavelet_fwhm_dir, "RMean.npy"), RMean)
    np.save(os.path.join(wavelet_fwhm_dir, "RStd.npy"), RStd)
    
    print(f'Wavelet {wavelet} FWHM MSE ', MSE_x)

if __name__ == "__main__":
    
   
    # Full configuration for production
    wavelets = ['db8']
    
    configs = [
        (paths['configs_dir'] / 'training_UNet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        (paths['configs_dir'] / 'training_UNet_L2.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'), 
        (paths['configs_dir'] / 'training_SUNet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        (paths['configs_dir'] / 'training_SUNet_L2.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        (paths['configs_dir'] / 'training_LeLet_L1.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
        (paths['configs_dir'] / 'training_LeLet_L2.yaml', '/model_bestPSNR_ep-500_bs-4_ps-4.pth'),
    ]

    for wavelet in wavelets:
        print(f"\n=== Testing with {wavelet} wavelet FWHM ===")
        
        for yaml_path, model_path in configs:
            print(f"Testing {yaml_path}...")
            
            try:
                start_time = time.time()
                exp_Wavelet_Hallucination(yaml_path, model_path, wavelet)
                end_time = time.time()
                print(f'Testing time: {end_time - start_time:.2f} s')
                
            except Exception as e:
                print(f"Error testing {yaml_path}: {e}")
                continue
    
    print("\n=== Wavelet FWHM hallucination testing completed ===")

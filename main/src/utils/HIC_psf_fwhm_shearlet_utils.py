import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import random
from torch.utils.data import TensorDataset, DataLoader
import utils
import pyshearlab
import warnings
import sys
import io

# Suppress pyshearlab warnings about filter configurations
warnings.filterwarnings("ignore", category=UserWarning, module="pyshearlab")

class Shearlet_Hallucination_Index(torch.nn.Module):
    """
    Shearlet-based Hallucination Index computation
    """
    def __init__(self, alpha, theta, scales=3):
        super(Shearlet_Hallucination_Index, self).__init__()
        
        self.alpha = alpha
        self.theta = theta
        self.scales = scales
        self.relu = torch.nn.ReLU()

    def image_to_shearlet_coeffs(self, image):
        """Convert image tensor to shearlet coefficients"""
        if pyshearlab is None:
            raise ImportError("pyshearlab is required for shearlet analysis")
            
        # image: [N, C, H, W]
        batch_size, channels, height, width = image.shape
        device = image.device  # Get the device from input tensor
        coeffs_list = []
        
        for b in range(batch_size):
            batch_coeffs = []
            for c in range(channels):
                img_np = image[b, c].cpu().numpy()
                coeffs, shearletSystem = DST(img_np, self.scales) #[H,W,B]
                coeffs = coeffs[:,:,:-1] # Remove low frequency band
                for j in range(coeffs.shape[2]):
                    # Normalize coefficients by RMS values
                    coeffs[:, :, j] /= shearletSystem["RMS"][j]
                coeffs_flat = coeffs.flatten()
                # Ensure tensor is on the same device as input
                batch_coeffs.append(torch.tensor(coeffs_flat, device=device, dtype=torch.float32))
            
            coeffs_list.append(torch.stack(batch_coeffs))
        
        
        result = torch.stack(coeffs_list)  # [N, C, total_coeffs]
        return result

    def forward(self, target, prediction, interval_radius):
        # Convert images to shearlet coefficients
        target_coeffs = self.image_to_shearlet_coeffs(target)
        pred_coeffs = self.image_to_shearlet_coeffs(prediction)
        
        # Compute residual in shearlet domain
        Residual = torch.abs(target_coeffs - pred_coeffs)
        
        device = target.device  
        
        if len(interval_radius.shape) == 4:
            # [N, C, H, W] format - use image_to_shearlet_coeffs method
            Radius = self.image_to_shearlet_coeffs(interval_radius)
        else:
            raise ValueError(f"Unexpected interval_radius shape: {interval_radius.shape}")
        
        R = -self.relu(self.theta - self.relu(Residual - Radius)) + self.theta
       
        # Compute mean and std, ensuring they are tensors
        Rm = torch.mean(R, dim=(1, 2), keepdim=False)
        Rd = torch.std(R, dim=(1, 2), keepdim=False)
      
        # Expand dimensions safely
        Rm_expanded = Rm.unsqueeze(1).unsqueeze(2)  # [N] -> [N, 1, 1]
        Rd_expanded = Rd.unsqueeze(1).unsqueeze(2)  # [N] -> [N, 1, 1]
        # (for classification) Normalize R
        #R = (R - Rm_expanded) / Rd_expanded
        
        # Memory cleanup for intermediate tensors
        del Residual, Radius
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return R, Rm, Rd  # Keep original API

class SuppressShearletWarnings:
    """Context manager to suppress pyshearlab warnings and stdout messages"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def DST(X, scales=3):
    """Discrete Shearlet Transform """
    with SuppressShearletWarnings():
        shearletSystem = pyshearlab.SLgetShearletSystem2D(0, X.shape[0], X.shape[1], scales)
        coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)
    
    return coeffs, shearletSystem

def IST(coeffs, scales=3):
    """Inverse Shearlet Transform"""
    with SuppressShearletWarnings():
        shearletSystem = pyshearlab.SLgetShearletSystem2D(0, coeffs.shape[0], coeffs.shape[1], scales)
        Xrec = pyshearlab.SLshearrec2D(coeffs, shearletSystem)
    
    del shearletSystem
    return Xrec


def shearlet_mse_full(target, prediction, scales=3):
    """
    Shearlet-domain MSE that honours Parseval's theorem.
    Computes the global squared-error over *all* coefficients.
    """
    batch, channels = target.shape[:2]
    total_se = 0.0   # sum of squared errors
    total_coeff = 0  # number of coefficients

    for b in range(batch):
        for c in range(channels):
            t = target[b, c].detach().cpu().numpy().astype(np.float32)
            p = prediction[b, c].detach().cpu().numpy().astype(np.float32)
            
            # DST returns (coeffs, shearletSystem) - we only need coeffs
            t_coeffs, _ = DST(t, scales)
            p_coeffs, _ = DST(p, scales)

            # Compute squared error over all coefficients
            diff = t_coeffs - p_coeffs
            total_se += np.sum(diff**2)
            total_coeff += diff.size

    return total_se / total_coeff


def Shearlet_Hallucination(model, input_, label_, alpha, interval_radius, fwhm_level, psf, theta, device, scales=3):
    """
    Lightweight Shearlet Hallucination function - only computes mean R values
    For detailed coefficient analysis, use generate_detailed_shearlet_analysis
    """
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    print("^^^ Shearlet Hallucination device: ", device)
    print(f"^^^ Using shearlet scales: {scales}")
    
    model.to(device)
    
    interval_radius = torch.tensor(interval_radius, dtype=torch.float)
    interval_radius = interval_radius.to(device)
    
    print(f"^^^ Interval radius shape: {interval_radius.shape}")
    print(f"^^^ Input shape: {input_.shape}")
    print(f"^^^ Label shape: {label_.shape}")
    
    MSEerror = np.zeros(len(fwhm_level))
    RMean = np.zeros((len(fwhm_level), input_.shape[0]))
    RStd = np.zeros((len(fwhm_level), input_.shape[0]))
    
    Hmeasure = Shearlet_Hallucination_Index(alpha, theta, scales)
    Hmeasure.to(device)
    
    for i in range(len(fwhm_level)):
        
        print(f"Processing FWHM level {i+1}/{len(fwhm_level)}: {fwhm_level[i]}")
        
        # Data Preparation 
        noisy_np = input_.cpu().numpy()
        targets_np = label_.cpu().numpy()
        psf_np = psf
        
        # Convert from NCHW to NHWC for utils functions
        noisy_np = np.moveaxis(noisy_np, 1, -1).squeeze(-1)  # Remove channel dim
        targets_np = np.moveaxis(targets_np, 1, -1).squeeze(-1)
        
        # Generate new input with different FWHM
        input_fwhm = utils.dataset_fwhm(noisy=noisy_np, targets=targets_np, psf=psf_np, fwhm=fwhm_level[i])
        
        # Normalization
        x_test = targets_np - np.mean(targets_np, axis=(1,2), keepdims=True)
        norm_fact = np.max(x_test, axis=(1,2), keepdims=True) 
        x_test /= norm_fact
        
        # Normalize & scale FWHM inputs using same normalization factor
        y_test = input_fwhm - np.mean(input_fwhm, axis=(1,2), keepdims=True)
        y_test /= norm_fact
        
        # Convert to tensor format (NCHW)
        y_test = np.expand_dims(y_test.astype(np.float32), 1)
        x_test = np.expand_dims(x_test.astype(np.float32), 1)
        
        # Convert to torch tensor
        y_test = torch.tensor(y_test).float()
        x_test = torch.tensor(x_test).float()
        
      
        batch_size = 4  
        
        test_dataset = TensorDataset(y_test.to(device), x_test.to(device))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                drop_last=False)
        
        del test_dataset, input_fwhm, y_test, x_test
        
        model.eval()
        Hmeasure.eval()
        
        RM = []
        RD = []
        mseerror = [] 
        
        for ii, data_test in enumerate(test_loader, 0):
            
            # Progress reporting for large datasets
            if ii % 100 == 0 and ii > 0:
                print(f"   Processed {ii * batch_size} images...")
            
            target = data_test[1].to(device)
            noisyinput = data_test[0].to(device)
        
            with torch.no_grad():
                pred_ = model(noisyinput)
                
                # Compute hallucination metrics 
                R, rm, rd = Hmeasure(target, pred_, interval_radius)
                
                mse = shearlet_mse_full(target, pred_, scales)
                
                mseerror.append(mse)
                
                if rm.numel() == 1:
                    RM.append(rm.detach().cpu().item())
                else:
                    RM.extend(rm.detach().cpu().numpy().tolist())
                
                if rd.numel() == 1:
                    RD.append(rd.detach().cpu().item())
                else:
                    RD.extend(rd.detach().cpu().numpy().tolist())
                
                # Immediate cleanup after extracting values
                del noisyinput, pred_, R, rm, rd, hi
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        MSEerror[i] = np.mean(mseerror)
        
        if len(RM) > 0:  # Handle case where debug mode limits samples
            RMean[i, :min(len(RM), RMean.shape[1])] = RM[:RMean.shape[1]]
            RStd[i, :min(len(RD), RStd.shape[1])] = RD[:RStd.shape[1]]
        
    
    print('Shearlet FWHM MSE   shape: ', MSEerror.shape)
    print('Shearlet FWHM Rmean shape: ', RMean.shape)
    print('Shearlet FWHM RStd  shape:', RStd.shape)
    
    return MSEerror, RMean, RStd

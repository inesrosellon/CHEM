import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import random
import pywt
from torch.utils.data import TensorDataset, DataLoader
import utils

class Wavelet_Hallucination_Index(torch.nn.Module):
    """
    Wavelet-based Hallucination Index computation
    """
    def __init__(self, alpha, theta, wavelet='haar', mode='periodization'):
        super(Wavelet_Hallucination_Index, self).__init__()
        
        self.alpha = alpha
        self.theta = theta
        self.wavelet = wavelet
        self.mode = mode
        self.relu = torch.nn.ReLU()

    def image_to_wavelet_coeffs(self, image):
        """Convert image tensor to wavelet coefficients"""
        # image: [N, C, H, W]
        batch_size, channels, height, width = image.shape
        coeffs_list = []
        
        for b in range(batch_size):
            batch_coeffs = []
            for c in range(channels):
                # Convert to numpy for PyWavelets
                img_np = image[b, c].detach().cpu().numpy()
                try:
                    coeffs = pywt.wavedec2(img_np, self.wavelet, mode=self.mode)
                    # Skip coarsest detail level 
                    coeffs = coeffs[1:]
                    coeffs_flat = []
                    for level in coeffs:
                        if isinstance(level, tuple):
                            for subband in level:
                                coeffs_flat.append(torch.tensor(subband.flatten(), device=image.device))
                        else:
                            coeffs_flat.append(torch.tensor(level.flatten(), device=image.device))
                    batch_coeffs.append(torch.cat(coeffs_flat))
                except Exception as e:
                    print(f"Error in wavelet transform for batch {b}, channel {c}: {e}")
                    print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
            coeffs_list.append(torch.stack(batch_coeffs))
        
        result = torch.stack(coeffs_list)  # [N, C, total_coeffs]
        return result

    def forward(self, target, prediction, interval_radius):
        # Ensure all tensors are on the same device
        device = target.device
        if interval_radius.device != device:
            interval_radius = interval_radius.to(device)
        
        target_coeffs = self.image_to_wavelet_coeffs(target)
        pred_coeffs = self.image_to_wavelet_coeffs(prediction)
        
        Residual = torch.abs(target_coeffs - pred_coeffs)
        
        # Convert interval radius to wavelet domain to match coefficients shape
        
        if len(interval_radius.shape) == 4:
            # Already in correct format [N, C, H, W]
            pass
        else:
            raise ValueError(f"Unexpected interval_radius shape: {interval_radius.shape}")
        
        Radius = self.image_to_wavelet_coeffs(interval_radius)
          
        
        R = -self.relu(self.theta - self.relu(Residual - Radius)) + self.theta
        
        Rm = torch.mean(R, dim=(1, 2))
        Rd = torch.std(R, dim=(1, 2))
            # Expand dimensions
        Rm_expanded = Rm.unsqueeze(1).unsqueeze(2)  # [4] -> [4, 1, 1]
        Rd_expanded = Rd.unsqueeze(1).unsqueeze(2)  # [4] -> [4, 1, 1]
        # (for classification) Normalize R
        #R = (R - Rm_expanded) / Rd_expanded

        return R, Rm, Rd

def wavelet_mse_full(target, prediction, wavelet='haar', mode='periodization'):
    """
    Wavelet-domain MSE that honours Parseval’s theorem.
    Computes the global squared-error over *all* coefficients.
    """
    batch, channels = target.shape[:2]
    total_se   = 0.0   # sum of squared errors
    total_coeff = 0    # number of coefficients

    for b in range(batch):
        for c in range(channels):
            t = target[b, c].detach().cpu().numpy()
            p = prediction[b, c].detach().cpu().numpy()

            t_coeffs = pywt.wavedec2(t, wavelet, mode=mode)
            p_coeffs = pywt.wavedec2(p, wavelet, mode=mode)

            # Flatten every sub-band and accumulate
            for t_lvl, p_lvl in zip(t_coeffs, p_coeffs):
                if isinstance(t_lvl, tuple):          # (H, V, D)
                    for t_sub, p_sub in zip(t_lvl, p_lvl):
                        diff = t_sub - p_sub
                        total_se   += np.sum(diff**2)
                        total_coeff += diff.size
                else:                                 # Approximation band
                    diff = t_lvl - p_lvl
                    total_se   += np.sum(diff**2)
                    total_coeff += diff.size

    return total_se / total_coeff

def Wavelet_Hallucination(model, input_, label_, alpha, interval_radius, fwhm_level, psf, theta, device, wavelet='haar'):
    """Modified Hallucination function using wavelet coefficients with FWHM variation"""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    print("^^^ Wavelet Hallucination device: ", device)
    print(f"^^^ Using wavelet: {wavelet}")
    
    model.to(device)
    
    interval_radius = torch.tensor(interval_radius, dtype=torch.float)
    interval_radius = interval_radius.to(device)
    
    print(f"^^^ Interval radius shape: {interval_radius.shape}")
    print(f"^^^ Input shape: {input_.shape}")
    print(f"^^^ Label shape: {label_.shape}")
    
    MSEerror = np.zeros(len(fwhm_level))
    RMean = np.zeros((len(fwhm_level), input_.shape[0]))
    RStd = np.zeros((len(fwhm_level), input_.shape[0]))
    
    Hmeasure = Wavelet_Hallucination_Index(alpha, theta, wavelet)
    Hmeasure.to(device)
    
    for i in range(len(fwhm_level)):
        
        print(f"Processing FWHM level {i+1}/{len(fwhm_level)}: {fwhm_level[i]}")
        
        # Data Preparation
        noisy_np = input_.cpu().numpy()
        targets_np = label_.cpu().numpy()
        psf_np = psf
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
        
        batch_size =  2
        
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

            target = data_test[1].to(device)
            noisyinput = data_test[0].to(device)
        
            with torch.no_grad():
                pred_ = model(noisyinput)
                
                R, rm, rd = Hmeasure(target, pred_, interval_radius)
                mse = wavelet_mse_full(target, pred_, wavelet)
                
                mseerror.append(mse)
                RM.append(rm.detach().cpu().item())
                RD.append(rd.detach().cpu().item())
                
                del noisyinput, pred_, R, rm, rd,
                
        MSEerror[i] = np.mean(mseerror)
        
        if len(RM) > 0:  # Handle case where debug mode limits samples
            RMean[i, :min(len(RM), RMean.shape[1])] = RM[:RMean.shape[1]]
            RStd[i, :min(len(RD), RStd.shape[1])] = RD[:RStd.shape[1]]
    
    print('Wavelet FWHM MSE   shape: ', MSEerror.shape)
    print('Wavelet FWHM Rmean shape: ', RMean.shape)
    print('Wavelet FWHM RStd  shape:', RStd.shape)
    
    return MSEerror, RMean, RStd

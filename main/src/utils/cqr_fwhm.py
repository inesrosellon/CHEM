# -*- coding: utf-8 -*-

"""
Conformalized Quantile Regression

"""


import warnings

import numpy as np
from scipy import stats, optimize

import torch
import torch.nn as nn
import torch.nn.functional as func
import random
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import interpolate
#import image_utils
from pytorch_msssim import ssim
#from . import utils

from . import generate_dataset_varying_fwhm
from .generate_dataset_varying_fwhm import dataset_fwhm


def get_min_nimgs_calib(alpha):
    """
    Get minimal size for the calibration set (otherwise the adjusted quantile is above 1)
    
    """
    return np.ceil((1 - alpha) / alpha).astype(int)


class BaseCQR:
    """
    Base class for conformalized quantile regression.

    Attributes
    ----------
    alpha (float)
        Target error level

    """
    def __init__(self, alpha):
        self.alpha = alpha


    # def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
    #     raise NotImplementedError
        
    def _conformity_scores(self, pred_calib, label_calib): # res_calib could be anything
        return np.abs(pred_calib - label_calib)


    def _adjusted_quantiles(self, conformity_scores):
        """
        Get quantiles of a given array of conformity scores over a calibration
        set, with finite-sample correction.

        Parameters
        ----------
        conformity_scores (array-like)
            Array of shape (nimgs_calib, nx, ny), where nimgs_calib denotes
            the number of images in the calibration set
        
        Returns
        -------
        quantile_vals (array-like)
            Array of shape (ns, ny): the adjusted quantiles
        adjusted_quantile (float)
            Adjusted quantile index (between 0 and 1)
        
        """
        nimgs_calib = conformity_scores.shape[0]
        assert nimgs_calib >= get_min_nimgs_calib(self.alpha)
        adjusted_quantile = (1 - self.alpha) * (1 + 1/nimgs_calib)
        quantile_vals = np.percentile(conformity_scores, adjusted_quantile*100, axis=0)
        
        interval_radius = quantile_vals
        return interval_radius


    def conformalize(self, pred_calib, label_calib):
        """
        Perform conformal calibration.

        Parameters
        ----------
        res_test (array-like)
            Estimated residuals to be calibrated (test set), shape = (nimgs_test, nx, ny).
        pred_calib, res_calib (array-like)
            Estimated convergence maps and residuals (calibration set),
            shape = (nimgs_calib, nx, ny).
        kappa_calib (array-like)
            Ground-truth convergence maps (calibration set),
            shape = (nimgs_calib, nx, ny).
        
        """
        conformity_scores = self._conformity_scores(pred_calib, label_calib)
        interval_radius = self._adjusted_quantiles(conformity_scores)

        return interval_radius
    

    def get_bounds_proba(self, nimgs_calib):
        lower_bound_proba = self.alpha - 1 / (nimgs_calib + 1)
        upper_bound_proba = self.alpha
        return lower_bound_proba, upper_bound_proba
    
    
class AddCQR:
    """
    Base class for conformalized quantile regression.

    Attributes
    ----------
    alpha (float)
        Target error level

    """
    def __init__(self, alpha):
        self.alpha = alpha


    # def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
    #     raise NotImplementedError
    
    def _calibration_fun(self, interval_test, Lambda):
        return np.maximum(interval_test + Lambda, 0)

    # def _conformity_scores(self, pred_calib, input_calib, interval_calib):
    #     return np.abs(pred_calib - input_calib) - interval_calib
        
    def _conformity_scores(self, pred_calib, label_calib, interval_calib): # res_calib could be anything
        # print('pred_calib ', pred_calib.shape)
        # print('label_calib ', label_calib.shape)
        # print('interval_calib ', interval_calib.shape)
        
        return np.abs(pred_calib - label_calib) - interval_calib


    def _adjusted_quantiles(self, conformity_scores):
        """
        Get quantiles of a given array of conformity scores over a calibration
        set, with finite-sample correction.

        Parameters
        ----------
        conformity_scores (array-like)
            Array of shape (nimgs_calib, nx, ny), where nimgs_calib denotes
            the number of images in the calibration set
        
        Returns
        -------
        quantile_vals (array-like)
            Array of shape (ns, ny): the adjusted quantiles
        adjusted_quantile (float)
            Adjusted quantile index (between 0 and 1)
        
        """
        nimgs_calib = conformity_scores.shape[0]
        assert nimgs_calib >= get_min_nimgs_calib(self.alpha)
        adjusted_quantile = (1 - self.alpha) * (1 + 1/nimgs_calib)
        quantile_vals = np.percentile(conformity_scores, adjusted_quantile*100, axis=0)
        
        best_Lambda = quantile_vals
        return best_Lambda


    def conformalize(self, pred_calib, label_calib, interval_calib, interval_test):
        """
        Perform conformal calibration.

        Parameters
        ----------
        res_test (array-like)
            Estimated residuals to be calibrated (test set), shape = (nimgs_test, nx, ny).
        pred_calib, res_calib (array-like)
            Estimated convergence maps and residuals (calibration set),
            shape = (nimgs_calib, nx, ny).
        kappa_calib (array-like)
            Ground-truth convergence maps (calibration set),
            shape = (nimgs_calib, nx, ny).
        
        """
        conformity_scores = self._conformity_scores(pred_calib, label_calib, interval_calib)
        best_Lambda = self._adjusted_quantiles(conformity_scores)
        
        interval_better = self._calibration_fun(interval_test, best_Lambda)

        return interval_better
    

    def get_bounds_proba(self, nimgs_calib):
        lower_bound_proba = self.alpha - 1 / (nimgs_calib + 1)
        upper_bound_proba = self.alpha
        return lower_bound_proba, upper_bound_proba





def model_inference(model, input_cal0, device):
    print("^^^ model_inference device: ", device)
    
    model.to(device)
    model.eval()
    
    
    restored_cal0 = torch.zeros(input_cal0.size())
    print('inference: input dataset size: ', input_cal0.size())

    process_bs = 100
    for i in range(0, input_cal0.size()[0], process_bs):
        
        if i+process_bs > input_cal0.size()[0]:
            ind = input_cal0.size()[0]
        else:
            ind = i + process_bs

        input_ = input_cal0[i:ind].to(device)
        #print('cal0 bs: ', input_.size())
        
        with torch.no_grad():
            restored_cal0[i:ind] = model(input_)
            
        del input_
        
    return restored_cal0
    

def confidence_radius(model, input_cal0, label_cal0, input_cal1, label_cal1, alpha, device):

    print("^^^ confidence_radius device: ", device)
    print(f"^^^ Calibration sizes: cal0={input_cal0.shape[0]}, cal1={input_cal1.shape[0]}")
    print(f"^^^ Alpha: {alpha}")

    model.to(device)
    
    model.eval()
    
    
    restored_cal0 = model_inference(model, input_cal0, device)
    
    # Convert arrays
    restored_cal0 = np.squeeze(restored_cal0.cpu().detach().numpy())
    #input_cal0 = np.squeeze(input_cal0.permute(0, 2, 3, 1).cpu().detach().numpy())
    label_cal0 = np.squeeze(label_cal0.detach().numpy())
    
    #print('^^^ x_cal shape: ', x_cal.shape)
    
    CQR_initial = BaseCQR(alpha)
    interval_intial = CQR_initial.conformalize(restored_cal0, label_cal0)
    
    shape = label_cal1.shape
    # print('cal1.shape ', shape)
    # print('interval initial ', interval_intial.shape)
    
    interval_intial_un = torch.tensor(interval_intial,dtype=torch.float)
    interval_intial_un = interval_intial_un.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1, -1)
    interval_intial_un = interval_intial_un.numpy()
    
    # print('interval un.shape ', interval_intial_un.shape)
    
    del restored_cal0, input_cal0, label_cal0
    
    # print('input_cal1 ', input_cal1.shape)
    restored_cal1 = model_inference(model, input_cal1, device)
    # print('restored_cal1 ', restored_cal1.shape)
    
    restored_cal1 = np.squeeze(restored_cal1.cpu().detach().numpy())
    label_cal1 = np.squeeze(label_cal1.detach().numpy())
    interval_intial_un = np.squeeze(interval_intial_un)
    
    CQRadd = AddCQR(alpha)
    # print('restored_cal1 ', restored_cal1.shape)
    
    interval_final = CQRadd.conformalize(pred_calib = restored_cal1, label_calib = label_cal1, interval_calib = interval_intial_un, interval_test = interval_intial)
    
    return interval_final



def tPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def tSSIM(tar_img, prd_img):
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True)

def test_measurements(model, input_, label_, device):
    #torch.backends.cudnn.benchmark = True
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    test_dataset = TensorDataset(input_, label_)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False)
    del test_dataset,input_,label_
    
    
    print("^^^ test_measurements device: ", device)
    
    model.to(device)
    
    model.eval()
        
    psnr_rgb = []
    ssim_rgb = []
    
    psnr_in = []
    ssim_in = []
    
    for ii, data_test in enumerate(test_loader, 0):
        
        target = data_test[1].to(device)
        input_ = data_test[0].to(device)
        #shape = target.shape
        
        with torch.no_grad():
            restored = model(input_)
            if restored.size() != target.size():
                restored = interpolate(restored, size=(target.size()[-2], target.size()[-1]), mode='nearest-exact')
        
        
        # print('max input ', torch.max(torch.abs(input_)))
        # print('max label ', torch.max(torch.abs(target)))
        # print('max predi ', torch.max(torch.abs(restored)))
            
        psnr_rgb.append(tPSNR(restored, target).item())
        ssim_rgb.append(tSSIM(restored, target).item())
        
        psnr_in.append(tPSNR(input_, target).item())
        ssim_in.append(tSSIM(input_, target).item())
        
        # for res, tar in zip(restored, target):
        #     psnr_val_rgb.append(utils.torchPSNR(res, tar))
        #     ssim_val_rgb.append(utils.torchSSIM(restored, target))

        del target, input_, data_test,restored
        #free_gpu_cache() 

    
            
    return np.array(psnr_rgb), np.array(ssim_rgb), np.array(psnr_in), np.array(ssim_in),


if __name__ == "__main__":
    
    # pred_calib, input_calib, interval_calib, interval_test
    
    pred_calib     = np.random.rand(1000, 128, 128)  #f(x)
    label_calib    = np.random.rand(1000,128,128)     # r(x) test
    interval_calib = np.random.rand(1000,128,128)     # r(x) test
    interval_test  = np.random.rand(1000,128,128)   # y
    
    #CQR = AddCQR(alpha = 0.01)
    BaseCQR = BaseCQR(alpha = 0.01)
    
    interval_radius = BaseCQR.conformalize(pred_calib, label_calib)
    
    print('interval_radiu ', interval_radius.shape)
    
    AddCQR = AddCQR(alpha = 0.01)
    #interval_better = AddCQR.conformalize(pred_calib, input_calib, interval_calib, interval_test)
    interval_better = AddCQR.conformalize(pred_calib, label_calib, interval_radius, interval_test)

    
    print('interval_better ',interval_better.shape)
    
    print('interval-interval_better ', np.sum(np.abs(interval_radius - interval_better)))
    
    
    
    x = torch.rand(2,2)
    a = torch.repeat_interleave(x,3,dim = 0)
    
    
    a = np.percentile(pred_calib, 99, axis=0)
    
    a = torch.from_numpy(a)
    print('a.shape ', a.shape)
    b = a.unsqueeze(0).unsqueeze(0)
    print('b.shape', b.shape)
    shape = b.shape
    c = b.expand(shape[0], shape[1], -1, -1)
    print('c.shape', c.shape)    
    
    
    
    d = a.unsqueeze(0).unsqueeze(0).expand(1000, 2, -1, -1)
    print('d.shape', d.shape)
    
    i=0
    j=1
    print(d[i,j,:,:]- a)
    
    matrix = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9],
           [10,11,12]]
    
    tensor = torch.tensor(matrix)
    tensor = tensor.float()
    
    print(torch.mean(tensor, dim=1))

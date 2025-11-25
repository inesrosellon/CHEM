# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pickle
import torch
from collections import OrderedDict
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import convolve

from . import generate_psf
from .generate_psf import generate_gaussian_psf

from . import tikho_deconv
from .tikho_deconv import apply_tikhonov_deconv



def varying_psf(noisy, targets, psf, fwhm):
    
    #print(noisy.shape)
    #print(psf.shape)
    #print(targets.shape)
    noise = noisy - convolve(psf[0], targets, mode='same')
    #print(noise.shape)
    
    PSF_new = generate_gaussian_psf(fwhm=fwhm, kernel_size=128)
    
    #print(PSF_new.shape)
    
    input_ = convolve(PSF_new, targets, mode='same') + noise
    
    return input_, PSF_new
    


def dataset_fwhm(noisy, targets, psf, fwhm):
    
    
    x_hat = np.zeros(noisy.shape)
    
    for i in range(noisy.shape[0]):
        input_, PSF_new = varying_psf(noisy[i], targets[i], psf, fwhm)
        input_1 = input_[np.newaxis,:,:]
        x_i = apply_tikhonov_deconv(input_1, PSF_new)
        
        x_hat[i] = x_i[0]
        
    return x_hat
    
    
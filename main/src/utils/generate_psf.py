# -*- coding: utf-8 -*-
"""
"""

import numpy as np  
import matplotlib.pyplot as plt  
from scipy.ndimage import convolve  
from scipy.stats import norm  

  
# Function to generate Gaussian PSF
def generate_gaussian_psf(fwhm, kernel_size=None):  
    sigma = fwhm / 2.355  
    if kernel_size is None:  
        kernel_size = int(np.ceil(5 * sigma))  
        if kernel_size % 2 == 0:  
            kernel_size += 1  
    # x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)  
    # y = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)  
    
    x = np.arange(-(kernel_size // 2), kernel_size // 2 )  
    y = np.arange(-(kernel_size // 2), kernel_size // 2 )  
    
    x_grid, y_grid = np.meshgrid(x, y)  
    psf = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))  
    psf /= np.sum(psf)
    
    
    
    return psf  


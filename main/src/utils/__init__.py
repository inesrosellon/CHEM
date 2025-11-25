from .dir_utils import *
from .image_utils import *
from .model_utils import *
from .dataset_utils import *
from .generate_dataset_varying_fwhm import *

import os
import sys
import importlib
import re

def _get_sweep_from_filename():
    """
    Automatically detect the sweep type from the calling script filename.
    Expected patterns: test_HIC_<sweep>_<method>.py, class_HIC_<sweep>_<method>.py, or epochanalysis_HIC_<sweep>_<method>.py
    """
    # Get the main script filename
    main_script = sys.argv[0] if sys.argv else ""
    filename = os.path.basename(main_script)
    
    # Extract sweep from pattern: (test|class|epochanalysis)_HIC_<sweep>_<method>
    match = re.match(r'(?:test|class|epochanalysis)_HIC_([^_]+)_', filename)
    if match:
        return match.group(1)
    
    # Fallback: check if filename contains known sweep types
    if 'galaxy' in filename:
        return 'galaxy_fwhm'
    elif 'psf' in filename or 'fwhm' in filename:
        return 'psf'
    elif 'noise' in filename:
        return 'noise'
    
    # Default fallback
    return 'noise'

def _get_method_from_filename():
    """
    Automatically detect the method type from the calling script filename.
    Expected patterns: test_HIC_<sweep>_<method>.py, class_HIC_<sweep>_<method>.py, or epochanalysis_HIC_<sweep>_<method>.py
    Returns: 'wavelet', 'shearlet', or 'base'
    """
    # Get the main script filename
    main_script = sys.argv[0] if sys.argv else ""
    filename = os.path.basename(main_script)
    
    # Extract method from pattern: (test|class|epochanalysis)_HIC_<sweep>_<method>
    match = re.match(r'(?:test|class|epochanalysis)_HIC_[^_]+_([^.]+)\.py', filename)
    if match:
        method = match.group(1).lower()
        # Map variations to standard names
        if 'wavelet' in method:
            return 'wavelet'
        elif 'shearlet' in method:
            return 'shearlet'
        elif 'base' in method:
            return 'base'
        else:
            return method
    
    # Additional fallback: check for shearlet in the filename
    if 'shearlet' in filename.lower():
        return 'shearlet'
    elif 'wavelet' in filename.lower():
        return 'wavelet'
    
    return 'base'

# Automatically detect and import the appropriate CQR module
sweep_type = _get_sweep_from_filename()
method_type = _get_method_from_filename()

try:
    if sweep_type == 'noise':
        from .cqr_noise import *
        print(f"Auto-detected: Using CQR for noise level variation (from {os.path.basename(sys.argv[0]) if sys.argv else 'unknown script'})")
        if method_type == 'base':
            from .HIC_noise_base_utils import *
            print("Auto-detected: Using base utilities")
        if method_type == 'wavelet':
            from .HIC_noise_wavelet_utils import *
            print("Auto-detected: Using wavelet utilities")
    elif sweep_type == 'psf':
        from .cqr_fwhm import *
        print(f"Auto-detected: Using CQR for psf- FWHM variation (from {os.path.basename(sys.argv[0]) if sys.argv else 'unknown script'})")
        if method_type == 'base':
            from .HIC_psf_fwhm_base_utils import *
        if method_type == 'wavelet':
            from .HIC_psf_fwhm_wavelet_utils import *
        if method_type == 'shearlet':
            from .HIC_psf_fwhm_shearlet_utils import *
    elif sweep_type == 'galaxy':
        from .cqr_galaxy_fwhm import *
        print(f"Auto-detected: Using CQR for FWHM galaxy variation (from {os.path.basename(sys.argv[0]) if sys.argv else 'unknown script'})")
    else:
        # Default fallback
        from .cqr_noise import *
        print(f"Unknown sweep type '{sweep_type}', defaulting to noise CQR")
except ImportError as e:
    print(f"Warning: Could not import CQR module for '{sweep_type}': {e}")
    try:
        from .cqr_noise import *
        print("Falling back to noise CQR module")
    except ImportError:
        print("Error: No CQR module could be imported") 
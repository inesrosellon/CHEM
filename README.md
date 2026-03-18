# CHEM: Estimating and Understanding Hallucinations in Deep Learning for Image Processing
[![arXiv](https://img.shields.io/badge/arXiv-2512.09806-b31b1b.svg)](
http://arxiv.org/abs/2512.09806)

## Overview

This codebase provides tools for analyzing and visualizing hallucination artifacts in deep learning model predictions. The implementation includes training scripts for multiple architectures (U-Net, SUNet, Learnlets) with different loss functions, evaluation frameworks for hallucination detection, and comprehensive visualization tools. The codebase has been extended beyond astronomical image deconvolution to also cover natural image super-resolution, using models from the [deepinv](https://github.com/deepinv/deepinv) library and the DIV2K dataset.

**Note:** In this codebase, the metric is referred to as "HIC"  or "R", which was later renamed to "CHEM" (Conformal Hallucination Evaluation Metric). Variable names and function references will be updated.
If you have questions regarding this work or want to collaborate, feel free to reach out.

## Repository Structure

```
├── src/
│   ├── train/                    # Training scripts and model definitions
│   ├── eval/                     # Evaluation and analysis scripts
│   ├── models/                   # Trained model weights
│   ├── utils/                    # Utility functions and data processing
│   ├── viz/                      # Visualization and plotting tools
│   ├── A1_Deconvolution.ipynb    # Demo notebook: CHEM analysis for deconvolution
│   └── N2_Superresolution.ipynb  # Demo notebook: CHEM analysis for super-resolution
├── configs/             # Training and evaluation configurations
├── data/               # Training data
└── results/            # Experimental outputs and figures
```

## Key Components

### Training Scripts (`src/train/`)

Training scripts that monitor and log the CHEM throughout the training process, enabling analysis of hallucination evolution during training.

### Evaluation Framework (`src/eval/`)

The `test_HIC` scripts generate the data used in pyramid plot visualizations, while the `class_HIC` scripts perform coefficient classification that enables the reconstruction plots presented in the paper.

### Demo Notebooks (`src/`)

End-to-end notebooks that run the full CHEM analysis pipeline and reproduce the key figures from the paper.

- **`A1_Deconvolution.ipynb`**: CHEM analysis for astronomical image deconvolution on the CANDELS dataset. Covers U-Net, SUNet, LearnLet (L1 and L2 loss variants). 

- **`N2_Superresolution.ipynb`**: CHEM analysis for 4× super-resolution on natural images from the DIV2K dataset. Covers Bicubic, DRUNet-PnP, Unfolded-DRS, RAM, and DPS (Diffusion Posterior Sampling) models, using the [deepinv](https://github.com/deepinv/deepinv) library.

### Visualization Tools (`src/viz/`)
- **`FrequencyClassReconstruction.py`**: Multi-model comparison plots showing predictions and isolated hallucination maps
- **`fwhm_pyramid_plot.py`**: Pyramid visualization of CHEM metrics across different conditions

## Model Weights

Trained model weights should be placed in the `src/models/` directory following this structure:

```
src/models/
├── LearnLet/
│   ├── L1_Loss/
│   │   └── weights.pth
│   └── L2_Loss/
│       └── weights.pth
├── SUNet/
│   ├── L1_Loss/
│   │   └── weights.pth
│   └── L2_Loss/
│       └── weights.pth
└── U-Net/
    ├── L1_Loss/
    │   └── weights.pth
    └── L2_Loss/
        └── weights.pth
```

## Dependencies

The codebase requires separate environments depending on the analysis type:

### For Wavelet Analysis
```bash
pip install -r requirements_wavelets.txt
```

### For Shearlet Analysis
```bash
pip install -r requirements_shearlets.txt
```

### For DeepInv-based Analysis (super-resolution notebooks)
```bash
pip install -r requirements_deepinv.txt
```


## Experimental Reproducibility

If you trained the models from scratch:

1. Install the appropriate dependencies for your analysis type
2. Place the trained model weights in `src/models/` following the specified structure and the datasets in `data/` 
3. Run the evaluation scripts to generate data
4. Use the visualization tools to create the figures shown in the paper

If you are using the deepinv models, you can directly run the N2 Notebook.
## Code Attribution

### External Libraries and Adaptations

- **PyWavelets**: Wavelet transform implementations
- **PyTorch**: Deep learning framework
- **NumPy/SciPy**: Numerical computing libraries
- **Matplotlib**: Visualization library
- **deepinv**: Physics-based deep learning library providing models (DRUNet, DnCNN, DiffUNet, RAM), physics operators, and optimization algorithms used in the super-resolution notebook. https://github.com/deepinv/deepinv
  J. Tachella et al., "DeepInverse: A Python package for solving imaging inverse problems with deep learning," *Journal of Open Source Software*, vol. 10, no. 115, p. 8923, 2025. https://doi.org/10.21105/joss.08923

### Adapted Code

- **Conformalized Quantile Regression (`src/utils/cqr.py`)**  
  Adapted from *H. Leterme and A. Tersenov*, “Weak Lensing Uncertainty Quantification Project,” 2021. https://github.com/hubert-leterme/weaklensing_uq/blob/master/wlmmuq/models/cqr.py .  Original methods described in:  
  [1] H. Leterme, J. Fadili, and J.-L. Starck, “Distribution-free uncertainty quantification for inverse problems: Application to weak lensing mass mapping,” *A&A*, vol. 694, p. A267, Feb. 2025. 

- **SUNet Model, Training Scripts, and Utilities**  
  Adapted from *U. Akhaury*, “SUNet Project,” 2022. https://github.com/utsav-akhaury/SUNet/tree/main .
  Original method described in:  
  [2] K. Fan, S. X. Yu, and X. Jin, “SUNet: Swin Transformer UNet for Image Denoising,” arXiv:2202.14009, 2022.  
  [3] U. Akhaury, P. Jablonka, J.-L. Starck, & F. Courbin, "Ground-based image deconvolution with Swin Transformer UNet," *A&A*, 688, A6, 2024.

- **Learnlet Transform (`src/train/model/LearnLet.py`)**  
  Adapted from *V. Bonjean*, “Learnlet Repository,” 2025. https://github.com/vicbonj/learnlet/blob/main/learnlet.py .
  Original methods described in:  
  [4] A. Ramzi, J. Fadili, and J.-L. Starck, “Learnlets: Learning Wavelets from Data,” arXiv:2008.10317, 2020.


 








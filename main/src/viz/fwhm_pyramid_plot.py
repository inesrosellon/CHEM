# -*- coding: utf-8 -*-
"""
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MaxNLocator
# Add the parent directory (src) to the Python path
current_dir = Path(__file__).resolve().parent  # src/eval/
src_dir = current_dir.parent  # src/
sys.path.insert(0, str(src_dir))

# Add the train directory to the path for model imports
train_dir = src_dir / "train"
sys.path.insert(0, str(train_dir))
import utils
from utils.io import load_paths

def get_data_directories(mode, wavelet_type=None):
    """
    Get data directories based on the selected mode
    
    Args:
        mode: 'pixel', 'wavelet', or 'shearlet'
        wavelet_type: 'haar', 'db4', or 'db8' (required for wavelet mode)
    
    Returns:
        dict: Dictionary with model directories
    """
    paths = load_paths()
    
    if mode == 'pixel':
        # Load from main model directories
        base_dirs = {
            'zu1': paths['checkpoints_dir'] / 'U-Net/L1_Loss/fwhm',
            'zu2': paths['checkpoints_dir'] / 'U-Net/L2_Loss/fwhm', 
            'zs1': paths['checkpoints_dir'] / 'SUNet/L1_Loss/fwhm',
            'zs2': paths['checkpoints_dir'] / 'SUNet/L2_Loss/fwhm',
            'zl1': paths['checkpoints_dir'] / 'LearnLet/L1_Loss/fwhm',
            'zl2': paths['checkpoints_dir'] / 'LearnLet/L2_Loss/fwhm'
        }
    elif mode == 'shearlet':
        # Load from shearlet_3scales_fwhm subdirectories
        base_dirs = {
            'zu1': paths['checkpoints_dir'] / 'U-Net/L1_Loss/shearlet_3scales_fwhm',
            'zu2': paths['checkpoints_dir'] / 'U-Net/L2_Loss/shearlet_3scales_fwhm',
            'zs1': paths['checkpoints_dir'] / 'SUNet/L1_Loss/shearlet_3scales_fwhm',
            'zs2': paths['checkpoints_dir'] / 'SUNet/L2_Loss/shearlet_3scales_fwhm',
            'zl1': paths['checkpoints_dir'] / 'LearnLet/L1_Loss/shearlet_3scales_fwhm',
            'zl2': paths['checkpoints_dir'] / 'LearnLet/L2_Loss/shearlet_3scales_fwhm',
            'dpir': paths['checkpoints_dir'] / 'DPIR_drunet_gray/L2/shearlet_3scales_fwhm'
        }
    elif mode == 'wavelet':
        if wavelet_type is None:
            raise ValueError("Wavelet type must be specified for wavelet mode. Choose from: haar, db4, db8")
        if wavelet_type not in ['haar', 'db4', 'db8']:
            raise ValueError(f"Invalid wavelet type: {wavelet_type}. Choose from: haar, db4, db8")
        
        # Load from wavelet_{type}_fwhm subdirectories
        base_dirs = {
            'zu1': paths['checkpoints_dir'] / f'U-Net/L1_Loss/wavelet_{wavelet_type}_fwhm',
            'zu2': paths['checkpoints_dir'] / f'U-Net/L2_Loss/wavelet_{wavelet_type}_fwhm',
            'zs1': paths['checkpoints_dir'] / f'SUNet/L1_Loss/wavelet_{wavelet_type}_fwhm',
            'zs2': paths['checkpoints_dir'] / f'SUNet/L2_Loss/wavelet_{wavelet_type}_fwhm',
            'zl1': paths['checkpoints_dir'] / f'LearnLet/L1_Loss/wavelet_{wavelet_type}_fwhm',
            'zl2': paths['checkpoints_dir'] / f'LearnLet/L2_Loss/wavelet_{wavelet_type}_fwhm'
        }
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from: pixel, wavelet, shearlet")
    
    return base_dirs


def load_data_for_single_mode(mode, wavelet_type=None):
    """
    Load MSE and RMean data for a single mode
    
    Args:
        mode: 'pixel', 'wavelet', or 'shearlet'
        wavelet_type: 'haar', 'db4', or 'db8' (required for wavelet mode)
    
    Returns:
        tuple: (mse_data, rmean_data) dictionaries
    """
    dirs = get_data_directories(mode, wavelet_type)
    
    print(f"Loading data for mode: {mode}")
    if mode == 'wavelet':
        print(f"Wavelet type: {wavelet_type}")
    
    mse_data = {}
    rmean_data = {}
    
    for key, directory in dirs.items():
        mse_file = directory / "MSE_x.npy"
        rmean_file = directory / "RMean.npy"

        try:
            if mse_file.exists():
                mse_data[key] = np.load(mse_file)
                print(f" Loaded MSE data from: {mse_file}, shape: {mse_data[key].shape}")
            else:
                print(f" MSE file not found: {mse_file}")
                
            if rmean_file.exists():
                rmean_data[key] = np.load(rmean_file)
                print(f" Loaded RMean data from: {rmean_file}, shape: {rmean_data[key].shape}")
            else:
                print(f" RMean file not found: {rmean_file}")
                
        except Exception as e:
            print(f" Error loading data from {directory}: {e}")
    
    return mse_data, rmean_data

def plot_adaptive_pyramid_v2(
    data,
    y_axis_values,
    y_label="y",
    left_axis_label="Left Metric (Units)",
    right_axis_label="Right Metric (Units)",
    title="Comparison Plot",
    left_side_key='Male',
    right_side_key='Female',
    y_invert_axis=True,
    left_marker='o',
    right_marker='s',
    common_linewidth=1.5,
    center_line_color='gray',
    center_line_style='-',
    center_line_width=1.5,
    label_offset_factor=0.04,
    x_tick_count=8, 
    path = 'custom_plot.png'
):
    """
    Plot a two-sided comparison chart with left and right X-axes arranged
    symmetrically and adapted to different data magnitudes on each side.

    Args:
        data (dict): Dictionary containing multiple groups to compare.
        y_axis_values (np.array): Values for the Y axis (e.g. ages or noise levels).
        y_label (str): Label for the Y axis.
        left_axis_label (str): Label for the left-side X axis.
        right_axis_label (str): Label for the right-side X axis.
        title (str): Plot title.
        left_side_key (str): Key name in `data` used for left-side values.
        right_side_key (str): Key name in `data` used for right-side values.
        y_invert_axis (bool): Whether to invert the Y axis.
        left_marker (str): Marker style for the left-side lines.
        right_marker (str): Marker style for the right-side lines.
        common_linewidth (float): Common linewidth for all plotted lines.
        center_line_color (str): Color of the central vertical divider line.
        center_line_style (str): Line style of the central divider.
        center_line_width (float): Line width of the central divider.
        label_offset_factor (float): Factor to adjust vertical position of the
        left/right axis labels placed below the X axis.
        x_tick_count (int): Expected number of ticks per side.
    """

    fig, ax = plt.subplots(figsize=(14, 7))

    #Find the actual min/max values for left and right across all groups
    min_left_actual_val = np.inf
    max_left_actual_val = -np.inf
    min_right_actual_val = np.inf
    max_right_actual_val = -np.inf

    for group_name, group_data in data.items():
        min_left_actual_val = min(min_left_actual_val, np.min(group_data[left_side_key]))
        max_left_actual_val = max(max_left_actual_val, np.max(group_data[left_side_key]))
        min_right_actual_val = min(min_right_actual_val, np.min(group_data[right_side_key]))
        max_right_actual_val = max(max_right_actual_val, np.max(group_data[right_side_key]))

    #Define a common "normalized" display range for both sides
    normalized_max_x = 1.0

    #Define scaling functions to map actual values to normalized display values
    
    # Left side mapping function
    left_range = max_left_actual_val - min_left_actual_val
    left_offset = min_left_actual_val
    # Small epsilon to avoid division by zero if range is 0 (e.g., all data points are same)
    if left_range == 0:
        left_scale = 0 # Data won't change, will be at 0 normalized
        left_offset_norm = 0
    else:
        left_scale = normalized_max_x / left_range
        left_offset_norm = min_left_actual_val * left_scale

    def map_left_to_norm(val):
        return (val - left_offset) * left_scale

    # Right side mapping function
    right_range = max_right_actual_val - min_right_actual_val
    right_offset = min_right_actual_val
    if right_range == 0:
        right_scale = 0
        right_offset_norm = 0
    else:
        right_scale = normalized_max_x / right_range
        right_offset_norm = min_right_actual_val * right_scale

    def map_right_to_norm(val):
        return (val - right_offset) * right_scale


    # Plot data using normalized values
    for group_name, group_data in data.items():
        color = group_data['color']
        linestyle = group_data['linestyle']

        # Normalize data for plotting
        normalized_left_data = map_left_to_norm(group_data[left_side_key])
        normalized_right_data = map_right_to_norm(group_data[right_side_key])

        ax.plot(-normalized_left_data, y_axis_values,
                marker=left_marker, linestyle=linestyle,
                color=color, lw=common_linewidth, label='_nolegend_')

        ax.plot(normalized_right_data, y_axis_values,
                marker=right_marker, linestyle=linestyle,
                color=color, lw=common_linewidth, label='_nolegend_')

    # --- Y-axis Settings ---
    ax.set_ylabel(y_label, fontsize=18, labelpad=10)
    y_min, y_max = np.min(y_axis_values), np.max(y_axis_values)
    y_padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    if y_invert_axis:
        ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    # --- X-axis (Normalized Display) Settings ---
    ax.set_xlim(-(normalized_max_x), (normalized_max_x)) # Each side extends to normalized_max_x

    # Custom X-axis Tick Locations and Labels
    left_nice_ticks = MaxNLocator(nbins=x_tick_count, steps=[1, 2, 2.5, 5, 10]).tick_values(min_left_actual_val, max_left_actual_val)
    right_nice_ticks = MaxNLocator(nbins=x_tick_count, steps=[1, 2, 2.5, 5, 10]).tick_values(min_right_actual_val, max_right_actual_val)
    
    # Filter ticks to be within the actual data range
    left_nice_ticks = left_nice_ticks[(left_nice_ticks >= min_left_actual_val) & (left_nice_ticks <= max_left_actual_val)]
    right_nice_ticks = right_nice_ticks[(right_nice_ticks >= min_right_actual_val) & (right_nice_ticks <= max_right_actual_val)]

    # Convert these nice actual ticks to their normalized display positions
    left_display_ticks = -map_left_to_norm(left_nice_ticks)
    right_display_ticks = map_right_to_norm(right_nice_ticks)

    # Combine all display ticks and remove duplicates (0 will automatically handle)
    all_display_ticks = np.unique(np.concatenate((left_display_ticks, right_display_ticks)))
    # Further filter to ensure ticks are within the set xlim
    all_display_ticks = all_display_ticks[(all_display_ticks >= -normalized_max_x) & (all_display_ticks <= normalized_max_x)]
    ax.set_xticks(all_display_ticks)

    # Custom formatter to display actual values at normalized positions
    def custom_x_formatter(x, pos):
        actual_val = 0.0
        if x < 0: # Left side: need to inverse map from normalized display value to actual value
            if left_scale == 0: return f'{min_left_actual_val:.0f}' # Handle constant data
            actual_val = (abs(x) / left_scale) + left_offset
        else: # Right side: need to inverse map
            if right_scale == 0: return f'{min_right_actual_val:.0f}' # Handle constant data
            actual_val = (x / right_scale) + right_offset
        
        # Optimized formatting logic based on value magnitude
        if actual_val == 0:
            return '0'
        elif actual_val >= 100: # For values >= 100 display integer
            return f'{int(actual_val)}'
        elif actual_val >= 10: # Between 10 and 100: integer or one decimal
            return f'{actual_val:.0f}' if actual_val % 1 == 0 else f'{actual_val:.1f}'
        elif actual_val >= 1: # Between 1 and 10: one decimal
            return f'{actual_val:.1f}' if actual_val % 1 != 0 else f'{int(actual_val)}'
        elif actual_val >= 0.1: # Between 0.1 and 1: two decimals
            return f'{actual_val:.2f}'
        elif actual_val > 0: # Between 0 and 0.1: three decimals
            return f'{actual_val:.3f}'
        else: # Negative numbers or very small zero
            return '0' if actual_val == 0 else f'{actual_val:.1f}' # Should not happen with positive metrics usually


    ax.xaxis.set_major_formatter(FuncFormatter(custom_x_formatter))
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # --- Add the central vertical line ---
    ax.axvline(x=0, color=center_line_color, linestyle=center_line_style,
                     linewidth=center_line_width, alpha=0.8, zorder=0)

    # --- Custom Left/Right Labels below X-axis ---
    x_label_y_fig_coords = ax.xaxis.get_label().get_position()[1] if ax.xaxis.get_label().get_position() else 0.0
    new_y_fig_coords = x_label_y_fig_coords - (label_offset_factor * 2.5)  

    display_y = ax.transAxes.transform((0, new_y_fig_coords))[1]
    data_y = ax.transData.inverted().transform((0, display_y))[1]

    ax.text(-normalized_max_x * 0.5, data_y,
            left_axis_label,
            color='black', ha='center', va='top', fontsize=18,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.text(normalized_max_x * 0.5, data_y,
            right_axis_label,
            color='black', ha='center', va='top', fontsize=18,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # --- Legend ---
    legend_elements = []
    for group_name, group_data in data.items():
        legend_elements.append(
            plt.Line2D([0], [0], color=group_data['color'], linestyle=group_data['linestyle'],
                       lw=common_linewidth, label=group_name)
        )

    legend = ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.25),
                    ncol=3, fancybox=False, shadow=False, frameon=False, title="", fontsize=18)
    #legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(18)
    for text in legend.get_texts():
        text.set_fontweight('normal')

    # --- Title and Layout ---
    ax.set_title(title, fontsize=22, y=1.03, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  

    #plt.show()
    
    
    #plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
 
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate FWHM pyramid plots for different analysis modes')
    parser.add_argument('--mode', choices=['pixel', 'wavelet', 'shearlet'], required=True,
                       help='Analysis mode: pixel (main directories), wavelet (wavelet subdirectories), shearlet (shearlet subdirectories), or all (combined plot)')
    parser.add_argument('--wavelet-type', choices=['haar', 'db4', 'db8'], 
                       help='Wavelet type (required for wavelet mode): haar, db4, or db8')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'wavelet' and args.wavelet_type is None:
        parser.error("--wavelet-type is required when using wavelet mode")
    
    # FWHM levels (consistent across all modes)
    fwhm_level = [10, 20, 30, 40, 50, 60, 70]
    
    mse_data, rmean_data = load_data_for_single_mode(args.mode, args.wavelet_type)

    
    if not mse_data or not rmean_data:
        print("No data found for the specified mode. Please check that the directories exist and contain the required files.")
        return

    for key in rmean_data:
        if rmean_data[key].ndim > 1:
            rmean_data[key] = rmean_data[key].mean(axis=1)
    
    # Validate data dimensions
    expected_length = len(fwhm_level)
    
    # Remove models with incorrect data dimensions
    valid_models = []
    for key in list(mse_data.keys()):
        if key in rmean_data:
            mse_shape = mse_data[key].shape
            rmean_shape = rmean_data[key].shape
            
            print(f"Model {key}: MSE shape {mse_shape}, RMean shape {rmean_shape}")
            
            # Check if data has the right length
            if (len(mse_shape) > 0 and mse_shape[0] == expected_length and 
                len(rmean_shape) > 0 and rmean_shape[0] == expected_length):
                valid_models.append(key)
                print(f" Model {key} has valid dimensions")
            else:
                print(f" Model {key} has invalid dimensions. Expected length {expected_length}")
                del mse_data[key]
                del rmean_data[key]
    
    if not valid_models:
        print("No models with valid data dimensions found!")
        print(f"Expected data length: {expected_length} (for FWHM levels: {fwhm_level})")
        return
    
    # Colors and line styles for plotting
    colors = np.array(
        [[47, 127, 193, 255.],
         [47, 127, 193, 128.],
         [216, 56, 58, 255.],
         [216, 56, 58, 128.],
         [150, 195, 125, 255.],
         [150, 195, 125, 128.],
         [255, 165, 0, 255.]])  # Added orange color for DPIR
    colors = colors/255
    
    linestyles = ['-', '-.', '-', '-.', '-', '-.', '-']  # Added solid line for DPIR
    
    # Model names mapping
    model_mapping = {
        'zu1': 'U-Net, L1',
        'zu2': 'U-Net, L2', 
        'zs1': 'SUNet, L1',
        'zs2': 'SUNet, L2',
        'zl1': 'Learnlets, L1',
        'zl2': 'Learnlets, L2',
        'dpir': 'DPIR+L2'
    }
    
    # Create data dictionary for plotting
    data_dict = {}
    color_idx = 0
    
    for key in valid_models:
        model_name = model_mapping[key]
        data_dict[model_name] = {
            'PSNR': mse_data[key],
            'Stability': rmean_data[key],
            'color': colors[color_idx % len(colors)],  # Use modulo to avoid index errors
            'linestyle': linestyles[color_idx % len(linestyles)]
        }
        color_idx += 1
    
    if not data_dict:
        print("No valid data pairs found for plotting.")
        return
    
    # Generate output filename based on mode
    if args.mode == 'pixel':
        output_filename = "pyramid_FWHM_Hallucination_pixel.png"
    elif args.mode == 'shearlet':
        output_filename = "pyramid_FWHM_Hallucination_shearlet.png"
    elif args.mode == 'wavelet':
        output_filename = f"pyramid_FWHM_Hallucination_wavelet_{args.wavelet_type}.png"
    
    # Create title based on mode
    if args.mode == 'pixel':
        title = "Model Performance: FWHM vs. MSE & CHEM (Pixel Analysis)"
    elif args.mode == 'shearlet':
        title = "Model Performance: FWHM vs. MSE & CHEM (Shearlet Analysis)"
    elif args.mode == 'wavelet':
        if args.wavelet_type == 'db8':
            title = 'db8'
        elif args.wavelet_type == 'db4':
            title = 'db4'
        elif args.wavelet_type == 'haar':
            title = 'haar'
        else:
            title = f'{args.wavelet_type}'
    
    print(f"\nGenerating plot with {len(data_dict)} models...")
    print(f"Output file: {output_filename}")
    
    # Get paths for output
    paths = load_paths()
    
    # Generate the plot
    plot_adaptive_pyramid_v2(
        data=data_dict,
        y_axis_values=fwhm_level,
        y_label="FWHM",
        left_axis_label="MSE",
        right_axis_label="CHEM",
        title=title,
        left_side_key='PSNR',
        right_side_key='Stability',
        y_invert_axis=False,
        left_marker='o',
        right_marker='s',
        path=paths['results_dir'] / output_filename
    )
    
    print(f"Plot saved successfully: {paths['results_dir'] / output_filename}")

if __name__ == "__main__":
    main()


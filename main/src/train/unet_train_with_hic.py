# -*- coding: utf-8 -*-
"""
U-Net Training with HIC Tracking


Usage:
    python unet_train_with_hic.py --loss L1 --epochs 500 --batch-size 4
    python unet_train_with_hic.py --loss L2 --resume
    python unet_train_with_hic.py --help
"""

import os
import sys
from pathlib import Path
import time
import pickle
import argparse

# Add the parent directory (src) to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Lambda
from torch.nn.functional import interpolate

import utils
import numpy as np
import random
from warmup_scheduler.scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter
from model.Unet import UNet

# Import hallucination index computation modules
from utils.HIC_noise_base_utils import Hallucination as BaseHallucination
from utils.HIC_noise_wavelet_utils import Wavelet_Hallucination
from utils.cqr import confidence_radius
import gc

from pathlib import Path
from utils.io import load_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("^^^ device: ", device)

percentage_train = 0.7

def free_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()

free_gpu_cache() 

torch.backends.cudnn.benchmark = True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='U-Net Training with HIC tracking')
    parser.add_argument('--loss', choices=['L1', 'L2'], default='L1', 
                       help='Loss type (default: L1)')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from existing checkpoint')
    parser.add_argument('--epochs', type=int, default=500, 
                       help='Total number of epochs (default: 500)')
    parser.add_argument('--batch-size', type=int, default=4, 
                       help='Batch size (default: 4)')
    parser.add_argument('--hic-every', type=int, default=5, 
                       help='Compute HIC every N epochs (default: 5)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

## Load yaml configuration file
paths = load_paths()
cfg_path = paths['configs_dir'] / f'training_UNet_{args.loss}.yaml'

print(f"Using config: {cfg_path}")

with cfg_path.open("r") as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']
SUNet = opt['SWINUNET']

# Override config with command line arguments
Train['RESUME'] = args.resume
OPT['EPOCHS'] = args.epochs
OPT['BATCH'] = args.batch_size
HIC_EVAL_EVERY = args.hic_every

print(f"Training configuration:")
print(f"  Loss type: {args.loss}")
print(f"  Resume: {args.resume}")
print(f"  Epochs: {args.epochs}")
print(f"  Batch size: {args.batch_size}")
print(f"  HIC evaluation: Every {HIC_EVAL_EVERY} epochs")

## Build Model
print('==> Build the model')
print(SUNet['MODEL_NAME'])

model_restored = UNet(1) 
p_number = utils.network_parameters(model_restored)
model_restored.to(device)

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], SUNet['MODEL_NAME'], Train['LOSS'])
if not os.path.isabs(model_dir):
    project_root = src_dir.parent  
    model_dir = str(project_root / model_dir)
utils.mkdir(model_dir)

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]

## Log
log_dir = os.path.join(Train['SAVE_DIR'], 'log', SUNet['MODEL_NAME'], Train['LOSS'])
# Convert to absolute path if relative
if not os.path.isabs(log_dir):
    # Get project root (HallucinationMeasure folder)
    project_root = src_dir.parent  # Go up from src/ to project root
    log_dir = str(project_root / log_dir)
utils.mkdir(log_dir) 
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume 
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, 'model_latest_ep-%d_bs-%d_ps-%d.pth'%(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE']))
    print(path_chk_rest)
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')
    
    try:
        training_LOSS0 = np.load(os.path.join(model_dir, "train_loss_%d.npy"%(start_epoch-1)))
        var_PSNR0 = np.load(os.path.join(model_dir, "var_PSNR_%d.npy"%(start_epoch-1)))
        var_SSIM0 = np.load(os.path.join(model_dir, "var_SSIM_%d.npy"%(start_epoch-1)))
        
        # Load RMean tracking if available
        base_rmean_0 = np.load(os.path.join(model_dir, "base_rmean_%d.npy"%(start_epoch-1)))
        haar_rmean_0 = np.load(os.path.join(model_dir, "haar_rmean_%d.npy"%(start_epoch-1)))
        db4_rmean_0 = np.load(os.path.join(model_dir, "db4_rmean_%d.npy"%(start_epoch-1)))
        db8_rmean_0 = np.load(os.path.join(model_dir, "db8_rmean_%d.npy"%(start_epoch-1)))
        
        training_LOSS = training_LOSS0.tolist()
        var_PSNR = var_PSNR0.tolist()
        var_SSIM = var_SSIM0.tolist()
        base_rmean_history = base_rmean_0.tolist()
        haar_rmean_history = haar_rmean_0.tolist()
        db4_rmean_history = db4_rmean_0.tolist()
        db8_rmean_history = db8_rmean_0.tolist()
        
        del training_LOSS0, var_PSNR0, var_SSIM0
        del base_rmean_0, haar_rmean_0, db4_rmean_0, db8_rmean_0
        
        print('Loaded existing training history up to epoch: ', len(training_LOSS))
        print('Resuming from epoch: ', start_epoch)
        
    except FileNotFoundError as e:
        print(f"Warning: Training history files not found: {e}")
        print("Starting fresh training history from current checkpoint...")
        training_LOSS = []
        var_PSNR = []
        var_SSIM = []
        base_rmean_history = []
        haar_rmean_history = []
        db4_rmean_history = []
        db8_rmean_history = []
    
else:
    training_LOSS = []
    var_PSNR = []
    var_SSIM = []
    base_rmean_history = []
    haar_rmean_history = []
    db4_rmean_history = []
    db8_rmean_history = []

## Loss
if Train['LOSS'] == 'L1_Loss':
    print(Train['LOSS'])
    train_loss = nn.L1Loss()
elif Train['LOSS'] == 'L2_Loss':
    print(Train['LOSS'])
    train_loss = nn.MSELoss()

## Load training data
print('==> Loading datasets')
x_train = np.load(paths["x_train_file"])
y_train = np.load(paths["y_train_file"])

x_train = x_train[0:10000,:,:,:]
y_train = y_train[0:10000,:,:,:]

# Normalize targets
x_train = x_train - np.mean(x_train, axis=(1,2), keepdims=True)
norm_fact = np.max(x_train, axis=(1,2), keepdims=True) 
x_train /= norm_fact

# Normalize & scale tikho inputs
y_train = y_train - np.mean(y_train, axis=(1,2), keepdims=True)
y_train /= norm_fact

# NCHW convention
x_train = np.moveaxis(x_train, -1, 1)
y_train = np.moveaxis(y_train, -1, 1)

# Convert to torch tensor
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
print(x_train.size(), y_train.size())

# Calculate dataset PSNR statistics 
psnr = []
for i in range(x_train.shape[0]):
    psnr.append(utils.torchPSNR(y_train[i,0,:,:], x_train[i,0,:,:]))
    if i % 1000 == 0:
        print(i)

PSNR = torch.tensor(psnr)

print('DATASET Avg PSNR: ', PSNR.mean())
print('DATASET Max PSNR: ', PSNR.max())
print('DATASET Min PSNR: ', PSNR.min())

free_gpu_cache()

## Load calibration and test data for hallucination index computation
print('==> Loading calibration and test data for HIC')

# Load calibration data
x_cal_full = np.load(paths["x_train_file"])
y_cal_full = np.load(paths["y_train_file"])

x_cal = x_cal_full[10000::,:,:,:]
y_cal = y_cal_full[10000::,:,:,:]

del x_cal_full, y_cal_full

# Normalize calibration data
x_cal = x_cal - np.mean(x_cal, axis=(1,2), keepdims=True)
norm_fact_cal = np.max(x_cal, axis=(1,2), keepdims=True) 
x_cal /= norm_fact_cal

y_cal = y_cal - np.mean(y_cal, axis=(1,2), keepdims=True)
y_cal /= norm_fact_cal

# NCHW convention
x_cal = np.moveaxis(x_cal, -1, 1)
y_cal = np.moveaxis(y_cal, -1, 1)

# Convert to torch tensor
x_cal = torch.tensor(x_cal)
y_cal = torch.tensor(y_cal)

# Split calibration data
n_cal = x_cal.size()[0]
n_cal0 = np.int16(n_cal * 0.5)

y_cal0 = y_cal[0:n_cal0,:,:,:]
x_cal0 = x_cal[0:n_cal0,:,:,:]
y_cal1 = y_cal[n_cal0::,:,:,:]
x_cal1 = x_cal[n_cal0::,:,:,:]

# Load test data for hallucination index
f = open(paths['data_file'], 'rb')
dico = pickle.load(f)
f.close()

y_test = dico['inputs_tikho_laplacian']
x_test = dico['targets']

# Normalize test data
x_test = x_test - np.mean(x_test, axis=(1,2), keepdims=True)
norm_fact_test = np.max(x_test, axis=(1,2), keepdims=True) 
x_test /= norm_fact_test

y_test = y_test - np.mean(y_test, axis=(1,2), keepdims=True)
y_test /= norm_fact_test

# NCHW convention
y_test = np.expand_dims(y_test, 1)
x_test = np.expand_dims(x_test, 1)

# Convert to torch tensor
y_test = torch.tensor(y_test)
x_test = torch.tensor(x_test)

print(f"Test data shapes: y_test={y_test.shape}, x_test={x_test.shape}")

free_gpu_cache()

## Data Augmentation function
def augmentation(im, seed):
    random.seed(seed)
    a = random.choice([0,1,2,3])
    if a==0:
        return im
    elif a==1:
        ch = random.choice([1, 2, 3])
        return torch.rot90(im, ch, [-2,-1])
    elif a==2:
        ch = random.choice([-2, -1])
        return torch.flip(im, [ch])
    elif a==3:
        ch1 = random.choice([10, -10])
        ch2 = random.choice([-2, -1])
        return torch.roll(im, ch1, dims=ch2)

def compute_hallucination_indices(model, epoch):
    """
    Compute hallucination indices for both base and wavelet domains
    """
    print(f"==> Computing hallucination indices for epoch {epoch}")
    
    # Hallucination parameters
    alpha = 0.01
    theta = 1
    input_noise_level = [0] 
    
    model.eval()
    
    try:
        # Compute confidence radius
        print("Computing confidence radius...")
        confidence_interval = confidence_radius(
            model=model, 
            input_cal0=y_cal0, 
            label_cal0=x_cal0, 
            input_cal1=y_cal1, 
            label_cal1=x_cal1, 
            alpha=alpha, 
            device=device
        )

        # Base domain hallucination index
        try:
            base_result = BaseHallucination(
                model=model, 
                input_=y_test, 
                label_=x_test, 
                alpha=alpha, 
                interval_radius=confidence_interval, 
                input_noise_level=input_noise_level, 
                theta=theta, 
                device=device
            )
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        
        if not hasattr(base_result, '__len__') or len(base_result) != 3:
            raise ValueError(f"BaseHallucination returned unexpected result: {type(base_result)}")

        base_mse, base_rmean_array, base_rstd = base_result
        base_rmean_scalar = float(np.mean(base_rmean_array)) 

        # Wavelet domain hallucination indices
        wavelets = ['haar', 'db4', 'db8']
        wavelet_rmeans = {}

        for wavelet in wavelets:
            print(f"Computing {wavelet} wavelet domain HIC...")
            wavelet_result = Wavelet_Hallucination(
                model=model, 
                input_=y_test, 
                label_=x_test, 
                alpha=alpha, 
                interval_radius=confidence_interval, 
                input_noise_level=input_noise_level, 
                theta=theta, 
                device=device,
                wavelet=wavelet
            )
            
            if not hasattr(wavelet_result, '__len__') or len(wavelet_result) != 3:
                raise ValueError(f"Wavelet_Hallucination ({wavelet}) returned unexpected result: {type(wavelet_result)}")

            wv_mse, wv_rmean_array, wv_rstd = wavelet_result
            wavelet_rmeans[wavelet] = float(np.mean(wv_rmean_array))

        model.train()
        return (base_rmean_scalar, wavelet_rmeans['haar'], wavelet_rmeans['db4'], wavelet_rmeans['db8'])

    except Exception:
        model.train()
        import traceback
        traceback.print_exc()
        raise

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}
    HIC evaluation:     Every {HIC_EVAL_EVERY} epochs (base + haar/db4/db8 wavelets)''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

print("==> training device: ", device)


for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    num = 0
    
    n_obj = x_train.size()[0]
    n_train = np.int16(percentage_train*n_obj)

    ind = np.arange(n_obj)
    np.random.shuffle(ind)

    train_dataset = TensorDataset(y_train[ind][:n_train], x_train[ind][:n_train])
    val_dataset = TensorDataset(y_train[ind][n_train:], x_train[ind][n_train:])

    train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                            shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False)

    del train_dataset, val_dataset
    free_gpu_cache()

    model_restored.train()
    for i, data in enumerate(train_loader, 0):
        
        optimizer.zero_grad()
        
        target = data[1].to(device)
        input_ = data[0].to(device)
        
        seed = random.randint(0,1000000)
        target = Lambda(lambda x: torch.stack([augmentation(x_, seed) for x_ in x]))(target)
        input_ = Lambda(lambda x: torch.stack([augmentation(x_, seed) for x_ in x]))(input_)
        restored = model_restored(input_)

        if restored.size() != target.size():
            restored = interpolate(restored, size=(target.size()[-2], target.size()[-1]), mode='nearest-exact')

        loss = train_loss(restored, target)   

        del target, input_, data
        free_gpu_cache() 
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num = num + 1
    
    training_LOSS.append(epoch_loss/num)

    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            
            target = data_val[1].to(device)
            input_ = data_val[0].to(device)
            
            with torch.no_grad():
                restored = model_restored(input_)
                if restored.size() != target.size():
                    restored = interpolate(restored, size=(target.size()[-2], target.size()[-1]), mode='nearest-exact')
            
            psnr_val_rgb.append(utils.torchPSNR(restored, target))
            ssim_val_rgb.append(utils.torchSSIM(restored, target))

            del target, input_, data_val
            free_gpu_cache() 

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
        
        var_PSNR.append(psnr_val_rgb)
        var_SSIM.append(ssim_val_rgb)

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR_ep-{}_bs-{}_ps-{}.pth".format(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE']))) 
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)

    ## Hallucination Index Evaluation
    # Compute HIC at epoch 1 (for early error detection) and then every HIC_EVAL_EVERY epochs
    if epoch == 1 or epoch % HIC_EVAL_EVERY == 0:
        hic_start_time = time.time()
        try:
            base_rmean, haar_rmean, db4_rmean, db8_rmean = compute_hallucination_indices(model_restored, epoch)
    
            base_rmean_history.append(base_rmean)
            haar_rmean_history.append(haar_rmean)
            db4_rmean_history.append(db4_rmean)
            db8_rmean_history.append(db8_rmean)
        
            print(f"[epoch {epoch}] RMean - Base: {base_rmean:.4f}, Haar: {haar_rmean:.4f}, DB4: {db4_rmean:.4f}, DB8: {db8_rmean:.4f}")
            
            # Log to tensorboard
            writer.add_scalar('rmean/base', base_rmean, epoch)
            writer.add_scalar('rmean/haar', haar_rmean, epoch)
            writer.add_scalar('rmean/db4', db4_rmean, epoch)
            writer.add_scalar('rmean/db8', db8_rmean, epoch)
            
        except Exception as e:
            print(f"ERROR: Failed to compute HIC at epoch {epoch}: {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("Continuing training without HIC for this epoch...")
            free_gpu_cache()

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest_ep-{}_bs-{}_ps-{}.pth".format(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE'])))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

    free_gpu_cache() 

writer.close()

total_finish_time = (time.time() - total_start_time)
print('Total training time: {:.1f} seconds'.format((total_finish_time)))

# Save all training history
num = len(training_LOSS)

np.save(os.path.join(model_dir, "train_loss_%d.npy"%(num)), np.array(training_LOSS))
np.save(os.path.join(model_dir, "var_PSNR_%d.npy"%(num)), np.array(var_PSNR))
np.save(os.path.join(model_dir, "var_SSIM_%d.npy"%(num)), np.array(var_SSIM))

# Save RMean histories
np.save(os.path.join(model_dir, "base_rmean_%d.npy"%(num)), np.array(base_rmean_history))
np.save(os.path.join(model_dir, "haar_rmean_%d.npy"%(num)), np.array(haar_rmean_history))
np.save(os.path.join(model_dir, "db4_rmean_%d.npy"%(num)), np.array(db4_rmean_history))
np.save(os.path.join(model_dir, "db8_rmean_%d.npy"%(num)), np.array(db8_rmean_history))

print("==> Training completed!")
print("==> All training histories saved including hallucination indices")
print(f"==> Model saved in: {model_dir}")
print(f"==> To visualize results, run:")
print(f"    python src/viz/plot_training_with_hic.py --model {SUNet['MODEL_NAME']} --loss {args.loss}")
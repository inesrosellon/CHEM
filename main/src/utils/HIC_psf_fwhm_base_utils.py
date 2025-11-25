import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import random
from torch.utils.data import TensorDataset, DataLoader
import utils
class Hallucination_Index(nn.Module):
    r"""

    """
    def __init__(self, alpha, theta):
        super(Hallucination_Index, self).__init__()
        
        self.alpha = alpha
        self.theta = theta
        
        
        self.relu = nn.ReLU()

    def forward(self, target, prediction, interval_radius):
        
        # shape0 = self.res_cqr.shape
        shape = target.shape
        Radius = interval_radius.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1, -1)
        Residual = torch.abs(target-prediction)
        
        R = -self.relu(self.theta - self.relu( Residual - Radius ) ) + self.theta
        
        Rm = torch.mean(R, dim=(1,2,3))
        Rd = torch.std(R, dim=(1,2,3))

        # Expand dimensions
        Rm_expanded = Rm.unsqueeze(1).unsqueeze(2)  # [4] -> [4, 1, 1]
        Rd_expanded = Rd.unsqueeze(1).unsqueeze(2)  # [4] -> [4, 1, 1]
        # (for classification) Normalize R
        #R = (R - Rm_expanded) / Rd_expanded
        
        
        return Rm, Rd

def Hallucination(model, input_, label_, alpha, interval_radius, fwhm_level, psf, theta, device):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    print("^^^ Hallucination device: ", device)
    
    model.to(device)
    
    
    
    interval_radius = torch.tensor(interval_radius, dtype=torch.float)
    interval_radius = interval_radius.to(device)
    
    MSEerror = np.zeros(len(fwhm_level))
    RMean = np.zeros((len(fwhm_level), label_.shape[0]))
    RStd = np.zeros((len(fwhm_level), label_.shape[0]))
    
    
    Hmeasure = Hallucination_Index(alpha, theta)
    
    
    for i in range(len(fwhm_level)):
        
        input_fwhm = utils.dataset_fwhm(noisy=input_, targets=label_, psf=psf, fwhm=fwhm_level[i])
        
        
        
        x_test = label_ - np.mean(label_, axis=(1,2), keepdims=True)
        norm_fact = np.max(x_test, axis=(1,2), keepdims=True) 
        x_test /= norm_fact
        
        # Normalize & scale tikho inputs
        y_test = input_fwhm - np.mean(input_fwhm, axis=(1,2), keepdims=True)
        y_test /= norm_fact
        
        
        y_test = np.expand_dims(y_test.astype(np.float32), 1)
        x_test = np.expand_dims(x_test.astype(np.float32), 1)
        
        # Convert to torch tensor
        y_test = torch.tensor(y_test)
        x_test = torch.tensor(x_test)
        
        
        
        test_dataset = TensorDataset(y_test, x_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0,
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
                
                
                
                rm, rd = Hmeasure(target, pred_ ,interval_radius)

                mse = func.mse_loss(pred_, target, reduction='mean')
                
                mseerror.append(mse.detach().cpu().item())
    
                RM.append(rm.detach().cpu().item())
                RD.append(rd.detach().cpu().item())
    
                
                del noisyinput, pred_, rm, rd, mse
                
        MSEerror[i] = np.mean(mseerror)
        
        RMean[i,:] = RM
        RStd[i,:]  = RD
    
    print('MSE   shape: ', MSEerror.shape)
    print('Rmean shape: ', RMean.shape)
    print('RStd  shape:', RStd.shape)
    
            
    return MSEerror, RMean, RStd

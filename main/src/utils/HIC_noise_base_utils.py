import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import random
from torch.utils.data import TensorDataset, DataLoader

class Hallucination_Index(nn.Module):
    r"""

    """
    def __init__(self, alpha, theta):
        super(Hallucination_Index, self).__init__()
        
        self.alpha = alpha
        self.theta = theta
        
        
        self.relu = nn.ReLU()

    def forward(self, target, prediction, interval_radius):
        
        shape = target.shape
        Radius = interval_radius.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1, -1)
        Residual = torch.abs(target-prediction)
        
        R = -self.relu(self.theta - self.relu( Residual - Radius ) ) + self.theta
        
        Rm = torch.mean(R, dim=(1,2,3))
        Rd = torch.std(R, dim=(1,2,3))
        
        return Rm, Rd

def Hallucination(model, input_, label_, alpha, interval_radius, input_noise_level, theta, device):
    
    random.seed(1234)
    
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    
    print("^^^ Hallucination device: ", device)
    
    model.to(device)
    
    interval_radius = torch.tensor(interval_radius, dtype=torch.float)
    interval_radius = interval_radius.to(device)
    
    MSEerror = np.zeros(len(input_noise_level))
    RMean = np.zeros((len(input_noise_level), input_.shape[0]))
    RStd = np.zeros((len(input_noise_level), input_.shape[0]))
    
    
    shape = input_.shape
    
    input_max = torch.zeros(shape[0])
    
    for i in range(shape[0]):
        input_max[i] = torch.max(torch.abs(input_[i,:,:,:]))
    
    Hmeasure = Hallucination_Index(alpha, theta)
    for i in range(len(input_noise_level)):

        noise = torch.randn(shape[0],shape[1],shape[2],shape[3])
        input_addnoise = input_ + input_noise_level[i] * noise * input_max[:,None, None, None]
        
        test_dataset = TensorDataset(input_addnoise, label_)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                drop_last=False)
        
        del test_dataset, input_addnoise, noise
        
       
        model.eval()
        Hmeasure.eval()
        RM = []
        RD = []
        mseerror = []
        
        for ii, data_test in enumerate(test_loader, 0):
            if ii % 500 == 0:  # Print progress every 500 batches
                print(f"Debug - Processing batch {ii}")
            
            try:
                target = data_test[1].to(device)
                noisyinput = data_test[0].to(device)
            except Exception as e:
                raise
        
            with torch.no_grad():
                
                try:
                    pred_ = model(noisyinput)
                except Exception as e:
                    raise
                
                try:
                    rm, rd = Hmeasure(target, pred_ ,interval_radius)
                except Exception as e:
                    raise
                
                try:
                    mse = func.mse_loss(pred_, target, reduction='mean')
                except Exception as e:
                    raise
                
                try:
                    mseerror.append(mse.detach().cpu().item())
                    
                    RM.append(rm.detach().cpu().item())
                    RD.append(rd.detach().cpu().item())
                except Exception as e:
                    raise
                
                
                del noisyinput, pred_, rm, rd, mse
        
        try:
            MSEerror[i] = np.mean(mseerror)
            
            RMean[i,:] = RM
            RStd[i,:]  = RD
        except Exception as e:
            raise
    
    print('MSE   shape: ', MSEerror.shape)
    print('Rmean shape: ', RMean.shape)
    print('RStd  shape:', RStd.shape)
    
            
    return MSEerror, RMean, RStd


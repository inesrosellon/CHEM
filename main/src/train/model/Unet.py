import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.dconv_down2 = double_conv(64, 128)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.dconv_down3 = double_conv(128, 256)
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.dconv_down4 = double_conv(256, 512)

        
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        # print('down 1 ', x.shape)

        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        # print('down 2 ', x.shape)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        # print('down 3 ', x.shape)
        
        x = self.dconv_down4(x)
        # print('mid ', x.shape)
        
        x = self.upsample3(x)        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        # print('up 1', x.shape)
        
        x = self.upsample2(x)        
        x = torch.cat([x, conv2], dim=1)       
        x = self.dconv_up2(x)
        # print('up 2', x.shape)
        
        x = self.upsample1(x)        
        x = torch.cat([x, conv1], dim=1)   
        x = self.dconv_up1(x)
        # print('up 3', x.shape)
        
        out = self.conv_last(x)
        
        return out
    
if __name__ == "__main__":
    print("Hello, World!")
    
    x = torch.rand(32,1,128,128)
    model = UNet(1)
    y = model(x)
    print(y.shape)

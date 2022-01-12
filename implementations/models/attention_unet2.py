import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch import optim

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/
# https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py #AttentionBlockの実装


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2, diffY//2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttentionBlock,self).__init__()
        self.up_2size = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(F_g,F_g,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(F_g),
			nn.ReLU(inplace=True)
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g0 = self.up_2size(g)
        g1 = self.W_g(g0)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AttentionUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.name = "UNet"
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.att1 = AttentionBlock(1024, 512, 512)
        self.up1 = Up(1024, 512)
        self.att2 = AttentionBlock(512, 256, 256)
        self.up2 = Up(512, 256)
        self.att3 = AttentionBlock(256, 128, 128)
        self.up3 = Up(256, 128)
        self.att4 = AttentionBlock(128, 64, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        a = self.att1(x5, x4)
        x = self.up1(x5, a)
        a = self.att2(x, x3)
        x = self.up2(x, a)
        a = self.att3(x, x2)
        x = self.up3(x, a)
        a = self.att4(x, x1)
        x = self.up4(x, a)
        logits = self.outc(x)
        return logits

def test():
    net = AttentionUNet(3, 1)
    y = net(torch.randn(4, 3, 256, 256))
    print(y.size())

if __name__ == "__main__":
    test()
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch import optim

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/
# https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py #AttentionBlockの実装


class BDoubleConv(nn.Module):
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


class BDownDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class BUpDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
class BDown(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BDownDoubleConv(in_channels, out_channels, dropout_rate)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)
    
class BUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = BUpDoubleConv(in_channels, out_channels, dropout_rate, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = BUpDoubleConv(in_channels, out_channels, dropout_rate)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2, diffY//2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class BOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BOutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class BAttentionBlock(nn.Module):
    def __init__(self,F_g,F_l,F_int, dropout_rate):
        super(BAttentionBlock,self).__init__()
        self.up_2size = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(F_g,F_g,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(F_g),
			nn.ReLU(inplace=True)
        )

        self.W_g = nn.Sequential(
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Dropout2d(p=dropout_rate),
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

class BayesianAttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0.5, bilinear=True):
        super(BayesianAttentionUNet, self).__init__()

        # bilinear = kwargs["bilinear"] if "bilinear" in kwargs.keys() else True

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate

        self.name = "UNet"
        
        self.inc = BDoubleConv(n_channels, 64)
        self.down1 = BDown(64, 128, self.dropout_rate)
        self.down2 = BDown(128, 256, self.dropout_rate)
        self.down3 = BDown(256, 512, self.dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = BDown(512, 1024 // factor, self.dropout_rate)

        self.att1 = BAttentionBlock(1024 // factor, 512, 512, self.dropout_rate)
        self.up1 = BUp(1024, 512 // factor, self.dropout_rate, bilinear)
        self.att2 = BAttentionBlock(512 // factor, 256, 256, self.dropout_rate)
        self.up2 = BUp(512, 256 // factor, self.dropout_rate, bilinear)
        self.att3 = BAttentionBlock(256 // factor, 128, 128, self.dropout_rate)
        self.up3 = BUp(256, 128 // factor, self.dropout_rate, bilinear)
        self.att4 = BAttentionBlock(128 // factor, 64, 64, self.dropout_rate)
        self.up4 = BUp(128, 64, self.dropout_rate, bilinear)
        self.outc = BOutConv(64, n_classes)
        
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

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

def test():
    net = BayesianAttentionUNet(3, 1)
    y = net(torch.randn(4, 3, 256, 256))
    print(y.size())

if __name__ == "__main__":
    test()
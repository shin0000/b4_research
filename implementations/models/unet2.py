import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch import optim

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/
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
        # print("--") # No changes h, w Changes c
        # print(x.shape)
        x = self.double_conv(x)
        # print(x.shape)
        # print("--")
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        # print("--") # Changes h, w, c (c -> c//2)
        # print(x.shape)
        x = self.maxpool_conv(x)
        # print(x.shape)
        # print("--")
        return x
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        # print("--")
        # print(x1.shape, x2.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        # print("--")
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

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.name = "UNet"
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        # print(x.shape)
        x = self.up3(x, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        logits = self.outc(x)
        return logits

def test():
    net = UNet(3, 1)
    y = net(torch.randn(4, 3, 256, 256))
    print(y.size())

if __name__ == "__main__":
    test()

# torch.Size([4, 64, 256, 256])
# torch.Size([4, 128, 128, 128])
# torch.Size([4, 256, 64, 64])
# torch.Size([4, 512, 32, 32])
# torch.Size([4, 1024, 16, 16])
# torch.Size([4, 512, 32, 32])
# torch.Size([4, 256, 64, 64])
# torch.Size([4, 128, 128, 128])
# torch.Size([4, 64, 256, 256])
# torch.Size([4, 1, 256, 256])
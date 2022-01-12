import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch import optim

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/
# https://arxiv.org/pdf/2101.03249.pdf
# https://arxiv.org/pdf/1907.08915.pdf

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
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
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

class BayesianUNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0.5):
        super(BayesianUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.name = "BayesianUNet"
        self.dropout_rate = dropout_rate
        
        self.inc = BDoubleConv(n_channels, 64)
        self.down1 = BDown(64, 128, self.dropout_rate)
        self.down2 = BDown(128, 256, self.dropout_rate)
        self.down3 = BDown(256, 512, self.dropout_rate)
        self.down4 = BDown(512, 1024, self.dropout_rate)
        self.up1 = BUp(1024, 512, self.dropout_rate)
        self.up2 = BUp(512, 256, self.dropout_rate)
        self.up3 = BUp(256, 128, self.dropout_rate)
        self.up4 = BUp(128, 64, self.dropout_rate)
        self.outc = BOutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def test():
    net = BayesianUNet(3, 1)
    net.enable_dropout()
    y = net(torch.randn(4, 3, 256, 256))
    print(y.size())

if __name__ == "__main__":
    test()
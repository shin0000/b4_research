import torch
from torch import nn
from torch.nn import functional as F

try:
    from resnet import *
except:
    from .resnet import *

class DeepLabV3(nn.Module):
    def __init__(self, in_channels, out_channels, backbone="resnet50"):
        super(DeepLabV3, self).__init__()

        self.name = "DeepLabV3"
        null_number = 10
        # backbone = kwargs["backbone"] if "backbone" in kwargs.keys() else "resnet50"

        if backbone[:6] == "resnet":
            self.backbone_out_channels = 2048
            if backbone[6:] == "50":
                self.backbone = ResNet50(in_channels, null_number, last_i_layer=4)
            elif backbone[6:] == "101":
                self.backbone = ResNet101(in_channels, null_number, last_i_layer=4)
            elif  backbone[6:] == "152":
                self.backbone = ResNet152(in_channels, null_number, last_i_layer=4)
            else:
                print("未実装")
        else:
            print("未実装")


        self.classifier = DeepLabHead(self.backbone_out_channels, out_channels)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

def test():
    net = DeepLabV3(in_channels=3, out_channels=1, backbone="resnet50")
    y = net(torch.randn(4, 3, 256, 256))
    print(y.size())

if __name__ == "__main__":
    test()
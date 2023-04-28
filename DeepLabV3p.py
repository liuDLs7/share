import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.transforms import functional as TF

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()

        # Instantiate the ResNet model
        self.resnet_features = models.resnet101(pretrained=True)

        # Instantiate the ASPP module
        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18])

        self.decoder = Decoder(in_channels=256 + 48, out_channels=256, num_classes=num_classes)

        # Add an extra 1x1 conv layer to match the input size to Decoder
        self.extra_conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder: ResNet
        high_level_features = self.resnet_features(x)

        # ASPP module
        high_level_features = self.aspp(high_level_features)
        high_level_features = high_level_features.unsqueeze(2).unsqueeze(3)  # 转换成4D Tensor

        # Extra conv layer
        low_level_features = self.extra_conv(x)
        low_level_features = self.conv1(low_level_features)

        # Decoder
        decoder_output = self.decoder(torch.cat((high_level_features, low_level_features), dim=1))
        x = nn.functional.interpolate(decoder_output, size=x.size()[2:], mode='bilinear', align_corners=True)

        return x
    
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        h, w = x.shape[2:]
        pool = self.global_pool(x)
        pool = nn.functional.interpolate(pool, size=(h, w), mode='bilinear', align_corners=True)
        return pool

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()

        # Error handling of atrous_rates
        if len(atrous_rates) != 3:
            raise ValueError("atrous rates must be length 3.")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0]),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1]),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2]),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv5_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_output = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out5 = self.conv5_pool(x)

        h, w = out5.shape[2:]
        out5 = nn.functional.interpolate(out5, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat((out1, out2, out3, out4, out5), dim=1)
        out = self.conv_output(out)

        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.conv_last = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)

        x = nn.functional.interpolate(x, size=(x.size()[2]*4, x.size()[3]*4), mode='bilinear', align_corners=True)

        x = torch.cat([x, low_level_features], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2d(x)   # 在这里应用 dropout2d

        x = self.conv_last(x)

        return x

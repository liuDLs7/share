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

        self.decoder = Decoder(in_channels=256, out_channels=48)

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

        # 增加一个新的维度
        low_level_features = self.extra_conv(x).unsqueeze(2)

        # Decoder: Upsampling + Concatenation + Convolution
        low_level_features_shape = low_level_features.shape
        high_level_features_shape = high_level_features.shape
        x = self.decoder(torch.cat([low_level_features.expand(
            low_level_features_shape[0], low_level_features_shape[1], high_level_features_shape[2],
            high_level_features_shape[3]
        ), high_level_features], dim=1))
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

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
        super().__init__()

        self.aspp_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.aspp_convs.append(ASPPConv(in_channels, out_channels, rate))

        self.global_pool = ASPPPooling(in_channels, out_channels)

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        aspp_feature = []
        for aspp_conv in self.aspp_convs:
            aspp_feature.append(aspp_conv(x))
        global_feature = self.global_pool(x)
        aspp_feature.append(global_feature)
        aspp_cat = torch.cat(aspp_feature, dim=1)
        output = self.output_conv(aspp_cat)
        return output

    def forward(self, x):
        aspp_feature = []
        for aspp_conv in self.aspp_convs:
            aspp_feature.append(aspp_conv(x))
        global_feature = self.global_pool(x)
        aspp_feature.append(global_feature)
        aspp_cat = torch.cat(aspp_feature, dim=1)
        output = self.output_conv(aspp_cat)
        return output


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1, stride=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        height, width = x.shape[2:]
        pool = nn.functional.avg_pool2d(x, (height, width))
        pool = nn.functional.interpolate(pool, (height, width), mode='bilinear', align_corners=True)

        x = torch.cat([x, pool], dim=1)
        x = self.last_conv(x)

        return x
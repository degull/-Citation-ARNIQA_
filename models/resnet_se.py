import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.attention_se import DistortionAttention, HardNegativeCrossAttention

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetSE(nn.Module):
    def __init__(self):
        super(ResNetSE, self).__init__()
        base_model = resnet50(pretrained=True)
        self.layer0 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Distortion Attention and SE Modules
        self.distortion_attention1 = DistortionAttention(256)
        self.distortion_attention2 = DistortionAttention(512)
        self.distortion_attention3 = DistortionAttention(1024)
        self.distortion_attention4 = DistortionAttention(2048)

        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

        # Hard Negative Cross Attention
        self.hard_negative_attention = HardNegativeCrossAttention(2048)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Layer 0
        x = self.layer0(x)

        # Layer 1
        x = self.layer1(x)
        x = self.distortion_attention1(x)
        x = self.se1(x)

        # Layer 2
        x = self.layer2(x)
        x = self.distortion_attention2(x)
        x = self.se2(x)

        # Layer 3
        x = self.layer3(x)
        x = self.distortion_attention3(x)
        x = self.se3(x)

        # Layer 4
        x = self.layer4(x)
        x_attr = self.distortion_attention4(x)
        x_texture = self.se4(x)
        x = self.hard_negative_attention(x_attr, x_texture)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)  # (batch_size, 2048)

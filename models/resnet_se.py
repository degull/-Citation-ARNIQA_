# ver1
""" 
import torch
import torch.nn as nn
from torchvision.models import resnet50

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
    def __init__(self, encoder_params=None):
        super(ResNetSE, self).__init__()
        base_model = resnet50(pretrained=True)
        
        # Extract layers from ResNet-50
        self.layer0 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Integrate SE blocks after each ResNet layer
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)
        return x
     """

# ver2

import torch
import torch.nn as nn
from torchvision.models import resnet50

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, use_depthwise=False):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_depthwise:
            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, groups=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, groups=in_channels),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x, intensity=None):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if intensity is not None:
            y = y * intensity.view(b, 1, 1, 1)  # Distortion Intensity 기반 활성화
        return x * y


class ResNetSE(nn.Module):
    def __init__(self, encoder_params=None):
        super(ResNetSE, self).__init__()
        base_model = resnet50(pretrained=True)

        # ResNet-50의 기본 구조 (conv1, layer1~layer4)
        self.layer0 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # 각 ResNet 레이어 출력 뒤에 SEBlock 추가
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

    def forward(self, x):
        x = self.layer0(x)
        print(f"[Debug] After layer0: {x.shape}")

        # Layer 1
        x = self.layer1(x)
        print(f"[Debug] After layer1: {x.shape}")
        x = self.se1(x)
        print(f"[Debug] After SE1: {x.shape}")

        # Layer 2
        x = self.layer2(x)
        print(f"[Debug] After layer2: {x.shape}")
        x = self.se2(x)
        print(f"[Debug] After SE2: {x.shape}")

        # Layer 3
        x = self.layer3(x)
        print(f"[Debug] After layer3: {x.shape}")
        x = self.se3(x)
        print(f"[Debug] After SE3: {x.shape}")

        # Layer 4
        x = self.layer4(x)
        print(f"[Debug] After layer4: {x.shape}")
        x = self.se4(x)
        print(f"[Debug] After SE4: {x.shape}")

        return x





# resnet_se.py

# 각 레이어 뒤에 SEBlock을 추가하여 채널별 중요도 학습을 강화
# 기존 ResNet보다 채널별 중요도를 더 잘 반영 """
""" 
import torch
import torch.nn as nn
from torchvision.models import resnet50

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
        self.layer0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)
        return x
 """
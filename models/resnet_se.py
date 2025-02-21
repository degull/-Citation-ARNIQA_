import sys
import os

# ✅ 프로젝트 루트 경로를 추가 (models 폴더를 찾을 수 있도록 설정)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.attention_se import FeatureLevelAttention, HardNegativeCrossAttention

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
        if isinstance(x, tuple):
            x = x[0]  
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  




class ResNetSE(nn.Module):
    def __init__(self, dataset_type="synthetic"):
        super(ResNetSE, self).__init__()
        self.dataset_type = dataset_type
        base_model = resnet50(weights="IMAGENET1K_V1")

        self.layer0 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.feature_attention = FeatureLevelAttention(2048, 2048)
        self.hard_negative_attention = HardNegativeCrossAttention(2048)

    def forward(self, x, hard_neg=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.feature_attention(x)

        if self.dataset_type == "synthetic" and hard_neg is not None:
            x = self.hard_negative_attention(x, hard_neg)

        return x





if __name__ == "__main__":
    model = ResNetSE()
    x = torch.randn(32, 3, 224, 224)
    output = model(x)
    print("Output Shape:", output.shape)  # (32, 2048)

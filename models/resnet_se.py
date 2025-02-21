import sys
import os

# ✅ 프로젝트 루트 경로를 추가 (models 폴더를 찾을 수 있도록 설정)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.attention_se import FeatureLevelAttention, HardNegativeCrossAttention

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_dim = max(in_channels // reduction, 8)  # 최소 크기 보장

        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # ✅ 현재 x의 채널 크기 가져오기
        y = self.global_avg_pool(x).view(b, c)  # ✅ (batch, channels)
        y = self.fc(y).view(b, c, 1, 1)  # ✅ 다시 (batch, channels, 1, 1)로 변환
        return x * y  # ✅ 원본 x에 가중치 적용


class ResNetSE(nn.Module):
    def __init__(self, use_hnca=False):
        super(ResNetSE, self).__init__()
        base_model = resnet50(weights="IMAGENET1K_V1")

        self.layer0 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # ✅ Feature-Level Attention 적용
        self.feature_attention1 = FeatureLevelAttention(256, 256)
        self.feature_attention2 = FeatureLevelAttention(512, 512)
        self.feature_attention3 = FeatureLevelAttention(1024, 1024)
        self.feature_attention4 = FeatureLevelAttention(2048, 2048)

        # ✅ SE Blocks
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

        # ✅ Hard Negative Cross Attention 적용 (Layer 4 이후)
        self.use_hnca = use_hnca
        if use_hnca:
            self.hard_negative_attention = HardNegativeCrossAttention(2048)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.layer0(x)

        # ✅ Layer 1
        x = self.layer1(x)
        x = self.feature_attention1(x)
        x = self.se1(x)

        # ✅ Layer 2
        x = self.layer2(x)
        x = self.feature_attention2(x)
        x = self.se2(x)

        # ✅ Layer 3
        x = self.layer3(x)
        x = self.feature_attention3(x)
        x = self.se3(x)

        # ✅ Layer 4
        x = self.layer4(x)  # ✅ 4D 유지됨
        x_attr = self.feature_attention4(x)  # ✅ 4D 유지
        x_texture = self.se4(x)  # ✅ 4D 유지

        # ✅ Hard Negative Cross Attention 적용
        if self.use_hnca:
            x = self.hard_negative_attention(x_attr, x_texture)  # ✅ 4D 유지됨
        else:
            x = x_attr  # HNCA 사용 안 할 경우 feature attention 적용된 것만 사용

        return x  # ✅ 4D (batch, 2048, 7, 7) 유지됨



if __name__ == "__main__":
    model = ResNetSE()
    x = torch.randn(32, 3, 224, 224)
    output = model(x)
    print("Output Shape:", output.shape)  # (32, 2048)

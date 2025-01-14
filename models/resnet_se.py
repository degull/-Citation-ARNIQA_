import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.attention_se import DistortionAttention, HardNegativeCrossAttention

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: 채널 별 전역 평균 계산
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),   # 채널 축소
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),   # 채널 복원
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()   # (배치 크기, 채널, 높이, 너비)
        y = self.global_avg_pool(x).view(b, c)  # Squeeze 단계
        y = self.fc(y).view(b, c, 1, 1) # Excitation 단계
        return x * y    # 입력 텐서와 채널 중요도 스칼라 곱


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
        print(f"Layer0 output: {x.size()}")  # 출력 크기 확인

        # Layer 1
        x = self.layer1(x)
        x = self.distortion_attention1(x)
        x = self.se1(x)
        print(f"Layer1 output: {x.size()}")

        # Layer 2
        x = self.layer2(x)
        x = self.distortion_attention2(x)
        x = self.se2(x)
        print(f"Layer2 output: {x.size()}")

        # Layer 3
        x = self.layer3(x)
        x = self.distortion_attention3(x)
        x = self.se3(x)
        print(f"Layer3 output: {x.size()}")

        # Layer 4
        x = self.layer4(x)
        x_attr = self.distortion_attention4(x)
        x_texture = self.se4(x)
        print(f"Layer4 output (before attention): {x.size()}")

        x = self.hard_negative_attention(x_attr, x_texture)
        print(f"Layer4 output (after attention): {x.size()}")

        # Global Average Pooling
        x = self.global_avg_pool(x)
        print(f"Global Avg Pool output: {x.size()}")
        return x.view(x.size(0), -1)  # (batch_size, 2048)


if __name__ == "__main__":
    x = torch.randn(32, 256, 28, 28)  # 배치 크기 32, 채널 256, 크기 28x28
    se_block = SEBlock(256, reduction=16)  # 채널 축소 비율 16
    output = se_block(x)

    print("입력 크기:", x.size())  # [32, 256, 28, 28]
    print("출력 크기:", output.size())  # [32, 256, 28, 28]
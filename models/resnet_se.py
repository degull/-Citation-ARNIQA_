# ver1
import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.attention_se import DistortionAttention, HardNegativeCrossAttention
from utils.utils_visualization import visualize_feature_maps

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
        # 튜플 처리
        if isinstance(x, tuple):
            x = x[0]  # 첫 번째 요소를 사용
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
        print(f"Layer0 output: {x.size()}")  # 출력 크기 확인

        # Layer 1
        x = self.layer1(x)
        x = self.distortion_attention1(x)
        if isinstance(x, tuple):  # 튜플 처리
            x = x[0]
        x = self.se1(x)
        print(f"Layer1 output: {x.size()}")

        # Layer 2
        x = self.layer2(x)
        x = self.distortion_attention2(x)
        if isinstance(x, tuple):  # 튜플 처리
            x = x[0]
        x = self.se2(x)
        print(f"Layer2 output: {x.size()}")

        # Layer 3
        x = self.layer3(x)
        x = self.distortion_attention3(x)
        if isinstance(x, tuple):  # 튜플 처리
            x = x[0]
        x = self.se3(x)
        print(f"Layer3 output: {x.size()}")

        # Layer 4
        x = self.layer4(x)
        x_attr = self.distortion_attention4(x)
        if isinstance(x_attr, tuple):  # 튜플 처리
            x_attr = x_attr[0]
        x_texture = self.se4(x)
        if isinstance(x_texture, tuple):  # 튜플 처리
            x_texture = x_texture[0]
        print(f"Layer4 output (before attention): {x.size()}")

        x = self.hard_negative_attention(x_attr, x_texture)
        print(f"Layer4 output (after attention): {x.size()}")

        # Global Average Pooling
        x = self.global_avg_pool(x)
        print(f"Global Avg Pool output: {x.size()}")
        return x.view(x.size(0), -1)  # (batch_size, 2048)


class ResNetSEVisualizer(nn.Module):
    def __init__(self, base_model, distortion_attentions, hard_negative_attention, se_blocks):
        super(ResNetSEVisualizer, self).__init__()

        # 🔥 ResNet은 layer0이 없으므로 직접 생성해야 함
        self.layer0 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.distortion_attention1, self.distortion_attention2, self.distortion_attention3, self.distortion_attention4 = distortion_attentions
        self.se1, self.se2, self.se3, self.se4 = se_blocks
        self.hard_negative_attention = hard_negative_attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, original_image):
        """
        :param x: 입력 이미지 텐서
        :param original_image: 원본 이미지 (PIL or numpy)
        """
        activation_maps = {}

        # Layer 0 (Conv + MaxPool)
        x = self.layer0(x)
        activation_maps["Layer0"] = x

        # Layer 1
        x = self.layer1(x)
        x = self.distortion_attention1(x)[0]  # Distortion Attention 적용
        x = self.se1(x)
        activation_maps["Layer1"] = x

        # Layer 2
        x = self.layer2(x)
        x = self.distortion_attention2(x)[0]
        x = self.se2(x)
        activation_maps["Layer2"] = x

        # Layer 3
        x = self.layer3(x)
        x = self.distortion_attention3(x)[0]
        x = self.se3(x)
        activation_maps["Layer3"] = x

        # Layer 4 (Before Attention)
        x = self.layer4(x)
        activation_maps["Layer4_before_attention"] = x

        # Distortion Attention & SE
        x_attr = self.distortion_attention4(x)[0]
        x_texture = self.se4(x)
        activation_maps["Layer4_after_attention"] = x_attr

        # Hard Negative Cross Attention
        x = self.hard_negative_attention(x_attr, x_texture)
        activation_maps["HardNegativeCrossAttention"] = x

        # Global Avg Pooling 적용 후 벡터 출력
        x = self.global_avg_pool(x)
        activation_maps["GlobalAvgPool"] = x

        return activation_maps


if __name__ == "__main__":
    x = torch.randn(32, 256, 28, 28)  # 배치 크기 32, 채널 256, 크기 28x28
    se_block = SEBlock(256, reduction=16)  # 채널 축소 비율 16
    output = se_block(x)

    print("입력 크기:", x.size())  # [32, 256, 28, 28]
    print("출력 크기:", output.size())  # [32, 256, 28, 28]


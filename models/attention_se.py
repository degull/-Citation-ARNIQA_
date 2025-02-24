# 기존 논문 방법

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image




# ✅ VGG-16을 활용한 특징 추출
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # VGG-16에서 필요한 특징 추출 계층
        self.conv1_2 = nn.Sequential(*vgg16[:4])  # Conv1_2
        self.conv2_2 = nn.Sequential(*vgg16[4:9])  # Conv2_2
        self.conv3_3 = nn.Sequential(*vgg16[9:16])  # Conv3_3
        self.conv4_3 = nn.Sequential(*vgg16[16:23])  # Conv4_3
        self.conv5_3 = nn.Sequential(*vgg16[23:30])  # Conv5_3

    def forward(self, x):
        feat1 = self.conv1_2(x)  # (B, 64, 256, 256)
        feat2 = self.conv2_2(feat1)  # (B, 128, 128, 128)
        feat3 = self.conv3_3(feat2)  # (B, 256, 64, 64)
        feat4 = self.conv4_3(feat3)  # (B, 512, 32, 32)
        feat5 = self.conv5_3(feat4)  # (B, 512, 16, 16)

        return feat1, feat2, feat3, feat4, feat5


# ✅ Context-aware Pyramid Feature Extraction (CPFE 1)
class CPFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPFE, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

        self.fuse_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3(x)
        feat3 = self.conv5x5(x)

        fused = torch.cat([feat1, feat2, feat3], dim=1)
        return self.fuse_conv(fused)

# ✅ Context-aware Pyramid Feature Extraction (CPFE 2)
""" class CPFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPFE, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_r3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv3x3_r5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        self.conv3x3_r7 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=7, dilation=7)

        self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_r3(x)
        feat3 = self.conv3x3_r5(x)
        feat4 = self.conv3x3_r7(x)

        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.fuse_conv(fused) """



# ✅ Spatial Attention (SA) for Distortion Detection (1)
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)

    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn
    
# ✅ Spatial Attention (SA) for Distortion Detection (2)
""" class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        self.conv1xk = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 7), padding=(0, 3))
        self.convkx1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(7, 1), padding=(3, 0))
        self.convk1 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x):
        attn_h = self.conv1xk(x)
        attn_w = self.convkx1(x)
        attn = torch.sigmoid(self.convk1(attn_h + attn_w))
        return x * attn """



# ✅ Channel-wise Attention (CA) for Distortion Detection
class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()
        self.linear_1 = nn.Linear(in_channels, in_channels // 4)
        self.linear_2 = nn.Linear(in_channels // 4, in_channels)

    def forward(self, x):
        n_b, n_c, _, _ = x.size()
        avg_feats = F.adaptive_avg_pool2d(x, (1, 1)).view(n_b, n_c)
        std_feats = torch.std(x.view(n_b, n_c, -1), dim=2)
        std_feats = std_feats / (std_feats.max(dim=1, keepdim=True)[0] + 1e-6)
        combined_feats = avg_feats + std_feats
        combined_feats = F.relu(self.linear_1(combined_feats))
        combined_feats = torch.sigmoid(self.linear_2(combined_feats))
        return combined_feats.view(n_b, n_c, 1, 1).expand_as(x)


# ✅ Hard Negative Cross Attention (HNCA)
class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(HardNegativeCrossAttention, self).__init__()
        self.in_dim = in_dim
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.output_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)

        attn_scores = torch.softmax(Q @ K.transpose(-2, -1) / (self.in_dim ** 0.5), dim=-1)
        attn_output = attn_scores @ V
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)

        return self.output_proj(attn_output + x)


# ✅ 최종 모델
class DistortionDetectionModel(nn.Module):
    def __init__(self):
        super(DistortionDetectionModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.sa = SpatialAttention(64)
        self.cpfe = CPFE(512, 64)
        self.ca = ChannelwiseAttention(64)
        self.hnca = HardNegativeCrossAttention(64)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        low_feat = self.sa(feat1) * feat1
        high_feat = self.hnca(self.cpfe(feat5))
        high_feat = self.ca(high_feat) * high_feat
        high_feat = self.upsample(high_feat)

        print(f"Low_feat: {low_feat.shape}, High_feat: {high_feat.shape}")  # ✅ 크기 확인

        fused_feat = torch.cat([low_feat, high_feat], dim=1)
        output = self.final_conv(fused_feat)
        return output.view(output.shape[0], -1).mean(dim=1)  # ✅ (batch_size, 1, 224, 224) → (batch_size,)



def distortion_loss(pred, gt):
    mse_loss = nn.MSELoss()(pred, gt)  # ✅ MOS 값과 직접 비교
    perceptual_loss = torch.mean(torch.abs(pred - gt))  # ✅ L1 Loss 추가
    return mse_loss + 0.1 * perceptual_loss


# ✅ 테스트 코드
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)  # ✅ 입력 크기 224x224로 설정
    dummy_gt = torch.randn(2)  # ✅ MOS 점수 (batch_size,) 형태로 생성
    model = DistortionDetectionModel()
    output = model(dummy_input)
    
    loss = distortion_loss(output, dummy_gt)  # ✅ 크기 맞춤 후 손실 계산

    print("Model Output Shape:", output.shape)  # ✅ (batch_size,)가 출력되어야 함
    print("Loss:", loss.item())


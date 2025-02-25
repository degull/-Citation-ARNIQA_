""" import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import DropPath


# ✅ VGG-16을 활용한 Feature Extractor
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.conv1_2 = nn.Sequential(*vgg16[:4])  # Conv1_2
        self.conv2_2 = nn.Sequential(*vgg16[4:9])  # Conv2_2
        self.conv3_3 = nn.Sequential(*vgg16[9:16])  # Conv3_3
        self.conv4_3 = nn.Sequential(*vgg16[16:23])  # Conv4_3
        self.conv5_3 = nn.Sequential(*vgg16[23:30])  # Conv5_3

    def forward(self, x):
        feat1 = self.conv1_2(x)
        feat2 = self.conv2_2(feat1)
        feat3 = self.conv3_3(feat2)
        feat4 = self.conv4_3(feat3)
        feat5 = self.conv5_3(feat4)

        return feat1, feat2, feat3, feat4, feat5


# ✅ Context-aware Pyramid Feature Extraction (CPFE)
class CPFE(nn.Module):
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
        return self.fuse_conv(fused)


# ✅ CBAM (Convolutional Block Attention Module)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel Attention
        avg_out = self.mlp(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.mlp(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_attn = torch.sigmoid(avg_out + max_out)
        x = x * channel_attn

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))

        return x * spatial_attn


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
class EnhancedDistortionDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedDistortionDetectionModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.cbam = CBAM(64)
        self.cpfe = CPFE(512, 64)
        self.hnca = HardNegativeCrossAttention(64)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        low_feat = self.cbam(feat1) * feat1
        high_feat = self.hnca(self.cpfe(feat5))

        high_feat = self.upsample(high_feat)
        fused_feat = torch.cat([low_feat, high_feat], dim=1)
        output = self.final_conv(fused_feat)

        return output.view(output.shape[0], -1).mean(dim=1)


# ✅ 손실 함수
def distortion_loss(pred, gt):
    mse_loss = nn.MSELoss()(pred, gt)
    perceptual_loss = torch.mean(torch.abs(pred - gt))
    return mse_loss + 0.1 * perceptual_loss


# ✅ 테스트 코드
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)  # ✅ 입력 크기 224x224로 설정
    dummy_gt = torch.randn(2)  # ✅ MOS 점수 (batch_size,) 형태로 생성
    model = EnhancedDistortionDetectionModel()
    output = model(dummy_input)

    loss = distortion_loss(output, dummy_gt)

    print("Model Output Shape:", output.shape)  # ✅ (batch_size,)가 출력되어야 함
    print("Loss:", loss.item())
 """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import DropPath


# ✅ VGG-16을 활용한 Feature Extractor
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.conv1_2 = nn.Sequential(*vgg16[:4])  # Conv1_2
        self.conv2_2 = nn.Sequential(*vgg16[4:9])  # Conv2_2
        self.conv3_3 = nn.Sequential(*vgg16[9:16])  # Conv3_3
        self.conv4_3 = nn.Sequential(*vgg16[16:23])  # Conv4_3
        self.conv5_3 = nn.Sequential(*vgg16[23:30])  # Conv5_3

    def forward(self, x):
        feat1 = self.conv1_2(x)
        feat2 = self.conv2_2(feat1)
        feat3 = self.conv3_3(feat2)
        feat4 = self.conv4_3(feat3)
        feat5 = self.conv5_3(feat4)

        return feat1, feat2, feat3, feat4, feat5


# ✅ Context-aware Pyramid Feature Extraction (CPFE)
class CPFE(nn.Module):
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
        return self.fuse_conv(fused)


# ✅ CoordAttention (Coordinate Attention) - 2021
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CoordAttention, self).__init__()

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Height 방향
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))  # Width 방향

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // reduction)

        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # ✅ Height 방향과 Width 방향 평균 풀링
        h_attn = self.avg_pool_h(x).permute(0, 1, 3, 2)  # (B, C, 1, H) → (B, C, H, 1)
        w_attn = self.avg_pool_w(x)  # (B, C, 1, W)

        # ✅ 공유된 Convolution 적용
        shared_feat = torch.cat([h_attn, w_attn], dim=2)  # (B, C, H+W, 1)
        shared_feat = self.conv1(shared_feat)
        shared_feat = self.bn(shared_feat)
        shared_feat = F.relu(shared_feat)

        # ✅ Height / Width 방향 분리 후 각각 Attention 적용 (자동 split)
        split_size = shared_feat.shape[2] // 2  # 🔧 자동 계산된 크기로 split
        h_attn, w_attn = torch.split(shared_feat, [split_size, split_size], dim=2)  # ✅ 오류 해결

        h_attn = self.conv_h(h_attn.permute(0, 1, 3, 2))  # (B, C, H, 1) → (B, C, 1, H)
        w_attn = self.conv_w(w_attn)  # (B, C, 1, W)

        attn = torch.sigmoid(h_attn + w_attn)  # 최종 Attention Map
        return x * attn  # 입력 Feature에 Attention 적용

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
class EnhancedDistortionDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedDistortionDetectionModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.coord_attn = CoordAttention(64)  # ✅ CBAM → CoordAttention 변경
        self.cpfe = CPFE(512, 64)
        self.hnca = HardNegativeCrossAttention(64)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        low_feat = self.coord_attn(feat1) * feat1  # ✅ CBAM → CoordAttention 변경
        high_feat = self.hnca(self.cpfe(feat5))

        high_feat = self.upsample(high_feat)
        fused_feat = torch.cat([low_feat, high_feat], dim=1)
        output = self.final_conv(fused_feat)

        output = torch.sigmoid(output)

        return output.view(output.shape[0], -1).mean(dim=1)



# ✅ 손실 함수
def distortion_loss(pred, gt):
    mse_loss = nn.MSELoss()(pred, gt)
    perceptual_loss = torch.mean(torch.abs(pred - gt))
    return mse_loss + 0.1 * perceptual_loss


# ✅ 테스트 코드
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)  # ✅ 입력 크기 224x224로 설정
    dummy_gt = torch.randn(2)  # ✅ MOS 점수 (batch_size,) 형태로 생성
    model = EnhancedDistortionDetectionModel()
    output = model(dummy_input)

    loss = distortion_loss(output, dummy_gt)

    print("Model Output Shape:", output.shape)  # ✅ (batch_size,)가 출력되어야 함
    print("Loss:", loss.item())




#  수정모델3
""" import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import DropPath


# ✅ VGG-16을 활용한 Feature Extractor
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.conv1_2 = nn.Sequential(*vgg16[:4])  # Conv1_2
        self.conv2_2 = nn.Sequential(*vgg16[4:9])  # Conv2_2
        self.conv3_3 = nn.Sequential(*vgg16[9:16])  # Conv3_3
        self.conv4_3 = nn.Sequential(*vgg16[16:23])  # Conv4_3
        self.conv5_3 = nn.Sequential(*vgg16[23:30])  # Conv5_3

    def forward(self, x):
        feat1 = self.conv1_2(x)
        feat2 = self.conv2_2(feat1)
        feat3 = self.conv3_3(feat2)
        feat4 = self.conv4_3(feat3)
        feat5 = self.conv5_3(feat4)

        return feat1, feat2, feat3, feat4, feat5


# ✅ Multi-Scale Feature Aggregation (MSFA)
class MultiScaleFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureAggregation, self).__init__()
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


# ✅ CoordAttention (Coordinate Attention)
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CoordAttention, self).__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // reduction)
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        h_attn = self.avg_pool_h(x).permute(0, 1, 3, 2)
        w_attn = self.avg_pool_w(x)
        shared_feat = torch.cat([h_attn, w_attn], dim=2)
        shared_feat = self.conv1(shared_feat)
        shared_feat = self.bn(shared_feat)
        shared_feat = F.relu(shared_feat)
        split_size = shared_feat.shape[2] // 2
        h_attn, w_attn = torch.split(shared_feat, [split_size, split_size], dim=2)
        h_attn = self.conv_h(h_attn.permute(0, 1, 3, 2))
        w_attn = self.conv_w(w_attn)
        attn = torch.sigmoid(h_attn + w_attn)
        return x * attn


# ✅ Context-aware Pyramid Feature Extraction (CPFE)
class CPFE(nn.Module):
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
        return self.fuse_conv(fused)


# ✅ Hard Negative Cross Attention (HNCA)
class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(HardNegativeCrossAttention, self).__init__()
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
class EnhancedDistortionDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedDistortionDetectionModel, self).__init__()
        self.vgg = VGG16FeatureExtractor()
        self.msfa = MultiScaleFeatureAggregation(64, 64)
        self.coord_attn = CoordAttention(64)
        self.cpfe = CPFE(512, 64)
        self.hnca = HardNegativeCrossAttention(64)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        low_feat = self.msfa(feat1)
        low_feat = self.coord_attn(low_feat) * low_feat
        high_feat = self.hnca(self.cpfe(feat5))
        high_feat = self.upsample(high_feat)
        fused_feat = torch.cat([low_feat, high_feat], dim=1)
        output = self.final_conv(fused_feat)
        return output.view(output.shape[0], -1).mean(dim=1)


# ✅ 손실 함수
def distortion_loss(pred, gt):
    mse_loss = nn.MSELoss()(pred, gt)
    perceptual_loss = torch.mean(torch.abs(pred - gt))
    return mse_loss + 0.1 * perceptual_loss


# ✅ 테스트 코드
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)  # ✅ 입력 크기 224x224로 설정
    dummy_gt = torch.randn(2)  # ✅ MOS 점수 (batch_size,) 형태로 생성
    model = EnhancedDistortionDetectionModel()
    output = model(dummy_input)
    loss = distortion_loss(output, dummy_gt)

    print("Model Output Shape:", output.shape)  # ✅ (batch_size,)가 출력되어야 함
    print("Loss:", loss.item())  # ✅ 손실 값 출력
 """
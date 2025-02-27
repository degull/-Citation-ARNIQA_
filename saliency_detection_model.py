import torch
import torch.nn as nn
import torchvision.models as models

# ✅ VGG-16 기반 Feature Extractor
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

# ✅ CPFE (Context-aware Pyramid Feature Extraction)
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

# ✅ Channel Attention (CA)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * (avg_out + max_out)

# ✅ Spatial Attention (SA)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(attn))

# ✅ Saliency Detection Model (수정된 버전)
class SaliencyDetectionModel(nn.Module):
    def __init__(self):
        super(SaliencyDetectionModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        
        # CPFE 적용 (Conv3-3, Conv4-3, Conv5-3을 모두 사용)
        self.cpfe3 = CPFE(256, 64)
        self.cpfe4 = CPFE(512, 64)
        self.cpfe5 = CPFE(512, 64)

        # Channel Attention & Spatial Attention 추가
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        _, _, feat3, feat4, feat5 = self.vgg(x)
        
        # CPFE 적용
        feat3 = self.cpfe3(feat3)
        feat4 = self.cpfe4(feat4)
        feat5 = self.cpfe5(feat5)

        # Feature Fusion
        high_feat = feat3 + feat4 + feat5

        # Attention 적용
        high_feat = self.ca(high_feat)
        high_feat = self.sa(high_feat)

        # Upsampling
        high_feat = self.upsample(high_feat)
        output = self.final_conv(high_feat)

        return output.view(output.shape[0], -1).mean(dim=1)

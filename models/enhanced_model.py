import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg16_extractor import VGG16FeatureExtractor
from models.cpfe import CPFE
from models.coord_attention import CoordAttention
from models.hnca import HardNegativeCrossAttention

class EnhancedDistortionDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedDistortionDetectionModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.coord_attn = CoordAttention(in_channels=192)
        self.cpfe = CPFE(in_channels=512, out_channels=64)
        self.hnca = HardNegativeCrossAttention(in_channels=64)
        self.reduce_conv = nn.Conv2d(384, 64, kernel_size=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # ✅ in_channels 수정: 256 = 192 (low) + 64 (high)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Linear(1, 1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)

        feat2_up = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        low_feat = torch.cat([feat1, feat2_up], dim=1)  # → (B, 192, 224, 224)
        low_feat = self.coord_attn(low_feat)            # → (B, 64, 224, 224)

        high_feat = self.cpfe(feat5)                    # → (B, 64, 64, 64)
        high_feat = self.hnca(high_feat)                # → (B, 384, 64, 64)
        high_feat = self.reduce_conv(high_feat)         # → (B, 64, 64, 64)
        high_feat = self.upsample(high_feat)            # → (B, 64, 224, 224)

        fused = torch.cat([low_feat, high_feat], dim=1)  # → (B, 128 + 128, 224, 224) = (B, 256, 224, 224)
        out = self.final_conv(fused)                     # → (B, 1, 224, 224)

        out = self.gap(out)             # → (B, 1, 1, 1)
        out = out.view(out.size(0), -1) # → (B, 1)
        out = self.regressor(out)       # → (B, 1)
        return out.squeeze(1)           # → (B,)

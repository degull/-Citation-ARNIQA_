import torch
import torch.nn as nn
from saliency_detection_model import VGG16FeatureExtractor, CPFE


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


class DistortionAttentionCPFEModel(nn.Module):
    def __init__(self):
        super(DistortionAttentionCPFEModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.cpfe = CPFE(512, 64)
        self.hnca = HardNegativeCrossAttention(64)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        _, _, _, _, feat5 = self.vgg(x)
        high_feat = self.hnca(self.cpfe(feat5))

        high_feat = self.upsample(high_feat)
        output = self.final_conv(high_feat)

        return output.view(output.shape[0], -1).mean(dim=1)

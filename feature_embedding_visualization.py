import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ✅ 모델 경로 지정
checkpoint_path = "E:/ARNIQA - SE - mix/ARNIQA/experiments/my_experiment/regressors/1e-4/kadid/epoch_27_srocc_0.938.pth"

# ✅ VGG-16을 활용한 Feature Extractor
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.conv1_2 = nn.Sequential(*vgg16[:4])
        self.conv2_2 = nn.Sequential(*vgg16[4:9])
        self.conv3_3 = nn.Sequential(*vgg16[9:16])
        self.conv4_3 = nn.Sequential(*vgg16[16:23])
        self.conv5_3 = nn.Sequential(*vgg16[23:30])

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

# ✅ CoordAttention (Coordinate Attention)
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CoordAttention, self).__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // reduction)
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

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

# ✅ 모델 정의
class EnhancedDistortionDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedDistortionDetectionModel, self).__init__()
        self.vgg = VGG16FeatureExtractor()
        self.coord_attn = CoordAttention(64)
        self.cpfe = CPFE(512, 64)
        self.hnca = HardNegativeCrossAttention(64)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        low_feat = self.coord_attn(feat1) * feat1
        high_feat = self.hnca(self.cpfe(feat5))
        high_feat = self.upsample(high_feat)
        fused_feat = torch.cat([low_feat, high_feat], dim=1)
        output = self.final_conv(fused_feat)
        return output.view(output.shape[0], -1).mean(dim=1)

# ✅ 모델 로드
model = EnhancedDistortionDetectionModel()
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint)
model.eval()

# ✅ 더미 입력 데이터 생성
dummy_input = torch.randn(100, 3, 224, 224)

# ✅ 특징 벡터 추출
def extract_features(model, data):
    with torch.no_grad():
        feat1, feat2, feat3, feat4, feat5 = model.vgg(data)
        high_feat = model.hnca(model.cpfe(feat5))
        return high_feat.reshape(high_feat.size(0), -1).cpu().numpy()

features = extract_features(model, dummy_input)

# ✅ PCA 및 t-SNE 변환 및 시각화
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], alpha=0.7)
plt.title("PCA Feature Distribution")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_tsne = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], alpha=0.7)
plt.title("t-SNE Feature Distribution")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

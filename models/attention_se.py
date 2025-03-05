# 원본
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



# HAN-IQA+ (CoordAttention + Transformer + HNCA)
# 사용 안함
""" 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ✅ VGG-16 Feature Extractor
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
        feat4 = self.conv4_3(feat3)  # [B, 512, 28, 28]
        feat5 = self.conv5_3(feat4)  # [B, 512, 14, 14]

        return feat1, feat2, feat3, feat4, feat5


# ✅ CoordAttention
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CoordAttention, self).__init__()

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))  
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))  

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // reduction)

        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)

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
        attn = x * attn  

        return self.conv_out(attn)


# ✅ Transformer-Based Self-Attention
class TransformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(TransformerAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        x, _ = self.attention(x, x, x)  # Self-Attention
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


# ✅ Hard Negative Cross Attention (HNCA)
class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(HardNegativeCrossAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.output_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)

        attn_scores = torch.softmax(Q @ K.transpose(-2, -1) / (self.query.in_features ** 0.5), dim=-1)
        attn_output = attn_scores @ V
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)

        return self.bn(self.output_proj(attn_output + x))  


# ✅ Multi-Scale Feature Fusion
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFusion, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        return self.conv1(x) + self.conv3(x) + self.conv5(x)


# ✅ 최종 모델 (NR-IQA)
class HAN_IQA_PLUS(nn.Module):
    def __init__(self):
        super(HAN_IQA_PLUS, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.coord_attn = CoordAttention(512)  
        self.trans_attn = TransformerAttention(512)
        self.hnca = HardNegativeCrossAttention(512)
        self.multi_scale = MultiScaleFusion(512, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_A):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(img_A)

        coord_attention = self.coord_attn(feat4)  # [B, 512, 28, 28]
        transformer_output = self.trans_attn(feat5)  # [B, 512, 14, 14]
        hnca_output = self.hnca(feat5)  # [B, 512, 14, 14]

        # **크기 맞추기 (28x28)**
        transformer_output = F.interpolate(transformer_output, size=coord_attention.shape[2:], mode='bilinear', align_corners=True)
        hnca_output = F.interpolate(hnca_output, size=coord_attention.shape[2:], mode='bilinear', align_corners=True)

        fused_feat = self.multi_scale(coord_attention + transformer_output + hnca_output)
        output = self.final_conv(fused_feat)

        return self.sigmoid(self.relu(output)).view(output.shape[0], -1).mean(dim=1)


# ✅ 실행 테스트
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    model = HAN_IQA_PLUS()
    output = model(dummy_input)

    print("🚀 Model Output Shape:", output.shape)
 """


# 방식 1 + coord  + Hard Negative Sample Selection Strategy
# 사용 X
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


# 방식 2 + coord  + Hard Negative Sample Selection Strategy
# 사용 X
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


# ✅ Hard Negative Sample Selection Strategy
class HardNegativeSampleSelector:
    def __init__(self):
        pass

    def select_hard_negatives(self, features, perceptual_attrs):
        batch_size = features.shape[0]
        feature_vectors = features.view(batch_size, -1)
        attr_vectors = perceptual_attrs.view(batch_size, -1)

        hard_negatives = []

        for i in range(batch_size):
            feature_similarity = F.cosine_similarity(feature_vectors, feature_vectors[i].unsqueeze(0), dim=1)
            attr_similarity = F.cosine_similarity(attr_vectors, attr_vectors[i].unsqueeze(0), dim=1)

            # ✅ 자기 자신 제외
            feature_similarity[i] = -1
            attr_similarity[i] = -1

            # ✅ Feature + Perceptual Attribute 기반으로 Hard Negative 선택
            combined_similarity = feature_similarity + attr_similarity
            closest_index = torch.argmax(combined_similarity).item()

            if closest_index < batch_size:
                hard_negatives.append(features[closest_index])
            else:
                hard_negatives.append(features[i])  # 자기 자신을 대체

        return torch.stack(hard_negatives)


# ✅ Hard Negative Cross Attention (HNCA)
class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(HardNegativeCrossAttention, self).__init__()
        self.in_dim = in_dim
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.output_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    def forward(self, x, hard_negatives):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        hn_flat = hard_negatives.view(b, c, -1).permute(0, 2, 1)

        Q = self.query(x_flat)
        K = self.key(hn_flat)
        V = self.value(hn_flat)

        attn_scores = torch.softmax(Q @ K.transpose(-2, -1) / (self.in_dim ** 0.5), dim=-1)
        attn_output = attn_scores @ V

        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
        return self.output_proj(attn_output + x)


# ✅ 최종 모델
class EnhancedDistortionDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedDistortionDetectionModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.coord_attn = CoordAttention(64)
        self.cpfe = CPFE(512, 64)
        self.hnca = HardNegativeCrossAttention(64)
        self.hard_negative_selector = HardNegativeSampleSelector()

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x, perceptual_attrs):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        low_feat = self.coord_attn(feat1) * feat1
        high_feat = self.cpfe(feat5)

        hard_negatives = self.hard_negative_selector.select_hard_negatives(high_feat, perceptual_attrs)
        high_feat = self.hnca(high_feat, hard_negatives)

        high_feat = self.upsample(high_feat)
        fused_feat = torch.cat([low_feat, high_feat], dim=1)
        output = self.final_conv(fused_feat)

        return output.view(output.shape[0], -1).mean(dim=1)


# ✅ 실행 코드
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_perceptual_attrs = torch.randn(2, 4)

    model = EnhancedDistortionDetectionModel()
    output = model(dummy_input, dummy_perceptual_attrs)

    print("Model Output Shape:", output.shape)
 """

# cood + Hard Negative Sample Selection Strategy (HNSS)
# 사용 X

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
    def __init__(self, in_dim, num_negatives=5):  # ✅ `num_negatives` 추가
        super(HardNegativeCrossAttention, self).__init__()
        self.in_dim = in_dim
        self.num_negatives = num_negatives  # ✅ 저장

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.output_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.channel_fix = nn.Conv2d(3, in_dim, kernel_size=1)  # ✅ 채널 변환 추가

    def forward(self, x, neg_x):
        b, c, h, w = x.shape
        neg_b, neg_c, neg_h, neg_w = neg_x.shape

        # ✅ Batch 크기 맞추기 (x와 동일한 Batch 크기로 조정)
        if neg_b != b:
            neg_x = neg_x[:b]  # ✅ Batch 크기를 맞춰주기

        # ✅ 크기 변환 (neg_x: 3 → 64)
        neg_x = self.channel_fix(neg_x)

        # ✅ 크기 맞추기 (x와 동일한 H, W 크기로 변환)
        neg_x = F.interpolate(neg_x, size=(h, w), mode="bilinear", align_corners=False)

        # ✅ view 연산 시 크기 일치하도록 조정
        neg_x_flat = neg_x.view(b, c, -1).permute(0, 2, 1)  # 🔧 수정

        Q = self.query(x.view(b, c, -1).permute(0, 2, 1))
        K_neg = self.key(neg_x_flat)
        V_neg = self.value(neg_x_flat)

        attn_scores = torch.softmax(Q @ K_neg.transpose(-2, -1) / (self.in_dim ** 0.5), dim=-1)
        attn_output = attn_scores @ V_neg
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)

        return self.output_proj(attn_output + x)


# ✅ 최종 모델
class EnhancedDistortionDetectionModel(nn.Module):
    def __init__(self, num_negatives=5):
        super(EnhancedDistortionDetectionModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.coord_attn = CoordAttention(64)
        self.cpfe = CPFE(512, 64)
        self.hnca = HardNegativeCrossAttention(64, num_negatives)  # ✅ 이제 정상 작동

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x, hard_negatives):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        low_feat = self.coord_attn(feat1) * feat1
        high_feat = self.hnca(self.cpfe(feat5), hard_negatives)

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


# CBAM + Coord
""" import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ✅ VGG-16 Feature Extractor
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
        feat4 = self.conv4_3(feat3)  # [B, 512, 28, 28]
        feat5 = self.conv5_3(feat4)  # [B, 512, 14, 14]

        return feat1, feat2, feat3, feat4, feat5


# ✅ CBAM (Channel & Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # 🔥 채널 어텐션
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        # 🔥 공간 어텐션
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attn = torch.sigmoid(avg_out + max_out)
        x = x * channel_attn  # ✅ 채널 어텐션 적용

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attn  # ✅ 공간 어텐션 적용

        return x


# ✅ CoordAttention
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CoordAttention, self).__init__()

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))  
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))  

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // reduction)

        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)

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
        attn = x * attn  

        return self.conv_out(attn)


# ✅ Hard Negative Cross Attention (HNCA)
class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(HardNegativeCrossAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.output_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)

        attn_scores = torch.softmax(Q @ K.transpose(-2, -1) / (self.query.in_features ** 0.5), dim=-1)
        attn_output = attn_scores @ V
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)

        return self.bn(self.output_proj(attn_output + x)) 


# ✅ 최종 모델 (img_A만 사용)
class EnhancedDistortionDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedDistortionDetectionModel, self).__init__()

        self.vgg = VGG16FeatureExtractor()
        self.coord_attn = CoordAttention(512)  
        self.cbam = CBAM(512)  
        self.hnca = HardNegativeCrossAttention(512)

        self.reduce_dim = nn.Conv2d(512, 64, kernel_size=1)  
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)  # ✅ 128 → 1로 수정

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_A):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(img_A)

        coord_attention = self.coord_attn(feat4)  
        cbam_attention = self.cbam(feat5)
        cbam_attention = F.interpolate(cbam_attention, size=coord_attention.shape[2:], mode='bilinear', align_corners=True)

        hnca_output = self.hnca(feat5)
        high_feat = F.interpolate(self.reduce_dim(hnca_output), size=coord_attention.shape[2:], mode='bilinear', align_corners=True)

        fused_feat = torch.cat([self.reduce_dim(coord_attention * cbam_attention), high_feat], dim=1)
        output = self.final_conv(fused_feat)

        return self.sigmoid(self.relu(output)).view(output.shape[0], -1).mean(dim=1)


# ✅ 실행 테스트
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    model = EnhancedDistortionDetectionModel()
    output = model(dummy_input)

    print("🚀 Model Output Shape:", output.shape)
 """


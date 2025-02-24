import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms

# ✅ MOS 값을 CSV에서 가져오는 함수
def get_mos_value(image_name, csv_path):
    df = pd.read_csv(csv_path)
    mos_value = df[df["image_name"] == image_name]["MOS"].values
    if len(mos_value) > 0:
        return mos_value[0] / 100.0  # MOS 값을 0~1로 정규화
    else:
        return None

# ✅ VGG-16 특징 추출기
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


# ✅ Spatial Attention (SA) for Distortion Detection (1)
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)

    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn
    



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

        return feat1, feat2, feat5, output.view(output.shape[0], -1).mean(dim=1)  # ✅ 4개 반환



def distortion_loss(pred, gt):
    mse_loss = nn.MSELoss()(pred, gt)  # ✅ MOS 값과 직접 비교
    perceptual_loss = torch.mean(torch.abs(pred - gt))  # ✅ L1 Loss 추가
    return mse_loss + 0.1 * perceptual_loss


# ✅ 이미지 로드 및 전처리
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)
    return image, input_tensor

# ✅ Feature Map 시각화 (MOS 값을 반영 ✅)
def visualize_feature_maps(original_image, feat1, feat2, feat5, final_pred, mos_value):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # (a) 원본 이미지
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # ✅ Ground Truth (MOS-based) (MOS 값이 존재하면 표시)
    if mos_value is not None:
        gt_mask = np.full((224, 224), mos_value, dtype=np.float32)  # MOS 값으로 채움
    else:
        gt_mask = np.zeros((224, 224), dtype=np.float32)  # MOS 값이 없으면 검은 화면

    axes[0, 1].imshow(gt_mask, cmap="gray")
    axes[0, 1].set_title("Ground Truth (MOS-based)")
    axes[0, 1].axis("off")

    # (c) Low-level Features (Conv1)
    axes[0, 2].imshow(torch.mean(feat1.squeeze(), dim=0).detach().cpu().numpy(), cmap="gray")
    axes[0, 2].set_title("Low-level Features (Conv1)")
    axes[0, 2].axis("off")

    # (d) Low-level + Spatial Attention 적용
    axes[0, 3].imshow(torch.mean(feat2.squeeze(), dim=0).detach().cpu().numpy(), cmap="gray")
    axes[0, 3].set_title("Low-level + SA")
    axes[0, 3].axis("off")

    # (e) High-level Features (Conv5)
    axes[1, 0].imshow(torch.mean(feat5.squeeze(), dim=0).detach().cpu().numpy(), cmap="gray")
    axes[1, 0].set_title("High-level Features (Conv5)")
    axes[1, 0].axis("off")

    # (f) High-level + Channel-wise Attention 적용
    axes[1, 1].imshow(torch.mean(feat5.squeeze(), dim=0).detach().cpu().numpy(), cmap="gray")
    axes[1, 1].set_title("High-level + CA")
    axes[1, 1].axis("off")

    # ✅ Model Output Scaling (0~255 변환)
    model_output = np.uint8(gt_mask * 255)
    axes[1, 2].imshow(model_output, cmap="gray")
    axes[1, 2].set_title("Model Output")
    axes[1, 2].axis("off")

    # ✅ Boundary Map Enhancement
    laplacian_input = model_output.astype(np.uint8)
    boundary = cv2.Laplacian(laplacian_input, cv2.CV_64F)
    boundary = np.uint8(np.abs(boundary))

    axes[1, 3].imshow(boundary, cmap="gray")
    axes[1, 3].set_title("Boundary Map")
    axes[1, 3].axis("off")

    plt.show()

# ✅ 모델 실행 및 Feature Map 추출
if __name__ == "__main__":
    model = DistortionDetectionModel()

    # 이미지 및 CSV 경로 설정
    image_name = "5076506.jpg"
    image_path = f"E:/ARNIQA - SE - mix/ARNIQA/dataset/KONIQ10K/1024x768/{image_name}"
    csv_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KONIQ10K/meta_info_KonIQ10kDataset.csv"

    # MOS 값 불러오기
    mos_value = get_mos_value(image_name, csv_path)

    # ✅ MOS 값 확인 (CSV에서 제대로 가져왔는지 체크)
    print(f"✅ 이미지 이름: {image_name}")
    print(f"✅ 이미지 경로: {image_path}")
    print(f"✅ MOS 값: {mos_value}")

    # 이미지 로드
    original_image, input_tensor = load_image(image_path)

    # ✅ 이미지 텐서 크기 확인
    print(f"✅ 입력 이미지 텐서 크기: {input_tensor.shape}")

    # 모델 실행
    outputs = model(input_tensor)

    # ✅ 모델 출력 개수 확인
    print(f"✅ 모델 출력 타입: {type(outputs)}")
    print(f"✅ 모델 출력 개수: {len(outputs) if isinstance(outputs, tuple) else '1개'}")

    # 모델이 4개의 값을 반환하는 경우
    if isinstance(outputs, tuple) and len(outputs) == 4:
        feat1, feat2, feat5, final_pred = outputs
    else:
        raise ValueError("❌ 모델이 예상한 4개의 값을 반환하지 않음. 모델 구조 확인 필요!")

    # ✅ 모델 예측값 정규화 (0~1 스케일 조정)
    final_pred = torch.sigmoid(final_pred)
    print(f"✅ 모델 예측값 (final_pred): {final_pred}")

    # 시각화
    visualize_feature_maps(original_image, feat1, feat2, feat5, final_pred, mos_value)

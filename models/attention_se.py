import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import seaborn as sns
import scipy.stats


class FeatureLevelAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, num_heads=8, dropout=0.3):
        super(FeatureLevelAttention, self).__init__()

        self.num_heads = num_heads
        self.out_channels = out_channels

        # ✅ 입력 채널 변환 (3 → 32)
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.head_dim = max(out_channels // num_heads, 1)
        self.scale = self.head_dim ** -0.5  # Scale Dot-Product Attention

        # Query, Key, Value 생성
        self.qkv_conv = nn.Conv2d(out_channels, out_channels * 3, kernel_size=1, bias=False)
        self.output_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Feature Update (Residual Learning + Normalization)
        self.feature_update_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # ✅ 가중치 초기화 적용
        self._initialize_weights()

    def _initialize_weights(self):
        """ Xavier Uniform 초기화 적용 """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        # ✅ 입력 채널 변환 (3채널 → 32채널)
        x = self.input_conv(x)

        # Query, Key, Value 생성
        qkv = self.qkv_conv(x).reshape(b, self.num_heads, 3, self.head_dim, h * w)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        # ✅ Scaled Dot-Product Attention 대신 Sigmoid 적용 (분포 개선)
        attention = torch.sigmoid(torch.matmul(q, k.transpose(-2, -1)) * self.scale)
        A_final = torch.matmul(attention, v)
        A_final = A_final.reshape(b, self.out_channels, h, w)

        # 최종 Projection 적용
        A_final = self.output_proj(A_final)
        A_final = self.feature_update_conv(A_final)
        A_final = self.batch_norm(A_final)
        A_final = self.activation(A_final)
        A_final = self.dropout(A_final)

        # Residual Connection 적용
        out = A_final + x

        return out, attention


def visualize_attention(img_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        _, attention_map = model(img_tensor)

    # ✅ Attention Map 크기 확인 및 조정
    print(f"Original Attention Map Shape: {attention_map.shape}")  # (1, 8, 4, 4)

    # 🔥 (1, 8, 4, 4) → (1, 4, 4)로 평균화 후 Interpolation 적용
    attention_map = attention_map.mean(dim=1, keepdim=True)  # (1, 1, 4, 4)
    attention_map = torch.nn.functional.interpolate(
        attention_map, size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)  # (1, 224, 224)

    # Attention Map 후처리
    attention_map = attention_map.squeeze().cpu().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # ✅ Attention Map의 기본 통계 출력
    print(f"Attention Map Mean: {attention_map.mean():.4f}")
    print(f"Attention Map Std Dev: {attention_map.std():.4f}")
    print(f"Attention Map Min: {attention_map.min():.4f}")
    print(f"Attention Map Max: {attention_map.max():.4f}")

    # ✅ Attention Map 히스토그램 시각화
    plt.figure(figsize=(6, 4))
    sns.histplot(attention_map.flatten(), bins=30, kde=True)
    plt.title("Distribution of Attention Weights")
    plt.xlabel("Attention Score")
    plt.ylabel("Frequency")
    plt.show()

    # ✅ 원본 이미지 변환 (Grayscale인지 확인 후 변환)
    img_cv = cv2.imread(img_path)

    # 🔥 img_cv가 grayscale(1채널)인 경우 3채널로 변환
    if len(img_cv.shape) == 2 or img_cv.shape[-1] == 1:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

    img_cv = cv2.resize(img_cv, (224, 224))

    # ✅ Attention Map을 3채널로 변환 후 ColorMap 적용
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)

    # 🔥 크기와 채널 수가 같아야 OpenCV addWeighted 사용 가능
    if img_cv.shape != heatmap.shape:
        heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))

    # ✅ 가중치 합성 (두 이미지 크기/채널 일치 여부 확인 후 적용)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    # ✅ Attention Map과 원본 이미지 픽셀 값의 상관 관계 계산
    image_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) / 255.0  # Normalize to 0-1
    image_gray_resized = cv2.resize(image_gray, (attention_map.shape[1], attention_map.shape[0]))

    corr, p_value = scipy.stats.pearsonr(image_gray_resized.flatten(), attention_map.flatten())
    print(f"Correlation between Image Intensity and Attention: {corr:.4f}")

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(attention_map, cmap="jet")
    plt.title("Attention Map")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlayed Image")

    plt.show()


# 실행
if __name__ == "__main__":
    num_heads = 8
    feature_attention = FeatureLevelAttention(in_channels=3, out_channels=32, num_heads=num_heads)

    # 테스트 이미지 경로 설정
    test_img_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/dst_imgs/AWGN/family.AWGN.5.png"

    # Attention Map 시각화 실행
    visualize_attention(test_img_path, feature_attention)

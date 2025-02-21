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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def __call__(self, x):
        x.requires_grad = True
        feature_map, attention = self.model(x)
        feature_map.register_hook(self.save_gradient)
        
        # 가장 높은 Attention Score를 가진 위치를 Gradient로 계산
        score = torch.max(attention)
        score.backward()
        
        grad = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu(grad * feature_map).mean(dim=1).cpu().detach().numpy()
        
        return cam


class FeatureLevelAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, num_heads=8, dropout=0.3):
        super(FeatureLevelAttention, self).__init__()

        self.num_heads = num_heads
        self.out_channels = out_channels

        # ✅ 입력 채널 변환 (3 → 32, 해상도 유지)
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.head_dim = max(out_channels // num_heads, 1)
        self.scale = self.head_dim ** -0.5  

        # Query, Key, Value 생성 (해상도 유지)
        self.qkv_conv = nn.Conv2d(out_channels, out_channels * 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.output_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Feature Update (Residual Learning + Normalization)
        self.feature_update_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.size()

        # ✅ 입력 채널 변환 (해상도 유지)
        x = self.input_conv(x)

        # ✅ Query, Key, Value 생성 (해상도 유지)
        qkv = self.qkv_conv(x).reshape(b, self.num_heads, 3, self.head_dim, h, w)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)  

        # ✅ Gumbel Softmax 적용 (Feature Contrast 강화)
        attention = F.gumbel_softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, tau=1, hard=True, dim=-1)
        A_final = torch.matmul(attention, v)  
        A_final = A_final.reshape(b, self.out_channels, h, w)  

        # ✅ Attention Map 해상도 보존을 위해 Interpolation 적용
        A_final = F.interpolate(A_final, size=(224, 224), mode="bilinear", align_corners=False)

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

    print(f"Original Attention Map Shape: {attention_map.shape}")  # 확인용 출력

    # ✅ Attention Map 차원 조정 (num_heads 차원 제거)
    attention_map = attention_map.mean(dim=1, keepdim=True)  # (1, 8, 4, 224, 224) → (1, 1, 4, 224, 224)

    # ✅ Spatial Dimension 조정: (1, 1, 4, 224, 224) → (1, 1, 224, 224)
    attention_map = attention_map.mean(dim=2, keepdim=True)  # (1, 1, 4, 224, 224) → (1, 1, 1, 224, 224)
    attention_map = attention_map.squeeze(2)  # (1, 1, 1, 224, 224) → (1, 1, 224, 224)

    # ✅ Normalize Attention Map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    attention_map = attention_map.squeeze().cpu().numpy()  # (224, 224)로 변환

    # ✅ 원본 이미지 로드
    img_cv = cv2.imread(img_path)
    if len(img_cv.shape) == 2 or img_cv.shape[-1] == 1:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

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

    # 여러 이미지를 테스트하여 시각적 변화를 분석
    distortions = ["AWGN", "Blur", "JPEG"]
    for distortion in distortions:
        test_img_path = f"E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/dst_imgs/{distortion}/family.{distortion}.5.png"
        visualize_attention(test_img_path, feature_attention)

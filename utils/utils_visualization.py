import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def apply_gradcam_heatmap(feature_map):
    """
    Grad-CAM 스타일의 Heatmap 생성
    :param feature_map: 특징 맵 (Tensor, shape: [C, H, W])
    :return: Heatmap (numpy array)
    """
    heatmap = feature_map.mean(dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # 음수 제거
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # Normalize
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # 컬러맵 적용
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR → RGB 변환
    return heatmap

def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """
    원본 이미지에 Heatmap을 오버레이
    :param original_image: 원본 이미지 (PIL Image or numpy array)
    :param heatmap: Heatmap (numpy array)
    :param alpha: 오버레이 강도 (0 ~ 1)
    :return: 오버레이된 이미지 (numpy array)
    """
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    if original_image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    return overlay

def visualize_feature_maps(activation_maps, original_image):
    """
    각 레이어별 Grad-CAM 스타일의 특징 맵을 시각화하여 원본 이미지에 오버레이
    """
    plt.figure(figsize=(15, 8))
    num_maps = len(activation_maps)

    for i, (layer_name, feature_map) in enumerate(activation_maps.items()):
        # 🔥 Feature Map 크기 확인
        print(f"[Debug] {layer_name} Feature Map Shape: {feature_map.shape}")

        # 🔥 Feature Map 크기가 맞지 않으면 수정
        if feature_map.dim() == 4:  # [B, C, H, W]라면 첫 번째 배치 선택
            feature_map = feature_map[0]
        elif feature_map.dim() == 3:  # [C, H, W]라면 그대로 사용
            pass
        else:
            print(f"[Error] Unexpected feature map shape: {feature_map.shape}")
            continue  # 잘못된 feature map은 스킵

        heatmap = apply_gradcam_heatmap(feature_map)
        overlay_img = overlay_heatmap(original_image, heatmap)
        plt.subplot(2, num_maps // 2, i + 1)
        plt.imshow(overlay_img)
        plt.title(layer_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def apply_heatmap(original_image, feature_map, alpha=0.5):
    """
    원본 이미지에 feature_map을 오버레이하여 Heatmap을 생성
    :param original_image: 원본 이미지 (PIL Image or numpy array)
    :param feature_map: 특징 맵 (Tensor, shape: [C, H, W])
    :param alpha: Heatmap 강도 조절
    :return: Heatmap이 적용된 이미지
    """
    # 특징 맵을 평균내어 [H, W] 크기로 변환
    heatmap = feature_map.mean(dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # 음수 제거
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # Normalize
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # 컬러맵 적용
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR → RGB 변환

    # 원본 이미지 크기 조정
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    if original_image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # 이미지 오버레이 (alpha 값으로 조절)
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    return overlay

def visualize_feature_maps(activation_maps, original_image):
    """
    각 레이어별 특징맵을 시각화하여 원본 이미지에 오버레이
    """
    plt.figure(figsize=(15, 8))
    num_maps = len(activation_maps)
    
    for i, (layer_name, feature_map) in enumerate(activation_maps.items()):
        heatmap = apply_heatmap(original_image, feature_map[0])  # 첫 번째 채널 Feature Map 사용
        plt.subplot(2, num_maps // 2, i + 1)
        plt.imshow(heatmap)
        plt.title(layer_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

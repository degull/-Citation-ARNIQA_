import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

# 저장 경로 설정
SAVE_DIR = r"E:\ARNIQA - SE - mix\ARNIQA\feature_maps"
os.makedirs(SAVE_DIR, exist_ok=True)  # 폴더 없으면 생성

def apply_gradcam_heatmap(feature_map):
    """
    Grad-CAM 스타일의 Heatmap 생성
    :param feature_map: 특징 맵 (Tensor, shape: [C, H, W])
    :return: Heatmap (numpy array)
    """
    heatmap = feature_map.mean(dim=0).detach().cpu().numpy()  # 채널 평균
    heatmap = np.maximum(heatmap, 0)  # 음수 제거
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # 정규화
    heatmap = np.uint8(255 * heatmap)  # 0-255로 변환
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 컬러맵 적용
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
        original_image = np.array(original_image)  # PIL → numpy 변환
    if original_image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))  # 크기 맞추기
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)  # 합성
    return overlay

# 히트맵
def visualize_feature_maps(activation_maps, original_image):

    plt.figure(figsize=(12, 6))
    layer_names = list(activation_maps.keys())

    for i, layer_name in enumerate(layer_names):
        if "Layer0" in layer_name or "Layer1" in layer_name or "Layer2" in layer_name or "Layer3" in layer_name or "Layer4" in layer_name:
            feature_map = activation_maps[layer_name]

            if isinstance(feature_map, torch.Tensor):
                heatmap = apply_gradcam_heatmap(feature_map.squeeze(0))  # Grad-CAM 스타일 히트맵 생성
                overlay = overlay_heatmap(original_image, heatmap, alpha=0.5)  # 원본에 오버레이

                # 이미지 저장
                save_path = os.path.join(SAVE_DIR, f"{layer_name}.png")
                Image.fromarray(overlay).save(save_path)
                print(f"[✔] {layer_name} 저장 완료: {save_path}")

                # 시각화
                plt.subplot(2, 4, i + 1)
                plt.imshow(overlay)
                plt.title(layer_name)
                plt.axis("off")

    plt.tight_layout()
    plt.show()

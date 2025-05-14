import torch
import numpy as np
import cv2
from torchvision import transforms
from models.attention_se import MultiTaskIQA


import cv2
import numpy as np
import torch

# ✅ Saliency Map 생성 함수
def generate_saliency_map(model, input_tensor, device):
    input_tensor.requires_grad_()  # 입력 텐서에 gradient 활성화

    model.zero_grad()
    quality_score, _ = model(input_tensor)  # 품질 점수만 예측
    score = quality_score.sum()  # 스칼라 값으로 변환

    score.backward()  # 역전파 실행

    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)  # 채널별 max 값 사용
    saliency = saliency.squeeze().cpu().numpy()  # (H, W) 형태로 변환

    # ✅ Contrast Stretching 적용 (Saliency Map 강화)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(255 * saliency)  # 0~255 정규화 후 uint8 변환

    return saliency

# ✅ Saliency Map을 이미지 위에 시각화
def apply_saliency_overlay(image, saliency):
    # ✅ heatmap 크기를 원본 이미지 크기에 맞춤
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_HOT)  # COLORMAP_JET 대신 COLORMAP_HOT 사용
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # 원본 이미지 크기로 조정

    # ✅ 이미지와 Saliency Map 합성
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    return overlay

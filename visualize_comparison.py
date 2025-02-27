import sys
import os
import torch
import cv2
import numpy as np
from saliency_detection_model import SaliencyDetectionModel
from distortion_attention_model import DistortionAttentionCPFEModel
from grad_cam import GradCAM, preprocess_image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

image_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KONIQ10K/1024x768/5076506.jpg"

print("🔹 Loading Saliency Detection CPFE Model...")
saliency_model = SaliencyDetectionModel()
saliency_model.eval()
print("✅ Saliency Detection CPFE Model Loaded.")

print("🔹 Loading Distortion Attention CPFE Model...")
distortion_model = DistortionAttentionCPFEModel()
distortion_model.eval()
print("✅ Distortion Attention CPFE Model Loaded.")

# ✅ Grad-CAM 적용할 레이어 설정
target_layer_saliency = saliency_model.cpfe4.fuse_conv  
target_layer_distortion = distortion_model.cpfe.fuse_conv  
  

print(f"🔹 Selected Target Layer (Saliency): {target_layer_saliency}")
print(f"🔹 Selected Target Layer (Distortion): {target_layer_distortion}")

# ✅ Feature Map 크기 확인 (디버깅)
dummy_input = torch.randn(1, 3, 224, 224)
feat1, feat2, feat3, feat4, feat5 = saliency_model.vgg(dummy_input)
feat1_d, feat2_d, feat3_d, feat4_d, feat5_d = distortion_model.vgg(dummy_input)

print(f"🔹 Feature Map Shapes (Saliency Model): {feat3.shape}, {feat4.shape}, {feat5.shape}")
print(f"🔹 Feature Map Shapes (Distortion Model): {feat3_d.shape}, {feat4_d.shape}, {feat5_d.shape}")

# ✅ 입력 이미지 전처리
original_image, input_tensor = preprocess_image(image_path)

# ✅ Grad-CAM 적용할 레이어 설정 (마지막 CPFE의 Conv2d 사용)
target_layer_saliency = saliency_model.cpfe4.fuse_conv  
target_layer_distortion = distortion_model.cpfe.fuse_conv  


# ✅ Feature Map 크기 확인 (디버깅)
print("🔹 Checking Feature Map Sizes Before Grad-CAM Execution...")
dummy_input = torch.randn(1, 3, 224, 224)
feat1, feat2, feat3, feat4, feat5 = saliency_model.vgg(dummy_input)
feat1_d, feat2_d, feat3_d, feat4_d, feat5_d = distortion_model.vgg(dummy_input)

print(f"🔹 Feature Map Shapes (Saliency Model): {feat3.shape}, {feat4.shape}, {feat5.shape}")
print(f"🔹 Feature Map Shapes (Distortion Model): {feat3_d.shape}, {feat4_d.shape}, {feat5_d.shape}")

# ✅ Grad-CAM 실행
try:
    print("🔹 Visualizing Saliency Detection CPFE...")
    cam_saliency = GradCAM(saliency_model, target_layer_saliency)
    heatmap_saliency = cam_saliency.generate_heatmap(input_tensor)

    overlay_saliency = cam_saliency.apply_heatmap(np.array(original_image), heatmap_saliency)
    cv2.imshow("Saliency Detection CPFE", overlay_saliency)
    cv2.imwrite("saliency_cpfe_heatmap.png", overlay_saliency)
    print("✅ Saliency Detection CPFE Visualization Complete.")

    print("🔹 Visualizing Distortion Attention CPFE...")
    cam_distortion = GradCAM(distortion_model, target_layer_distortion)
    heatmap_distortion = cam_distortion.generate_heatmap(input_tensor)

    overlay_distortion = cam_distortion.apply_heatmap(np.array(original_image), heatmap_distortion)
    cv2.imshow("Distortion Attention CPFE", overlay_distortion)
    cv2.imwrite("distortion_cpfe_heatmap.png", overlay_distortion)
    print("✅ Distortion Attention CPFE Visualization Complete.")

except Exception as e:
    print(f"❌ Grad-CAM Error: {e}")

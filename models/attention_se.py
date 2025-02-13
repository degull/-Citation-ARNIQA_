
import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import ImageEnhance, ImageFilter, Image
import matplotlib.pyplot as plt
from torchvision import transforms



distortion_map = {
    # ✅ JPEG 압축 관련 (Sobel)
    "jpeg": 0,  
    "jpeg2000": 1,    
    "jpeg_artifacts": 2,
    "jpeg_transmission_errors": 3,
    "jpeg2000_transmission_errors": 4,

    # ✅ 블러 관련 (Sobel)
    "gaussian_blur": 5,  
    "lens_blur": 6,
    "motion_blur": 7,
    "blur": 8,

    # ✅ 노이즈 관련 (Sobel)
    "white_noise": 9,
    "awgn": 10,   
    "impulse_noise": 11,  
    "multiplicative_noise": 12,
    "additive_gaussian_noise": 13,
    "additive_noise_in_color_components": 14,
    "spatially_correlated_noise": 15,
    "masked_noise": 16,
    "high_frequency_noise": 17,
    "comfort_noise": 18,
    "lossy_compression_of_noisy_images": 19,
    "low-light_noise": 20,  # 저조도 노이즈
    "color_noise": 21,  # 색상 노이즈

    # ✅ 대비 및 색상 관련 (HSV)
    "contrast_change": 22,
    "contrast": 23,
    "brightness": 24,
    "colorfulness": 25,
    "sharpness": 26,
    "exposure": 27,
    "overexposure": 28,
    "underexposure": 29,
    "color_shift": 30,
    "color_saturation_1": 31,
    "color_saturation_2": 32,
    "change_of_color_saturation": 33,
    "white_balance_error": 34,

    # ✅ 공간적 왜곡 관련 (Sobel)
    "fnoise": 35,
    "fast_fading": 36,
    "color_diffusion": 37,
    "mean_shift": 38,
    "jitter": 39,
    "non_eccentricity_patch": 40,
    "pixelate": 41,
    "spatial_noise": 42,
    "non_eccentricity_pattern_noise": 43,
    "local_block_wise_distortions": 44,

    # ✅ 양자화 및 압축 관련 (Fourier)
    "quantization": 45,
    "color_quantization": 46,
    "image_color_quantization_with_dither": 47,
    "color_block": 48,
    "sparse_sampling_and_reconstruction": 49,

    # ✅ 영상 왜곡 (Fourier)
    "glare": 50,
    "haze": 51,
    "banding_artifacts": 52,
    "vignetting": 53,
    "chromatic_aberration": 54,
    "distortion": 55,
    "high_sharpen": 56,
    "image_denoising": 57
}



class DistortionClassifier(nn.Module):
    def __init__(self, in_channels, num_distortions=58):
        super(DistortionClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, num_distortions)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class AttributeFeatureProcessor(nn.Module):
    def __init__(self, in_channels):
        super(AttributeFeatureProcessor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class TextureBlockProcessor(nn.Module):
    def __init__(self, in_channels):
        super(TextureBlockProcessor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)


class DistortionAttention(nn.Module):
    def __init__(self, in_channels, expected_channels=256):
        super(DistortionAttention, self).__init__()
        self.expected_channels = expected_channels
        query_key_channels = max(1, expected_channels // 8)
        
        # ✅ 입력 채널을 ResNet의 expectation과 맞춤
        self.channel_adjust = nn.Conv2d(in_channels, expected_channels, kernel_size=1)
        
        self.query_conv = nn.Conv2d(expected_channels, query_key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(expected_channels, query_key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(expected_channels, expected_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.distortion_classifier = DistortionClassifier(expected_channels)
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # ✅ 채널 변환 적용
        x = self.channel_adjust(x)  # 512 → 256 변환
        
        # ✅ Attention Map 먼저 계산
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)
        value = self.value_conv(x).view(b, -1, h * w)
        
        scale = query.size(-1) ** -0.5
        attention = self.softmax(torch.bmm(query, key) * scale)
        attention_map = torch.bmm(value, attention.permute(0, 2, 1)).view(b, self.expected_channels, h, w)
        
        # ✅ Distortion Classifier를 통해 왜곡 유형 예측
        distortion_logits = self.distortion_classifier(x)
        distortion_types = torch.argmax(distortion_logits, dim=1)
        
        # ✅ Specialized Filter 적용
        filtered_tensors = []
        for i, dt in enumerate(distortion_types):
            filtered_x = self._apply_filter(x[i].unsqueeze(0), dt)
            if filtered_x.shape[1] == 1:
                filtered_x = filtered_x.expand(-1, self.expected_channels, -1, -1)
            filtered_tensors.append(filtered_x)
        
        filtered_x = torch.cat(filtered_tensors, dim=0)
        
        # ✅ Attention Map과 필터 적용 정보 결합
        feature_map = attention_map + filtered_x  # 🔥 Feature Map 생성 방식 변경
        
        return feature_map, distortion_logits  # ✅ 블록도와 일치하도록 수정



    def _apply_filter(self, x, distortion_type):
        """ 왜곡 유형에 따라 적절한 필터 적용 """

        if distortion_type in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
            11, 12, 13, 14, 15, 16, 17, 18, 19, 
            35, 36, 37, 38, 39, 40, 41, 42, 43, 
            44, 57
        ]:
            print(f"[Debug] Applying Sobel filter for {distortion_type}")
            return self._sobel_filter(x)  # ✅ Sobel 필터 적용

        elif distortion_type in [
            22, 23, 24, 25, 26, 27, 28, 29, 30, 
            31, 32, 33, 34
        ]:
            print(f"[Debug] Applying HSV analysis for {distortion_type}")
            return self._hsv_analysis(x)  # ✅ HSV 필터 적용

        elif distortion_type in [
            45, 46, 47, 48, 49, 50, 51, 52, 53, 
            54, 55, 56
        ]:
            print(f"[Debug] Applying Fourier analysis for {distortion_type}")
            return self._fourier_analysis(x)  # ✅ Fourier 필터 적용

        else:
            print(f"[Warning] Unknown distortion type: {distortion_type}. Returning original input.")
            return x


    def _sobel_filter(self, x):
        """Sobel 필터 적용 (Gradient Magnitude + Direction)"""
        c = x.size(1)
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).to(x.device).unsqueeze(0)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).to(x.device).unsqueeze(0)

        if c > 1:
            sobel_x = sobel_x.repeat(c, 1, 1, 1)
            sobel_y = sobel_y.repeat(c, 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=c)

        # ✅ 그래디언트 크기(G) 계산
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # ✅ 그래디언트 방향(θ) 계산 (라디안 값, -π ~ π 범위)
        gradient_direction = torch.atan2(grad_y, grad_x)

        # ✅ 방향을 0~1 사이로 정규화 (원래 -π ~ π 범위이므로)
        gradient_direction_normalized = (gradient_direction + torch.pi) / (2 * torch.pi)

        return torch.sigmoid(gradient_magnitude.mean(dim=1, keepdim=True)), gradient_direction_normalized.mean(dim=1, keepdim=True)

    def _hsv_analysis(self, x):
        hsv = self._rgb_to_hsv(x)
        return hsv[:, 1:2, :, :]  # Saturation 채널 반환
    
    def _histogram_analysis(self, x):
        hist_map = torch.mean(x, dim=1, keepdim=True)
        return F.normalize(hist_map, p=2, dim=[-2, -1])  # L2 정규화 추가



    def _fourier_analysis(self, x):
        """
        푸리에 변환(Fourier Transform) 적용하여 주파수 도메인에서 왜곡 분석
        공간 도메인(픽셀단위정보) -> 주파수 도메인으로 변환
        -> 이미지 패턴이 어떻게 생겼는지 알 수 있음

        (1) 2D 푸리에 변환 정의:
            - 이산 푸리에 변환(DFT)은 이미지의 공간 도메인(Spatial Domain)에서 
            주파수 도메인(Frequency Domain)으로 변환하여 신호 분석을 수행함.

            - 여기서:
                * f(x, y): 원본 이미지의 픽셀 값
                * F(u, v): 주파수 도메인에서 변환된 값
                * (u, v): 주파수 좌표
                * (M, N): 이미지 크기

        (2) 푸리에 변환을 활용한 왜곡 탐지:
            - JPEG 압축 아티팩트(artifact), 블러(blur) 등의 왜곡을 감지할 수 있음.
            - 저주파 성분 (Low Frequency): 이미지의 전체적인 구조 및 밝기 정보.
            - 고주파 성분 (High Frequency): 세부 텍스처 및 경계 정보.
            - 블러(Blur) 및 압축 노이즈(Artifacts)는 주파수 도메인에서 고주파 성분을 감소시키는 특징이 있음.

        """

        # ✅ 입력 이미지 크기 가져오기
        h, w = x.shape[-2:]

        # ✅ 2의 거듭제곱 크기로 패딩하여 푸리에 변환 성능 최적화
        new_h, new_w = 2 ** int(np.ceil(np.log2(h))), 2 ** int(np.ceil(np.log2(w)))
        padded_x = F.pad(x, (0, new_w - w, 0, new_h - h))

        # ✅ 2D 푸리에 변환 수행 (주파수 도메인으로 변환)
        fft = torch.fft.fft2(padded_x, dim=(-2, -1))

        # ✅ 진폭(Amplitude) 계산 (주파수 도메인의 특정 성분 강조)
        magnitude = torch.abs(fft)

        # ✅ 진폭을 1채널로 평균 내어 반환 후 sigmoid 정규화 적용
        return torch.sigmoid(magnitude[:, :, :h, :w].mean(dim=1, keepdim=True))



    def _rgb_to_hsv(self, x):
        max_rgb, _ = x.max(dim=1, keepdim=True)
        min_rgb, _ = x.min(dim=1, keepdim=True)
        delta = max_rgb - min_rgb
        saturation = delta / torch.clamp(max_rgb, min=1e-6)
        value = max_rgb  # 추가된 value 채널
        return torch.cat((delta, saturation, value), dim=1)  # HSV 3채널 반환
    

""" class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(HardNegativeCrossAttention, self).__init__()

        if in_channels % num_heads != 0:
            num_heads = max(1, in_channels // 8)

        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.attribute_processor = AttributeFeatureProcessor(in_channels)
        self.texture_processor = TextureBlockProcessor(in_channels)

        # ✅ 학습 가능한 α 가중치 추가
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 초기값 0.5 (0~1 범위 학습 가능)

        # ✅ Scale Factor 적용 (S)
        self.scale_factor = nn.Parameter(torch.tensor(1.0 / (in_channels ** 0.5)))

        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])

    def forward(self, x_attr, x_texture):
        # ✅ Attribute Features 변환
        x_attr = self.attribute_processor(x_attr)

        # ✅ Texture Features 변환 후 추가 연산 적용
        x_texture = self.texture_processor(x_texture)
        x_texture = x_texture * x_attr  # 🔥 **곱셈 (X 연산) 추가**

        if x_attr.size(2) != x_texture.size(2) or x_attr.size(3) != x_texture.size(3):
            min_h = min(x_attr.size(2), x_texture.size(2))
            min_w = min(x_attr.size(3), x_texture.size(3))
            x_attr = F.adaptive_avg_pool2d(x_attr, (min_h, min_w))
            x_texture = F.adaptive_avg_pool2d(x_texture, (min_h, min_w))

        b, c, h, w = x_attr.size()
        head_dim = c // self.num_heads

        query_attr = self.query_conv(x_attr).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        key_attr = self.key_conv(x_attr).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 2, 3)
        value_attr = self.value_conv(x_attr).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        query_tex = self.query_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        key_tex = self.key_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 2, 3)
        value_tex = self.value_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        scale = self.scale_factor.to(query_attr.device)

        # ✅ Attribute Feature Attention
        attention_attr = self.softmax(torch.matmul(query_attr, key_attr) * scale)
        A_attr = torch.matmul(attention_attr, value_attr).permute(0, 1, 3, 2).contiguous()

        # ✅ Texture Feature Attention
        attention_tex = self.softmax(torch.matmul(query_tex, key_tex) * scale)
        A_tex = torch.matmul(attention_tex, value_tex).permute(0, 1, 3, 2).contiguous()

        # ✅ **α 가중치 적용**
        A_final = self.alpha * A_attr + (1 - self.alpha) * A_tex  # 🔥 **α 적용**

        A_final = A_final.view(b, c, h, w)
        A_final = self.output_proj(A_final)

        out = nn.Dropout(p=0.1)(A_final)

        if not hasattr(self, "layer_norm") or self.layer_norm.normalized_shape != (c, h, w):
            self.layer_norm = nn.LayerNorm([c, h, w]).to(out.device)

        out = self.layer_norm(out)
        return out """


class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(HardNegativeCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.attribute_processor = AttributeFeatureProcessor(in_channels)
        self.texture_processor = TextureBlockProcessor(in_channels)

        # ✅ 학습 가능한 α 가중치 추가
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 초기값 0.3 (0~1 범위 학습 가능)

        self.scale_factor = nn.Parameter(torch.tensor(1.0 / (in_channels ** 0.5)))
        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])

    def forward(self, x_attr, x_texture):
        # ✅ Attribute Features 변환
        x_attr = self.attribute_processor(x_attr)

        # ✅ Texture Block Features 변환 후 추가 연산 적용
        x_texture = self.texture_processor(x_texture)

        # ✅ Transpose 연산 추가 (다이어그램과 일치하도록 수정)
        x_texture = F.interpolate(x_texture, scale_factor=2, mode='bilinear', align_corners=False)

        # ✅ α 가중치 적용한 선형 조합
        x_weighted = self.alpha * x_attr + (1 - self.alpha) * x_texture  # 🔥 α 적용

        # ✅ Concatenation 적용 (C 연산)
        x_concat = torch.cat([x_weighted, x_texture], dim=1)  # 🔥 α 가중치 적용 후 결합

        # Attention 적용
        b, c, h, w = x_concat.size()
        head_dim = c // self.num_heads

        query = self.query_conv(x_concat).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        key = self.key_conv(x_concat).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 2, 3)
        value = self.value_conv(x_concat).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        scale = self.scale_factor.to(query.device)
        attention = self.softmax(torch.matmul(query, key) * scale)
        A_final = torch.matmul(attention, value).permute(0, 1, 3, 2).contiguous()

        # ✅ Concat한 뒤 최종 Projection 적용
        A_final = A_final.view(b, c, h, w)
        A_final = self.output_proj(A_final)

        # ✅ α 가중치 적용한 Residual Connection
        out = self.layer_norm(A_final + x_weighted)  # 🔥 α 적용 후 Residual Connection

        return out

""" class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(HardNegativeCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.attribute_processor = AttributeFeatureProcessor(in_channels)
        self.texture_processor = TextureBlockProcessor(in_channels)

        # ✅ 학습 가능한 α 가중치 추가 (초기값 0.7)
        self.alpha = nn.Parameter(torch.tensor(0.3))  

        self.scale_factor = nn.Parameter(torch.tensor(1.0 / (in_channels ** 0.5)))
        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])

    def forward(self, x_attr, x_texture):
        # ✅ Attribute Features 변환
        x_attr = self.attribute_processor(x_attr)

        # ✅ Texture Block Features 변환 후 추가 연산 적용
        x_texture = self.texture_processor(x_texture)

        # ✅ Transpose 연산 추가 (다이어그램과 일치하도록 수정)
        x_texture = F.interpolate(x_texture, scale_factor=2, mode='bilinear', align_corners=False)

        # ✅ α 가중치 적용한 선형 조합 (학습 가능)
        x_weighted = self.alpha * x_attr + (1 - self.alpha) * x_texture  # 🔥 α 적용

        # ✅ Concatenation 적용 (C 연산)
        x_concat = torch.cat([x_weighted, x_texture], dim=1)  # 🔥 α 가중치 적용 후 결합

        # Attention 적용
        b, c, h, w = x_concat.size()
        head_dim = c // self.num_heads

        query = self.query_conv(x_concat).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        key = self.key_conv(x_concat).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 2, 3)
        value = self.value_conv(x_concat).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        scale = self.scale_factor.to(query.device)
        attention = self.softmax(torch.matmul(query, key) * scale)
        A_final = torch.matmul(attention, value).permute(0, 1, 3, 2).contiguous()

        # ✅ Concat한 뒤 최종 Projection 적용
        A_final = A_final.view(b, c, h, w)
        A_final = self.output_proj(A_final)

        # ✅ α 가중치 적용한 Residual Connection
        out = self.layer_norm(A_final + x_weighted)  # 🔥 α 적용 후 Residual Connection

        return out """




# 시각화 함수1
def visualize_distortion_classification(input_image, distortion_logits):
    distortion_probs = torch.softmax(distortion_logits, dim=1).detach().cpu().numpy()[0]

    # distortion_labels 크기를 distortion_probs 개수와 맞춤
    #distortion_labels = list(distortion_map.keys())[:len(distortion_probs)]  # 개수 일치

    distortion_labels = list(distortion_map.keys())
    if len(distortion_labels) != len(distortion_probs):
        distortion_labels = distortion_labels[:len(distortion_probs)]

    plt.figure(figsize=(12, 6))
    
    # 입력 이미지 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis("off")

    # 왜곡 분류 결과 시각화
    plt.subplot(1, 2, 2)
    plt.bar(distortion_labels, distortion_probs)
    plt.title("Distortion Classification Probabilities")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# 푸리에 시각화
""" def apply_fft(image_tensor):
    fft = torch.fft.fft2(image_tensor, dim=(-2, -1))
    magnitude = torch.abs(fft)  # 진폭(Amplitude) 계산
    log_magnitude = torch.log1p(magnitude)  # 로그 스케일 변환 (시각적으로 보기 편하도록)
    return log_magnitude

def high_pass_filter(fft, cutoff=10):
    h, w = fft.shape[-2:]
    center_h, center_w = h // 2, w // 2

    mask = torch.ones_like(fft)  # 전체를 1로 초기화
    mask[:, :, center_h - cutoff:center_h + cutoff, center_w - cutoff:center_w + cutoff] = 0  # 저주파 제거

    return fft * mask  # 고주파 부분만 남김

def low_pass_filter(fft, cutoff=10):
    h, w = fft.shape[-2:]
    center_h, center_w = h // 2, w // 2

    mask = torch.zeros_like(fft)  # 전체를 0으로 초기화
    mask[:, :, center_h - cutoff:center_h + cutoff, center_w - cutoff:center_w + cutoff] = 1  # 저주파 유지

    return fft * mask  # 저주파 부분만 남김

def apply_ifft(fft):
    return torch.fft.ifft2(fft, dim=(-2, -1)).real

def visualize_fourier(image_path, cutoff=10):
    
    # ✅ 원본 이미지 로드 및 변환
    image = Image.open(image_path).convert("L")  # 그레이스케일 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # (1, 1, 224, 224)

    # ✅ 푸리에 변환 수행
    fft = torch.fft.fft2(image_tensor, dim=(-2, -1))

    # ✅ 고주파 및 저주파 필터 적용
    high_pass_fft = high_pass_filter(fft, cutoff)
    low_pass_fft = low_pass_filter(fft, cutoff)

    # ✅ 역 푸리에 변환 수행
    high_pass_reconstructed = apply_ifft(high_pass_fft)
    low_pass_reconstructed = apply_ifft(low_pass_fft)

    # ✅ 시각화를 위한 데이터 변환
    log_fft = apply_fft(image_tensor)
    log_high_pass = apply_fft(high_pass_fft)
    log_low_pass = apply_fft(low_pass_fft)

    # ✅ 그래프 그리기
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    ax[0, 0].imshow(image, cmap="gray")
    ax[0, 0].set_title("원본 이미지 (Spatial Domain)")

    ax[0, 1].imshow(log_fft.squeeze().numpy(), cmap="gray")
    ax[0, 1].set_title("푸리에 변환 결과 (Frequency Domain)")

    ax[0, 2].imshow(log_high_pass.squeeze().numpy(), cmap="gray")
    ax[0, 2].set_title("고주파 강조 (High-Pass Filtering)")

    ax[1, 0].imshow(log_low_pass.squeeze().numpy(), cmap="gray")
    ax[1, 0].set_title("저주파 강조 (Low-Pass Filtering)")

    ax[1, 1].imshow(high_pass_reconstructed.squeeze().numpy(), cmap="gray")
    ax[1, 1].set_title("역 푸리에 변환 (High-Pass)")

    ax[1, 2].imshow(low_pass_reconstructed.squeeze().numpy(), cmap="gray")
    ax[1, 2].set_title("역 푸리에 변환 (Low-Pass)")

    plt.tight_layout()
    plt.show()

# ✅ 실행: 논문에 넣을 시각화 생성
image_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/dst_imgs/BLUR/1600.BLUR.5.png"  # 원본 이미지 경로 설정
visualize_fourier(image_path, cutoff=20)

 """

# Main Execution for Debugging
if __name__ == "__main__":
    # 테스트를 위한 임의 입력 데이터
    batch_size = 32
    in_channels = 2048
    height, width = 7, 7  # 입력 크기

        # 입력 이미지 로드
    input_image_path = r"E:\ARNIQA - SE - mix\ARNIQA\dataset\KONIQ10K\1024x768\11706252.jpg"
    input_image = Image.open(input_image_path).convert("RGB")
    input_tensor = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])(input_image).unsqueeze(1)  # 1채널 -> 3채널 확장 필요

    # DistortionAttention 모델 초기화
    in_channels = 3  # RGB 3채널로 설정
    distortion_attention = DistortionAttention(in_channels=in_channels)

    # DistortionAttention 적용
    with torch.no_grad():
        output, distortion_logits = distortion_attention(input_tensor.repeat(1, 3, 1, 1))  # 1채널 → 3채널 확장

    # 결과 시각화
    visualize_distortion_classification(np.array(input_image), distortion_logits)




    x_attr = torch.randn(batch_size, in_channels, height, width)  # Attribute Features
    x_texture = torch.randn(batch_size, in_channels, height, width)  # Texture Block Features

    print(f"Input x_attr shape: {x_attr.shape}")
    print(f"Input x_texture shape: {x_texture.shape}")

    # HardNegativeCrossAttention 초기화
    hnca = HardNegativeCrossAttention(in_channels=in_channels, num_heads=8)

    # Forward Pass
    output = hnca(x_attr, x_texture)
    print("HardNegativeCrossAttention Output shape:", output.shape)

    # Global Average Pooling 적용
    global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    pooled_output = global_avg_pool(output)
    print("Global Avg Pool Output shape:", pooled_output.shape)

    # Projection Head
    proj_head = nn.Linear(in_channels, 128)
    final_output = proj_head(pooled_output.view(batch_size, -1))
    print("Projection Head Output shape:", final_output.shape)
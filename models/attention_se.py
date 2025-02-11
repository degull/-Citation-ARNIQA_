
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
    # JPEG 압축 관련 (Sobel)
    "jpeg": 0,  
    "jpeg2000": 1,    
    "jpeg_artifacts": 2,
    "jpeg_transmission_errors": 3,
    "jpeg2000_transmission_errors": 4,

    # 블러 관련 (Sobel)
    "gaussian_blur": 5,  
    "lens_blur": 6,
    "motion_blur": 7,
    "blur": 8,

    # 노이즈 관련 (Sobel)
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

    # 대비 및 색상 관련 (HSV)
    "contrast_change": 20,
    "contrast": 21,
    "brightness": 22,
    "colorfulness": 23,
    "sharpness": 24,
    "exposure": 25,
    "overexposure": 26,
    "underexposure": 27,
    "color_shift": 28,
    "color_saturation_1": 29,
    "color_saturation_2": 30,
    "change_of_color_saturation": 31,
    "white_balance_error": 32,

    # 공간적 왜곡 관련 (Sobel)
    "fnoise": 33,
    "fast_fading": 34,
    "color_diffusion": 35,
    "mean_shift": 36,
    "jitter": 37,
    "non_eccentricity_patch": 38,
    "pixelate": 39,
    "spatial_noise": 40,
    "non_eccentricity_pattern_noise": 41,
    "local_block_wise_distortions": 42,

    # 양자화 및 압축 관련 (Fourier)
    "quantization": 43,
    "color_quantization": 44,
    "image_color_quantization_with_dither": 45,
    "color_block": 46,
    "sparse_sampling_and_reconstruction": 47,

    # 영상 왜곡 (Fourier)
    "glare": 48,
    "haze": 49,
    "banding_artifacts": 50,
    "vignetting": 51,
    "chromatic_aberration": 52,
    "distortion": 53,
    "high_sharpen": 54,
    "image_denoising": 55
}


class DistortionClassifier(nn.Module):
    def __init__(self, in_channels, num_distortions=48):
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
    def __init__(self, in_channels, expected_channels=256):  # 🔥 expected_channels 추가
        super(DistortionAttention, self).__init__()
        self.expected_channels = expected_channels  # ✅ 클래스 변수로 저장

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
        x = self.channel_adjust(x)  # 🔥 512 → 256으로 변환

        distortion_logits = self.distortion_classifier(x)
        distortion_types = torch.argmax(distortion_logits, dim=1)

        # ✅ 필터 적용 후 채널 크기 맞춤
        filtered_tensors = []
        for i, dt in enumerate(distortion_types):
            filtered_x = self._apply_filter(x[i].unsqueeze(0), dt)

            # ✅ 1채널이면 expected_channels(256)로 확장
            if filtered_x.shape[1] == 1:
                filtered_x = filtered_x.expand(-1, self.expected_channels, -1, -1)  # 🔥 수정된 코드

            filtered_tensors.append(filtered_x)
        
        filtered_x = torch.cat(filtered_tensors, dim=0)
        
        query = self.query_conv(filtered_x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(filtered_x).view(b, -1, h * w)
        value = self.value_conv(filtered_x).view(b, -1, h * w)
        
        scale = query.size(-1) ** -0.5
        attention = self.softmax(torch.bmm(query, key) * scale)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, self.expected_channels, h, w)

        return out + x, distortion_logits


    def _apply_filter(self, x, distortion_type):
        """ 왜곡 유형에 따라 적절한 필터 적용 """
        
        if distortion_type in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
            11, 12, 13, 14, 15, 16, 17, 18, 19, 
            33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 55
        ]:
            print(f"[Debug] Applying Sobel filter for {distortion_type}")
            return self._sobel_filter(x)

        elif distortion_type in [
            20, 21, 22, 23, 24, 25, 26, 27, 28, 
            29, 30, 31, 32
        ]:
            print(f"[Debug] Applying HSV analysis for {distortion_type}")
            return self._hsv_analysis(x)

        elif distortion_type in [
            43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
        ]:
            print(f"[Debug] Applying Fourier analysis for {distortion_type}")
            return self._fourier_analysis(x)

        else:
            print(f"[Warning] Unknown distortion type: {distortion_type}. Returning original input.")
            return x





    def _sobel_filter(self, x):
        """Sobel 필터 적용"""
        c = x.size(1)
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).to(x.device).unsqueeze(0)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).to(x.device).unsqueeze(0)

        if c > 1:
            sobel_x = sobel_x.repeat(c, 1, 1, 1)
            sobel_y = sobel_y.repeat(c, 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=c)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return torch.sigmoid(gradient_magnitude.mean(dim=1, keepdim=True))  # 1채널로 반환

    def _hsv_analysis(self, x):
        hsv = self._rgb_to_hsv(x)
        return hsv[:, 1:2, :, :]  # Saturation 채널 반환
    
    def _histogram_analysis(self, x):
        hist_map = torch.mean(x, dim=1, keepdim=True)
        return F.normalize(hist_map, p=2, dim=[-2, -1])  # L2 정규화 추가


    def _fourier_analysis(self, x):
        h, w = x.shape[-2:]
        new_h, new_w = 2 ** int(np.ceil(np.log2(h))), 2 ** int(np.ceil(np.log2(w)))
        padded_x = F.pad(x, (0, new_w - w, 0, new_h - h))

        # ✅ `view_as_real()` 대신 `torch.abs()` 사용
        fft = torch.fft.fft2(padded_x, dim=(-2, -1))
        magnitude = torch.abs(fft)
        return torch.sigmoid(magnitude[:, :, :h, :w].mean(dim=1, keepdim=True))



    def _rgb_to_hsv(self, x):
        max_rgb, _ = x.max(dim=1, keepdim=True)
        min_rgb, _ = x.min(dim=1, keepdim=True)
        delta = max_rgb - min_rgb
        saturation = delta / torch.clamp(max_rgb, min=1e-6)
        value = max_rgb  # 추가된 value 채널
        return torch.cat((delta, saturation, value), dim=1)  # HSV 3채널 반환

    

class HardNegativeCrossAttention(nn.Module):
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
        return out


# 시각화 함수1
def visualize_distortion_classification(input_image, distortion_logits):
    distortion_probs = torch.softmax(distortion_logits, dim=1).detach().cpu().numpy()[0]

    # distortion_labels 크기를 distortion_probs 개수와 맞춤
    distortion_labels = list(distortion_map.keys())[:len(distortion_probs)]  # 개수 일치

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
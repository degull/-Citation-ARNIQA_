
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
    "jpeg": 0,  # Sobel
    "jpeg2000": 1,    #Sobel
    "gaussian_blur": 2, #Sobel
    "white_noise": 3,   #Sobel
    "contrast_change": 4,  #HSV

    "fast_fading": 5,  #Fourier
    "awgn": 6,   #Sobel
    "blur": 7,   #Sobel
    "contrast": 8, #HSV
    "fnoise": 9,   #Fourier

    "brightness": 10,  #HSV
    "colorfulness": 11,    #HSV
    "sharpness": 12, #Sobel
    "exposure": 13,     #HSV

    "lens_blur": 14,  #Sobel
    "motion_blur": 15,    #Sobel
    "color_diffusion": 16,    #Sobel
    "color_shift": 17,  #HSV

    "impulse_noise": 18,  #Sobel
    "multiplicative_noise": 19,     #Fourier
    "denoise": 20,  #Fourier
    "mean_shift": 21,  #Histogram
    "jitter": 22,  #Fourier
    "non_eccentricity_patch": 23,     #Sobel
    "pixelate": 24,   #Sobel
    "quantization": 25,     #Fourier
    "color_block": 26,  #Fourier
    "high_sharpen":27,     #Fourier

    "spatial_noise": 28,  #Sobel
    "color_saturation": 29,     #HSV
    "color_quantization": 30,   #Fourier
    "overexposure": 31,     #HSV
    "underexposure": 32,    #HSV
    "chromatic_aberration": 33,   #Sobel

    "additive_gaussian_noise": 34,    #Sobel
    "additive_noise_in_color_components": 35,    #Sobel
    "spatially_correlated_noise": 36,     #Sobel
    "masked_noise": 37,   #Sobel
    "high_frequency_noise": 38,   #Sobel
    "image_denoising": 39,  #Fourier
    "jpeg_transmission_errors": 40,   #Sobel
    "jpeg2000_transmission_errors": 41,   #Sobel
    "non_eccentricity_pattern_noise": 42,     #Sobel
    "local_block_wise_distortions": 43,  #Sobel
    "comfort_noise": 44,    #Fourier
    "lossy_compression_of_noisy_images": 45,    #Fourier
    "image_color_quantization_with_dither": 46,     #Fourier
    "sparse_sampling_and_reconstruction": 47,   #Fourier
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
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 첫 번째 Conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 두 번째 Conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 세 번째 Conv
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
        if distortion_type in [0, 1, 2, 3, 6, 7, 12, 14, 15, 16, 18, 23, 24, 28, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43]:
            print(f"[Debug] Applying Sobel filter for distortion type: {distortion_type}")
            return self._sobel_filter(x)  # ✅ Sobel 필터 적용 (다른 필터와 독립적으로 실행)

        elif distortion_type in [4, 8, 10, 11, 13, 17, 29, 31, 32]:
            print(f"[Debug] Applying HSV analysis for distortion type: {distortion_type}")
            return self._hsv_analysis(x)  # ✅ HSV 필터 적용

        elif distortion_type == 21:
            print(f"[Debug] Applying Histogram analysis for distortion type: {distortion_type}")
            return self._histogram_analysis(x)  # ✅ Histogram 필터 적용

        elif distortion_type in [5, 9, 19, 20, 22, 25, 26, 27, 30, 39, 44, 45, 46, 47]:
            print(f"[Debug] Applying Fourier analysis for distortion type: {distortion_type}")
            return self._fourier_analysis(x)  # ✅ Fourier 필터 적용

        else:
            print(f"[Warning] Unknown distortion type: {distortion_type}, returning original input.")
            return x  # ✅ 필터 미적용 (원본 그대로 반환)




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

        # num_heads 자동 조정
        if in_channels % num_heads != 0:
            print(f"[경고] in_channels={in_channels}는 num_heads={num_heads}로 나눌 수 없습니다.")
            num_heads = max(1, in_channels // 8)
            print(f"[수정] num_heads={num_heads}로 변경되었습니다.")

        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.attribute_processor = AttributeFeatureProcessor(in_channels)
        self.texture_processor = TextureBlockProcessor(in_channels)

        # ✅ 미리 LayerNorm 초기화
        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])

    def forward(self, x_attr, x_texture):
        x_attr = self.attribute_processor(x_attr)
        x_texture = self.texture_processor(x_texture)

        if x_attr.size(2) != x_texture.size(2) or x_attr.size(3) != x_texture.size(3):
            min_h = min(x_attr.size(2), x_texture.size(2))
            min_w = min(x_attr.size(3), x_texture.size(3))
            x_attr = F.adaptive_avg_pool2d(x_attr, (min_h, min_w))
            x_texture = F.adaptive_avg_pool2d(x_texture, (min_h, min_w))

        b, c, h, w = x_attr.size()
        head_dim = c // self.num_heads

        multi_head_query = self.query_conv(x_attr).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        multi_head_key = self.key_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 2, 3)
        multi_head_value = self.value_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32).clamp(min=1e-6)).to(multi_head_query.device)
        attention = self.softmax(torch.matmul(multi_head_query, multi_head_key) / scale)
        out = torch.matmul(attention, multi_head_value).permute(0, 1, 3, 2).contiguous()

        out = out.view(b, c, h, w)
        out = self.output_proj(out)
        out = nn.Dropout(p=0.1)(out) + x_attr

        # ✅ `self.layer_norm`이 None일 경우 즉시 생성
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
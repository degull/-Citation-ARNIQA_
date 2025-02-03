
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

# Distortion name to number mapping
distortion_map = {
    "gaussian_blur": 0,  # Sobel 필터
    "lens_blur": 1,  # Sobel 필터
    "motion_blur": 2,  # Sobel 필터
    "color_diffusion": 3,  # Sobel 필터
    "color_shift": 4,  # Sobel 필터
    "color_quantization": 5,  # Sobel 필터
    "color_saturation_1": 6,  # HSV 색공간 분석
    "color_saturation_2": 7,  # HSV 색공간 분석
    "jpeg2000": 8,  # Sobel 필터
    "jpeg": 9,  # Sobel 필터
    "white_noise": 10,  # Sobel 필터
    "white_noise_color_component": 11,  # Sobel 필터
    "impulse_noise": 12,  # Sobel 필터
    "multiplicative_noise": 13,  # Sobel 필터
    "denoise": 14,  # Fourier Transform
    "brighten": 15,  # HSV 색공간 분석
    "darken": 16,  # HSV 색공간 분석
    "mean_shift": 17,  # 히스토그램 분석
    "jitter": 18,  # Fourier Transform
    "non_eccentricity_patch": 19,  # Sobel 필터
    "pixelate": 20,  # Sobel 필터
    "quantization": 21,  # Fourier Transform
    "color_block": 22,  # Fourier Transform
    "high_sharpen": 23,  # Fourier Transform
    "contrast_change": 24  # Fourier Transform
}

class DistortionClassifier(nn.Module):
    def __init__(self, in_channels, num_distortions=25):
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
    def __init__(self, in_channels):
        super(DistortionAttention, self).__init__()
        query_key_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, query_key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, query_key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.distortion_classifier = DistortionClassifier(in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        distortion_logits = self.distortion_classifier(x)
        distortion_types = torch.argmax(distortion_logits, dim=1)
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)
        value = self.value_conv(x).view(b, -1, h * w)
        scale = query.size(-1) ** -0.5
        attention = self.softmax(torch.bmm(query, key) * scale)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)
        return out + x , distortion_logits

    def _apply_filter(self, x, distortion_type):
        if isinstance(distortion_type, str):
            distortion_type = distortion_map.get(distortion_type, -1)

        if distortion_type in [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 19, 20]:  # Sobel applicable distortions
            return self._sobel_filter(x)
        elif distortion_type in [6, 7, 15, 16]:  # HSV analysis applicable distortions
            return self._hsv_analysis(x)
        elif distortion_type == 17:  # Histogram analysis applicable distortions
            return self._histogram_analysis(x)
        elif distortion_type in [14, 18, 21, 22, 23, 24]:  # Fourier Transform applicable distortions
            return self._fourier_analysis(x)
        else:
            return torch.ones_like(x[:, :1, :, :])

    def _sobel_filter(self, x):
        sobel_x = self.sobel_x.repeat(x.size(1), 1, 1, 1).to(x.device)  # Repeat for each input channel
        sobel_y = self.sobel_y.repeat(x.size(1), 1, 1, 1).to(x.device)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))  # Convolution with Sobel X
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))  # Convolution with Sobel Y
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # Compute gradient magnitude

        # Normalize gradient magnitude
        return torch.sigmoid(gradient_magnitude.mean(dim=1, keepdim=True))

    def _hsv_analysis(self, x):
        hsv = self._rgb_to_hsv(x)
        return hsv[:, 1:2, :, :]  # Saturation channel

    def _histogram_analysis(self, x):
        hist_map = torch.mean(x, dim=1, keepdim=True)
        return torch.sigmoid(hist_map)

    def _fourier_analysis(self, x):
        # 입력 텐서를 패딩하여 크기를 2의 거듭제곱으로 만듭니다.
        h, w = x.shape[-2:]
        new_h = 2 ** int(np.ceil(np.log2(h)))
        new_w = 2 ** int(np.ceil(np.log2(w)))
        
        # 패딩 추가
        padded_x = F.pad(x, (0, new_w - w, 0, new_h - h))  # (left, right, top, bottom)
        
        # FFT 계산
        fft = torch.fft.fft2(padded_x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.sqrt(fft_shift.real ** 2 + fft_shift.imag ** 2)
        
        # 원래 크기로 잘라내기
        magnitude = magnitude[:, :, :h, :w]
        
        # 출력 결과 정규화 및 반환
        return torch.sigmoid(magnitude.mean(dim=1, keepdim=True))


    def _rgb_to_hsv(self, x):
        max_rgb, _ = x.max(dim=1, keepdim=True)
        min_rgb, _ = x.min(dim=1, keepdim=True)
        delta = max_rgb - min_rgb + 1e-6
        saturation = delta / (max_rgb + 1e-6)
        value = max_rgb
        return torch.cat((delta, saturation, value), dim=1)


class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(HardNegativeCrossAttention, self).__init__()
        # num_heads와 in_channels가 호환되지 않을 경우 조정
        if in_channels % num_heads != 0:
            print(f"[경고] in_channels={in_channels}는 num_heads={num_heads}로 나눌 수 없습니다.")
            num_heads = max(1, in_channels // 8)  # 적절한 num_heads 재설정
            print(f"[수정] num_heads={num_heads}로 변경되었습니다.")

        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = None

        self.attribute_processor = AttributeFeatureProcessor(in_channels)
        self.texture_processor = TextureBlockProcessor(in_channels)

    def forward(self, x_attr, x_texture):
        # Process attribute and texture features
        x_attr = self.attribute_processor(x_attr)
        x_texture = self.texture_processor(x_texture)

        # Ensure compatibility for feature dimensions
        if x_attr.size(2) != x_texture.size(2) or x_attr.size(3) != x_texture.size(3):
            min_h = min(x_attr.size(2), x_texture.size(2))
            min_w = min(x_attr.size(3), x_texture.size(3))
            x_attr = F.adaptive_avg_pool2d(x_attr, (min_h, min_w))
            x_texture = F.adaptive_avg_pool2d(x_texture, (min_h, min_w))

        b, c, h, w = x_attr.size()
        head_dim = c // self.num_heads

        # Query, Key, Value 계산
        multi_head_query = self.query_conv(x_attr).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        multi_head_key = self.key_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 2, 3)
        multi_head_value = self.value_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        # Scale factor 및 Attention 계산
        scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32).clamp(min=1e-6)).to(multi_head_query.device)
        attention = self.softmax(torch.matmul(multi_head_query, multi_head_key) / scale)
        out = torch.matmul(attention, multi_head_value).permute(0, 1, 3, 2).contiguous()

        # Reshape 및 Output Projection
        out = out.view(b, c, h, w)
        out = self.output_proj(out)
        out = nn.Dropout(p=0.1)(out) + x_attr

        # Apply LayerNorm
        if self.layer_norm is None or self.layer_norm.normalized_shape != (c, h, w):
            self.layer_norm = nn.LayerNorm([c, h, w]).to(out.device)

        out = self.layer_norm(out)
        return out


# 시각화 함수
def visualize_distortion_classification(input_image, distortion_logits):
    distortion_probs = torch.softmax(distortion_logits, dim=1).detach().cpu().numpy()[0]
    distortion_labels = list(distortion_map.keys())
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
        transforms.Resize((64, 64)),  # 이미지 크기를 64x64로 축소
        transforms.ToTensor()
    ])(input_image).unsqueeze(0)

    # DistortionAttention 모델 초기화
    in_channels = input_tensor.size(1)
    distortion_attention = DistortionAttention(in_channels=in_channels)

    # DistortionAttention 통과
    with torch.no_grad():
        output, distortion_logits = distortion_attention(input_tensor)

    # 분류 결과 시각화
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
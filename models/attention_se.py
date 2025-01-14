""" import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Distortion name to number mapping
distortion_map = {
    "gaussian_blur": 0, # Sobel 필터
    "lens_blur": 1, # Sobel 필터
    "motion_blur": 2, # Sobel 필터
    "color_diffusion": 3, # Sobel 필터
    "color_shift": 4, # Sobel 필터
    "color_quantization": 5, # Sobel 필터
    "color_saturation_1": 6, # HSV 색공간 분석
    "color_saturation_2": 7,# HSV 색공간 분석
    "jpeg2000": 8, # Sobel 필터
    "jpeg": 9, # Sobel 필터
    "white_noise": 10, # Sobel 필터
    "white_noise_color_component": 11, # Sobel 필터
    "impulse_noise": 12, # Sobel 필터
    "multiplicative_noise": 13, # Sobel 필터
    "denoise": 14,  # Fourier Transform
    "brighten": 15, # HSV 색공간 분석
    "darken": 16, # HSV 색공간 분석
    "mean_shift": 17, # 히스토그램 분석
    "jitter": 18,  # Fourier Transform
    "non_eccentricity_patch": 19,   # Sobel 필터
    "pixelate": 20, # Sobel 필터
    "quantization": 21,  # Fourier Transform
    "color_block": 22,  # Fourier Transform
    "high_sharpen": 23,  # Fourier Transform
    "contrast_change": 24  # Fourier Transform
}

class DistortionClassifier(nn.Module):
    def __init__(self, in_channels, num_distortions=25):
        super(DistortionClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, num_distortions)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x



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
        return out + x  # Residual connection

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

    #def _fourier_analysis(self, x):
    #    fft = torch.fft.fft2(x, dim=(-2, -1))
    #    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    #    magnitude = torch.sqrt(fft_shift.real ** 2 + fft_shift.imag ** 2)
    #    return torch.sigmoid(magnitude.mean(dim=1, keepdim=True))

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

# 수정 전
class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(HardNegativeCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_attr, x_texture):
        b, c, h, w = x_attr.size()

        # Generate query, key, value tensors
        query = self.query_conv(x_attr).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x_texture).view(b, -1, h * w)
        value = self.value_conv(x_texture).view(b, -1, h * w).permute(0, 2, 1)

        # Scale factor for stability
        scale = query.size(-1) ** -0.5  # Scaling with sqrt(d_k)
        attention = self.softmax(torch.bmm(query, key) * scale)
        out = torch.bmm(attention, value).permute(0, 2, 1).contiguous().view(b, c, h, w)

        return out + x_attr  # Residual connection


# 수정 후 
class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(HardNegativeCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = None  # 동적 LayerNorm

    def forward(self, x_attr, x_texture):
        b, c, h, w = x_attr.size()
        head_dim = c // self.num_heads
        assert c % self.num_heads == 0, "Number of heads must divide channels evenly."

        # Generate query, key, value tensors
        query = self.query_conv(x_attr).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        key = self.key_conv(x_texture).view(b, self.num_heads, head_dim, h * w)
        value = self.value_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        # Scale factor for stability
        scale = head_dim ** -0.5
        attention = self.softmax(torch.matmul(query, key) * scale)  # Shape: (b, num_heads, hw, hw)
        out = torch.matmul(attention, value).permute(0, 1, 3, 2).contiguous()

        # Reshape and merge heads
        out = out.view(b, c, h, w)

        # Output projection and residual connection
        out = self.output_proj(out) + x_attr

        # 동적 LayerNorm 적용
        if self.layer_norm is None or self.layer_norm.normalized_shape != (c, h, w):
            self.layer_norm = nn.LayerNorm([c, h, w]).to(out.device)

        out = self.layer_norm(out)
        return out



if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)  # Random input tensor
    distortion_attention = DistortionAttention(in_channels=3)
    output = distortion_attention(x)
    print("DistortionAttention Output shape:", output.shape)


 """

### 처리 흐름:
""" 
1.입력 이미지 분석
   - DistortionClassifier를 통해 입력 이미지의 왜곡 유형을 분류합니다.
   - 분류 결과는 25가지 왜곡 유형 중 하나로 나타납니다.

2. 왜곡 유형에 따른 필터 선택
   - 왜곡 유형이 숫자로 매핑되며, 이 숫자는 distortion_map에서 해당하는 필터(소벨, HSV, 히스토그램, 푸리에)로 연결됩니다.
   - DistortionAttention 클래스는 _apply_filter 메서드를 통해 왜곡 유형에 맞는 필터를 호출합니다.

3. 필터 적용
   - 왜곡 유형에 적합한 필터를 사용하여 이미지에 적용합니다.
   - 예를 들어, color_saturation_1과 같은 왜곡은 HSV 분석 필터가 적용됩니다.

4. **결과 생성**:
   - 처리된 필터 결과는 원래 입력 이미지와 결합하여 최종 출력으로 반환됩니다.

---

### 장점:
- **자동 왜곡 감지**: 사용자가 `distortion_type`을 명시적으로 입력하지 않아도, 모델이 `DistortionClassifier`를 통해 자동으로 왜곡 유형을 감지합니다.
- **확장 가능성**: 새로운 왜곡 유형을 추가하려면 `distortion_map`과 관련 필터 로직만 업데이트하면 됩니다.
- **모듈화된 필터**: 각 필터(소벨, HSV, 히스토그램, 푸리에)가 독립적으로 설계되어 유지보수가 용이합니다.

---

### 테스트 및 확인:
테스트 단계에서:
- 다양한 유형의 왜곡이 적용된 샘플 이미지를 모델에 입력합니다.
- 출력이 원래 이미지와 비교하여 각 필터가 적절히 적용되었는지 확인합니다.

### 기대 결과:
- 사용자가 임의의 이미지를 넣으면, 모델은 입력 이미지의 왜곡 유형을 감지하여 해당 유형에 적합한 필터를 사용해 이미지를 처리할 수 있습니다. """

import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
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
        return out + x  # Residual connection

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

    #def _fourier_analysis(self, x):
    #    fft = torch.fft.fft2(x, dim=(-2, -1))
    #    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    #    magnitude = torch.sqrt(fft_shift.real ** 2 + fft_shift.imag ** 2)
    #    return torch.sigmoid(magnitude.mean(dim=1, keepdim=True))

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

""" class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(HardNegativeCrossAttention, self).__init__()
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

        b, c, h, w = x_attr.size()
        head_dim = c // self.num_heads
        assert c % self.num_heads == 0, "Number of heads must divide channels evenly."

        # Debugging: Check tensor sizes
        print(f"[Debug] x_attr shape after attribute processor: {x_attr.size()}")
        print(f"[Debug] x_texture shape after texture processor: {x_texture.size()}")
        print(f"[Debug] head_dim: {head_dim}, h: {h}, w: {w}")

        # Ensure compatibility for view
        query = self.query_conv(x_attr).view(b, self.num_heads, head_dim, -1).permute(0, 1, 3, 2)
        key = self.key_conv(x_texture).view(b, self.num_heads, head_dim, -1).permute(0, 1, 3, 2)
        value = self.value_conv(x_texture).view(b, self.num_heads, head_dim, -1).permute(0, 1, 3, 2)

        # Debugging: Check intermediate sizes
        print(f"[Debug] query shape: {query.size()}")
        print(f"[Debug] key shape: {key.size()}")
        print(f"[Debug] value shape: {value.size()}")

        # Scale factor for stability
        scale = head_dim ** -0.5
        attention = self.softmax(torch.matmul(query, key.transpose(-2, -1)) * scale)
        out = torch.matmul(attention, value).permute(0, 1, 3, 2).contiguous()

        # Reshape and merge heads
        out = out.view(b, c, h, w)

        # Output projection and residual connection
        out = self.output_proj(out) + x_attr

        # Apply dynamic LayerNorm
        if self.layer_norm is None or self.layer_norm.normalized_shape != (c, h, w):
            self.layer_norm = nn.LayerNorm([c, h, w]).to(out.device)

        out = self.layer_norm(out)
        return out
 """

class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(HardNegativeCrossAttention, self).__init__()
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
        assert c % self.num_heads == 0, "Number of heads must divide channels evenly."

        # Reshape query, key, and value
        multi_head_query = self.query_conv(x_attr).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        multi_head_key = self.key_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 2, 3)
        multi_head_value = self.value_conv(x_texture).view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        # Scale factor for stability
        scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32).clamp(min=1e-6)).to(multi_head_query.device)
        attention = self.softmax(torch.matmul(multi_head_query, multi_head_key) / scale)
        out = torch.matmul(attention, multi_head_value).permute(0, 1, 3, 2).contiguous()

        # Reshape and apply output projection
        out = out.view(b, c, h, w)
        out = self.output_proj(out)
        out = nn.Dropout(p=0.1)(out) + x_attr

        # Apply dynamic LayerNorm
        if self.layer_norm is None or self.layer_norm.normalized_shape != (c, h, w):
            self.layer_norm = nn.LayerNorm([c, h, w]).to(out.device)

        out = self.layer_norm(out)
        return out


# Main Execution for Debugging
if __name__ == "__main__":
    # 테스트를 위한 임의 입력 데이터
    batch_size = 32
    in_channels = 2048
    height, width = 7, 7  # 입력 크기

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

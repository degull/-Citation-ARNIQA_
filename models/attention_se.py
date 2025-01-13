import torch
import torch.nn as nn
import torch.nn.functional as F

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

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

        self.spatial_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.channel_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.distortion_classifier = DistortionClassifier(in_channels)

    def forward(self, x):
        b, c, h, w = x.size()

        # Predict distortion type
        distortion_logits = self.distortion_classifier(x)
        distortion_types = torch.argmax(distortion_logits, dim=1)

        # Query, Key, Value tensors
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)
        value = self.value_conv(x).view(b, -1, h * w)

        # Attention computation
        scale = query.size(-1) ** -0.5
        attention = self.softmax(torch.bmm(query, key) * scale)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)

        # Apply filter based on predicted distortion type for each sample
        spatial_maps = []
        for i in range(b):
            distortion_type = distortion_types[i].item()
            spatial_map = self._apply_filter(x[i:i+1], distortion_type)
            spatial_maps.append(spatial_map)

        # Concatenate spatial maps
        spatial_maps = torch.cat(spatial_maps, dim=0)

        # Resize spatial_maps if needed to match out size
        if spatial_maps.size(2) != h or spatial_maps.size(3) != w:
            spatial_maps = F.interpolate(spatial_maps, size=(h, w), mode='bilinear', align_corners=False)

        # Combine attention output with spatial maps
        out = out * (self.spatial_weight * spatial_maps)
        return out + x  # Residual connection


    def _apply_filter(self, x, distortion_type):
        if isinstance(distortion_type, str):
            distortion_type = distortion_map.get(distortion_type, -1)

        if distortion_type in [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 19, 20]:  # Sobel applicable distortions
            spatial_map = self._sobel_filter(x)
        elif distortion_type in [6, 7, 15, 16]:  # HSV analysis applicable distortions
            spatial_map = self._hsv_analysis(x)
        elif distortion_type == 17:  # Histogram analysis applicable distortions
            spatial_map = self._histogram_analysis(x)
        elif distortion_type in [14, 18, 21, 22, 23, 24]:  # Fourier Transform applicable distortions
            spatial_map = self._fourier_analysis(x)
        else:
            spatial_map = torch.ones_like(x[:, :1, :, :])

        # Ensure spatial_map size matches input size
        _, _, h, w = x.size()
        if spatial_map.size(2) != h or spatial_map.size(3) != w:
            spatial_map = F.interpolate(spatial_map, size=(h, w), mode='bilinear', align_corners=False)

        return spatial_map


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
        _, _, h, w = x.shape

        # Ensure the input is in float32 for compatibility with cuFFT
        x = x.to(dtype=torch.float32)

        # Adjust dimensions to powers of two
        new_h = 2 ** (h - 1).bit_length()
        new_w = 2 ** (w - 1).bit_length()
        padding_h = (new_h - h) // 2
        padding_w = (new_w - w) // 2

        # Apply padding to make dimensions powers of two
        x_padded = F.pad(x, (padding_w, padding_w, padding_h, padding_h), mode="constant", value=0)

        # Compute FFT
        fft = torch.fft.fft2(x_padded, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.sqrt(fft_shift.real ** 2 + fft_shift.imag ** 2)

        # Remove padding to restore original size
        magnitude = magnitude[:, :, padding_h:padding_h + h, padding_w:padding_w + w]

        # Return normalized spatial map
        return torch.sigmoid(magnitude.mean(dim=1, keepdim=True))



    def _rgb_to_hsv(self, x):
        max_rgb, _ = x.max(dim=1, keepdim=True)
        min_rgb, _ = x.min(dim=1, keepdim=True)
        delta = max_rgb - min_rgb + 1e-6
        saturation = delta / (max_rgb + 1e-6)
        value = max_rgb
        return torch.cat((delta, saturation, value), dim=1)


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


if __name__ == "__main__":
    x = torch.randn(2, 3, 7, 7)  # 입력 크기가 7x7인 텐서
    distortion_attention = DistortionAttention(in_channels=3)
    output = distortion_attention._fourier_analysis(x)
    print("Output Shape:", output.shape)





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
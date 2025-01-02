# ver1
""" 
import torch
import torch.nn as nn

class DistortionAttention(nn.Module):
    def __init__(self, in_channels):
        super(DistortionAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.sigmoid(self.conv(x))
        return x * attention_map

class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(HardNegativeCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return out + x
    
 """

# ver2
import torch
import torch.nn as nn

# 입력 이미지 -> ResNet(Backbone) -> 특징 맵 변환(B x C x H x W)
class DistortionAttention(nn.Module):
    def __init__(self, in_channels):
        super(DistortionAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)    # nn.Conv2d -> 입력채널 C를 1로 축소 (B x 1 x H x W 크기의 attentio map 생성) => 공간 중요도 학습
        self.sigmoid = nn.Sigmoid()     # attention map 값을 0-1 사이로 정규화 --> 각 픽셀이 얼마나 중요한지 나타냄

    def forward(self, x, se_weights=None):
        attention_map = self.sigmoid(self.conv(x))  # Attention Map 생성
        if se_weights is not None:
            # SE 가중치 결합
            combined_weights = attention_map * se_weights.unsqueeze(2).unsqueeze(3)
            return x * combined_weights
        return x * attention_map


class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(HardNegativeCrossAttention, self).__init__()
        # Quert : 1×1 컨볼루션을 사용해 입력에서 저차원 특징 벡터 생성
        # 입력의 공간적 정보를 유지하면서 해상도가 다른 데이터 간 유사도를 비교할 수 있는 표현을 만듭
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # Key : Query와 동일한 방식으로 생성되며, 입력 특징 맵을 다른 해상도에서 추출된 Key와 비교
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # Value : 입력의 전체 정보를 보존하는 고차원 특징을 생성
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # Query와 Key는 해상도 차이로 인한 패턴 변화가 반영되도록 저차원 특징을 추출.
        # Value는 왜곡 정보의 중요한 부분을 유지하여 Attention 결과에 반영



    def forward(self, x):
        b, c, h, w = x.size()

        # Query와 Key는 해상도 차이를 반영하기 위해 설계
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)

        # Query와 Key 간 내적(dot product)을 통해 유사도를 계산.
        # Softmax로 정규화하여 특정 위치 간 유사도를 강조.
        attention = self.softmax(torch.bmm(query, key))

        # Value는 왜곡 정보를 보존하고, Attention Score를 통해 해상도 차이에 민감한 정보를 선택적으로 반영
        value = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        # Residual Connection
        # Attention 결과를 원본 입력에 더하여, 정보 손실을 방지하고 학습 안정성을 증가
        return out + x


# attention_se.py
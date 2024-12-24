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


# attention_se.py
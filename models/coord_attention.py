import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CoordAttention, self).__init__()

        # 🔹 Channel 축소 크기
        mip = max(8, in_channels // reduction)

        # 🔹 Pooling 연산: H 방향 / W 방향 평균값 추출
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]

        # 🔹 Shared 1x1 conv → BN → ReLU
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        # 🔹 개별 attention 생성 (복원)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()

        # 🔹 H 방향 평균 (→ [B, C, H, 1])
        x_h = self.pool_h(x)
        # 🔹 W 방향 평균 후 permute (→ [B, C, W, 1])
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 🔹 Concat along H+W
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 🔹 다시 H / W 방향으로 분할
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [B, C, W, 1] → [B, C, 1, W]

        # 🔹 각 방향에 대한 attention 생성
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 🔹 적용: H, W 방향 attention
        out = identity * a_h * a_w
        return out

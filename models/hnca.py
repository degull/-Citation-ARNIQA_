import torch
import torch.nn as nn
import torch.nn.functional as F

class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels, expansion=6):
        super(HardNegativeCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.expanded_channels = in_channels * expansion

        # Q, K, V 생성
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1)

        # 최종 출력 projection
        self.out_proj = nn.Conv2d(self.expanded_channels, self.expanded_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape  # C = 64

        # Q, K, V 계산
        Q = self.query_conv(x).reshape(B, C, -1).permute(0, 2, 1)       # B x HW x C
        K = self.key_conv(x).reshape(B, C, -1)                          # B x C x HW
        V = self.value_conv(x).reshape(B, self.expanded_channels, -1).permute(0, 2, 1)  # B x HW x 384

        # Attention score
        attention = torch.softmax(torch.bmm(Q, K) / (C ** 0.5), dim=-1)  # B x HW x HW

        # Attention output
        out = torch.bmm(attention, V)  # B x HW x 384
        out = out.permute(0, 2, 1).reshape(B, self.expanded_channels, H, W)  # B x 384 x H x W

        return self.out_proj(out)  # 최종 shape: [B, 384, H, W]

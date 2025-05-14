import torch
import torch.nn as nn

class CPFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPFE, self).__init__()

        # 각 dilation scale의 feature 추출
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=3, padding=3)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=5, padding=5)
        self.branch4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=7, padding=7)

        # concat 후 통합
        self.fuse = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        feat1 = self.branch1(x)  # 1x1 conv
        feat2 = self.branch2(x)  # dilated conv r=3
        feat3 = self.branch3(x)  # dilated conv r=5
        feat4 = self.branch4(x)  # dilated conv r=7

        # Concatenate along channel axis
        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1)

        # Final output conv
        out = self.fuse(fused)
        return out

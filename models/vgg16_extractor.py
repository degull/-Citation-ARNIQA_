import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # 🔹 VGG 단계별 layer 추출
        self.conv1_2 = nn.Sequential(*vgg16[:4])    # Conv1_1, ReLU, Conv1_2, ReLU → 64ch, 224x224
        self.conv2_2 = nn.Sequential(*vgg16[4:9])   # → 128ch, 112x112
        self.conv3_3 = nn.Sequential(*vgg16[9:16])  # → 256ch, 56x56
        self.conv4_3 = nn.Sequential(*vgg16[16:23]) # → 512ch, 28x28
        self.conv5_3 = nn.Sequential(*vgg16[23:30]) # → 512ch, 14x14

    def forward(self, x):
        feat1 = self.conv1_2(x)    # (B, 64, 224, 224)
        feat2 = self.conv2_2(feat1)  # (B, 128, 112, 112)
        feat3 = self.conv3_3(feat2)  # (B, 256, 56, 56)
        feat4 = self.conv4_3(feat3)  # (B, 512, 28, 28)
        feat5 = self.conv5_3(feat4)  # (B, 512, 14, 14)

        return feat1, feat2, feat3, feat4, feat5

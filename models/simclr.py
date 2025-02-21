import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_se import ResNetSE
from models.attention_se import HardNegativeCrossAttention, FeatureLevelAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_se import ResNetSE
from models.attention_se import HardNegativeCrossAttention, FeatureLevelAttention

class SimCLR(nn.Module):
    def __init__(self, embedding_dim=128, temperature=0.5, use_hnca=False):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.use_hnca = use_hnca  # ✅ Hard Negative 사용 여부

        self.backbone = ResNetSE()
        self.feature_attention = FeatureLevelAttention(in_channels=2048, out_channels=2048)

        if self.use_hnca:
            self.hard_negative_attention = HardNegativeCrossAttention(2048)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, inputs_A, inputs_B):
        features_A = self.backbone(inputs_A)
        features_B = self.backbone(inputs_B)

        features_A = self.feature_attention(features_A)
        features_B = self.feature_attention(features_B)

        if self.use_hnca:
            features_A = self.hard_negative_attention(features_A, features_B)
            features_B = self.hard_negative_attention(features_B, features_A)

        features_A = torch.flatten(self.global_avg_pool(features_A), start_dim=1)
        features_B = torch.flatten(self.global_avg_pool(features_B), start_dim=1)

        proj_A = self.projector(features_A)
        proj_B = self.projector(features_B)

        return proj_A, proj_B



    def compute_loss(self, proj_A, proj_B, proj_negatives):
        # ✅ NT-Xent Loss 계산
        positive_similarity = torch.exp(torch.sum(proj_A * proj_B, dim=1) / self.temperature)
        negative_similarity = torch.exp(torch.matmul(proj_A, proj_negatives.T) / self.temperature)
        denom = positive_similarity + torch.sum(negative_similarity, dim=1)
        loss = -torch.mean(torch.log(positive_similarity / denom))
        return loss
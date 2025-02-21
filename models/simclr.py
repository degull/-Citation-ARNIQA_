import sys
import os

# ✅ 프로젝트 루트 경로를 추가 (models 폴더를 찾을 수 있도록 설정)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import torch.nn as nn
from models.resnet_se import ResNetSE
from models.attention_se import FeatureLevelAttention, HardNegativeCrossAttention

class SimCLR(nn.Module):
    def __init__(self, embedding_dim=128, temperature=0.5, use_hnca=False):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.use_hnca = use_hnca

        # ✅ ResNetSE 백본 적용
        self.backbone = ResNetSE(use_hnca=use_hnca)

        # ✅ Feature-Level Attention 적용 (4D 텐서 유지)
        self.feature_attention = FeatureLevelAttention(in_channels=2048, out_channels=2048)

        if self.use_hnca:
            self.hard_negative_attention = HardNegativeCrossAttention(2048)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # ✅ 2D 변환을 여기에 위치시킴

        # ✅ Projection Head
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, inputs_A, inputs_B):
        features_A = self.backbone(inputs_A)  # ✅ 4D 유지 (batch, 2048, 7, 7)
        features_B = self.backbone(inputs_B)  # ✅ 4D 유지 (batch, 2048, 7, 7)

        # ✅ 디버깅 출력
        print(f"features_A shape before feature_attention: {features_A.shape}")  # (batch, 2048, 7, 7)

        features_A = self.feature_attention(features_A)  # ✅ 4D 유지
        features_B = self.feature_attention(features_B)  # ✅ 4D 유지

        # ✅ 디버깅 출력
        print(f"features_A shape after feature_attention: {features_A.shape}")  # (batch, 2048, 7, 7)

        if self.use_hnca:
            features_A = self.hard_negative_attention(features_A, features_B)
            features_B = self.hard_negative_attention(features_B, features_A)

        features_A = self.global_avg_pool(features_A)  # ✅ 이제 2D 변환
        features_B = self.global_avg_pool(features_B)  # ✅ 이제 2D 변환

        features_A = features_A.view(features_A.size(0), -1)  # ✅ (batch, 2048)
        features_B = features_B.view(features_B.size(0), -1)  # ✅ (batch, 2048)

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

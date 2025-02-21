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

import torch
import torch.nn as nn
from models.resnet_se import ResNetSE

class SimCLR(nn.Module):
    def __init__(self, dataset_type="synthetic", embedding_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.dataset_type = dataset_type

        self.backbone = ResNetSE(dataset_type)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, inputs_A, inputs_B, hard_neg=None):
        features_A = self.backbone(inputs_A, hard_neg if self.dataset_type == "synthetic" else None)
        features_B = self.backbone(inputs_B, hard_neg if self.dataset_type == "synthetic" else None)

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
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_se import ResNetSE
from models.attention_se import DistortionAttention, HardNegativeCrossAttention, DistortionClassifier

class SimCLR(nn.Module):
    def __init__(self, embedding_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature

        # Backbone (ResNetSE)
        self.backbone = ResNetSE()

        # Projection Head 정의
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, inputs_A, inputs_B):
        print(f"[Debug] inputs_A shape before ResNet: {inputs_A.shape}")
        print(f"[Debug] inputs_B shape before ResNet: {inputs_B.shape}")

        # ResNet Backbone 통과
        features_A = self.backbone(inputs_A)
        features_B = self.backbone(inputs_B)
        print(f"[Debug] features_A shape after ResNet: {features_A.shape}")
        print(f"[Debug] features_B shape after ResNet: {features_B.shape}")

        # Projection Head
        proj_A = self.projector(features_A)
        proj_B = self.projector(features_B)
        print(f"[Debug] proj_A shape after Projector: {proj_A.shape}")
        print(f"[Debug] proj_B shape after Projector: {proj_B.shape}")

        return proj_A, proj_B



    def compute_loss(self, proj_A, proj_B, proj_negatives):
        # NT-Xent Loss 계산
        positive_similarity = torch.exp(torch.sum(proj_A * proj_B, dim=1) / self.temperature)
        negative_similarity = torch.exp(torch.matmul(proj_A, proj_negatives.T) / self.temperature)
        denom = positive_similarity + torch.sum(negative_similarity, dim=1)
        loss = -torch.mean(torch.log(positive_similarity / denom))
        return loss


if __name__ == "__main__":
    model = SimCLR()
    inputs_A = torch.randn(32, 3, 224, 224)  # Batch of 32 images
    inputs_B = torch.randn(32, 3, 224, 224)

    proj_A, proj_B = model(inputs_A, inputs_B)
    print(f"Final proj_A shape: {proj_A.shape}")
    print(f"Final proj_B shape: {proj_B.shape}")

# SE-weighted logic 추가
""" 
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from models.resnet_se import ResNetSE

class SimCLR(nn.Module):
    def __init__(self, embedding_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature

        # Backbone (ResNetSE)
        self.backbone = ResNetSE()

        # Projection Head 정의
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, inputs_A, inputs_B):
        # ResNet Backbone 통과
        features_A, se_weights_A = self.backbone(inputs_A, return_se_weights=True)
        features_B, se_weights_B = self.backbone(inputs_B, return_se_weights=True)

        # Projection Head
        proj_A = self.projector(features_A)
        proj_B = self.projector(features_B)

        return proj_A, proj_B, se_weights_A, se_weights_B

    def compute_loss(self, proj_A, proj_B, proj_negatives, se_weights_A, se_weights_B):
        # SE 가중치를 proj_A와 proj_B의 차원에 맞게 변환
        se_weights_A = se_weights_A.mean(dim=1).view(-1, 1)  # (batch_size, 1)
        se_weights_B = se_weights_B.mean(dim=1).view(-1, 1)  # (batch_size, 1)

        # SE 가중치를 적용한 투영
        weighted_proj_A = proj_A * se_weights_A
        weighted_proj_B = proj_B * se_weights_B

        # NT-Xent Loss 계산
        positive_similarity = torch.exp(torch.sum(weighted_proj_A * weighted_proj_B, dim=1) / self.temperature)
        negative_similarity = torch.exp(torch.matmul(weighted_proj_A, proj_negatives.T) / self.temperature)
        denom = positive_similarity + torch.sum(negative_similarity, dim=1)
        loss = -torch.mean(torch.log(positive_similarity / denom))
        return loss


if __name__ == "__main__":
    model = SimCLR()
    inputs_A = torch.randn(32, 3, 224, 224)  # Batch of 32 images
    inputs_B = torch.randn(32, 3, 224, 224)

    proj_A, proj_B, se_weights_A, se_weights_B = model(inputs_A, inputs_B)
    print(f"Final proj_A shape: {proj_A.shape}")
    print(f"Final proj_B shape: {proj_B.shape}")
    print(f"SE Weights A: {se_weights_A.shape}, SE Weights B: {se_weights_B.shape}")
 """
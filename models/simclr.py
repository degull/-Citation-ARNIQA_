# ver1
""" 
import torch
import torch.nn as nn
from models.resnet_se import ResNetSE
from models.attention_se import DistortionAttention


class SimCLR(nn.Module):
    def __init__(self, encoder_params=None, embedding_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature

        # Initialize ResNet-SE with encoder parameters
        self.backbone = ResNetSE(encoder_params)

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )

        # Attention mechanism
        self.attention = DistortionAttention(2048)

    def forward(self, inputs_A, inputs_B=None):
        # Reshape if inputs_A is 5D
        if inputs_A.dim() == 5:
            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
        features_A = self.backbone(inputs_A)
        features_A = features_A.mean([2, 3])  # Global Average Pooling
        proj_A = self.projector(features_A)

        if inputs_B is not None:
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])
            features_B = self.backbone(inputs_B)
            features_B = features_B.mean([2, 3])  # Global Average Pooling
            proj_B = self.projector(features_B)
            return proj_A, proj_B

        return proj_A


    def compute_loss(self, proj_A, proj_B, proj_negatives):
        # Positive similarity
        positive_similarity = torch.exp(torch.sum(proj_A * proj_B, dim=1) / self.temperature)

        # Negative similarity
        negative_similarity = torch.exp(torch.matmul(proj_A, proj_negatives.T) / self.temperature)

        # Normalization term
        denom = positive_similarity + torch.sum(negative_similarity, dim=1)

        # NT-Xent loss
        loss = -torch.mean(torch.log(positive_similarity / denom))
        return loss
    
 """

# ver2

# simclr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_se import ResNetSE
from models.attention_se import DistortionAttention

class SimCLR(nn.Module):
    def __init__(self, encoder_params=None, embedding_dim=128, temperature=0.1):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.backbone = ResNetSE(encoder_params)
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )
        self.attention = DistortionAttention(2048)

    def forward(self, inputs_A, inputs_B=None):
        if inputs_A.dim() == 5:
            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
        features_A = self.backbone(inputs_A)
        se_weights = torch.clamp(features_A.mean(dim=[2, 3]), 0, 1)
        features_A = self.attention(features_A, se_weights)
        features_A = features_A.mean([2, 3])
        proj_A = self.projector(features_A)

        if inputs_B is not None:
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])
            features_B = self.backbone(inputs_B)
            se_weights_B = torch.clamp(features_B.mean(dim=[2, 3]), 0, 1)
            features_B = self.attention(features_B, se_weights_B)
            features_B = features_B.mean([2, 3])
            proj_B = self.projector(features_B)
            return proj_A, proj_B

        return proj_A

    def compute_loss(self, proj_A, proj_B, proj_negatives, se_weights):
        se_weights = se_weights.mean(dim=1, keepdim=False).unsqueeze(1)
        positive_similarity = torch.exp((torch.sum(proj_A * proj_B, dim=1) * se_weights.squeeze()) / self.temperature)
        negative_similarity = torch.exp((torch.matmul(proj_A, proj_negatives.T)) / self.temperature)
        denom = positive_similarity + torch.sum(negative_similarity, dim=1)
        loss = -torch.mean(torch.log(positive_similarity / denom))
        return loss

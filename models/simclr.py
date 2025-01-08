import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_se import ResNetSE
from models.attention_se import DistortionAttention, HardNegativeCrossAttention

class SimCLR(nn.Module):
    def __init__(self, embedding_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature

        # Initialize ResNet-SE
        self.backbone = ResNetSE()

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )

        # Attention mechanisms
        self.distortion_attention = DistortionAttention(2048)
        self.hard_negative_attention = HardNegativeCrossAttention(2048)

    def forward(self, inputs_A, inputs_B=None):
        # Reshape inputs if 5D
        if inputs_A.dim() == 5:
            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
        if inputs_B is not None and inputs_B.dim() == 5:
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

        # Process inputs_A
        features_A = self.backbone(inputs_A)
        se_weights_A = features_A.mean(dim=[2, 3])  # SE weights for inputs_A
        features_A = self.distortion_attention(features_A, se_weights=se_weights_A)

        if inputs_B is not None:
            # Process inputs_B
            features_B = self.backbone(inputs_B)
            se_weights_B = features_B.mean(dim=[2, 3])  # SE weights for inputs_B
            features_B = self.distortion_attention(features_B, se_weights=se_weights_B)

            # Apply hard negative cross-attention
            features_A = self.hard_negative_attention(features_A, features_B)

            # Global Average Pooling
            features_A = features_A.mean([2, 3])
            features_B = features_B.mean([2, 3])

            # Projected outputs
            proj_A = self.projector(features_A)
            proj_B = self.projector(features_B)
            return proj_A, proj_B

        # If inputs_B is None, process only inputs_A
        features_A = features_A.mean([2, 3])
        proj_A = self.projector(features_A)
        return proj_A, None

    def compute_loss(self, proj_A, proj_B, proj_negatives, se_weights):
        se_weights = se_weights / se_weights.sum(dim=1, keepdim=True)  # Normalize SE weights
        positive_similarity = torch.exp(torch.sum(proj_A * proj_B, dim=1) / self.temperature)
        negative_similarity = torch.exp(torch.matmul(proj_A, proj_negatives.T) / self.temperature)
        denom = positive_similarity + torch.sum(negative_similarity, dim=1)
        loss = -torch.mean(torch.log(positive_similarity / denom))
        return loss

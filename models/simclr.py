import torch
import torch.nn as nn
from models.resnet_se import ResNetSE
from models.attention_se import DistortionAttention, HardNegativeCrossAttention

class SimCLR(nn.Module):
    def __init__(self, encoder_params=None, embedding_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature  # Add temperature parameter
        
        # Initialize ResNet-SE with encoder parameters
        self.backbone = ResNetSE(encoder_params)  # ResNet-50 with SE
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )
        
        # Attention mechanism
        self.attention = DistortionAttention(2048)

    def forward(self, inputs_A, inputs_B=None):
        # Feature extraction
        features_A = self.backbone(inputs_A)
        features_A = features_A.mean([2, 3])  # Global Average Pooling
        proj_A = self.projector(features_A)

        if inputs_B is not None:
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


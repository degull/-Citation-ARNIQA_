

import torch
import torch.nn as nn
import torch.nn.functional as F

# 논문
class FeatureLevelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureLevelAttention, self).__init__()
        self.out_channels = out_channels

        # 입력 채널 변환
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Channel-wise Attention
        self.channel_fc1 = nn.Linear(out_channels, out_channels // 4)
        self.channel_fc2 = nn.Linear(out_channels // 4, out_channels)

        # Spatial Attention
        self.spatial_conv1 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=(1, 9), padding=(0, 4))
        self.spatial_bn1 = nn.BatchNorm2d(out_channels // 2)
        self.spatial_conv2 = nn.Conv2d(out_channels // 2, 1, kernel_size=(9, 1), padding=(4, 0))
        self.spatial_bn2 = nn.BatchNorm2d(1)

        # Feature Update
        self.feature_update_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.input_conv(x)

        # Channel-wise Attention
        channel_weights = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        channel_weights = F.relu(self.channel_fc1(channel_weights))
        channel_weights = torch.sigmoid(self.channel_fc2(channel_weights))
        channel_weights = channel_weights.view(b, c, 1, 1)
        channel_attention = x * channel_weights

        # Spatial Attention
        spatial_weights = F.relu(self.spatial_bn1(self.spatial_conv1(channel_attention)))
        spatial_weights = F.relu(self.spatial_bn2(self.spatial_conv2(spatial_weights)))
        spatial_weights = torch.sigmoid(spatial_weights)
        spatial_attention = channel_attention * spatial_weights

        # Feature Update & Residual Connection
        out = self.feature_update_conv(spatial_attention)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = out + x  # Residual Connection

        return out
    

# Feature-Level Attention
""" class FeatureLevelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.3):
        super(FeatureLevelAttention, self).__init__()

        self.num_heads = num_heads
        self.out_channels = out_channels

        # ✅ 입력 채널 변환 (해상도 유지)
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.head_dim = max(out_channels // num_heads, 1)
        self.scale = self.head_dim ** -0.5  

        # Query, Key, Value 생성
        self.qkv_conv = nn.Conv2d(out_channels, out_channels * 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.output_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Feature Update
        self.feature_update_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.size()  # ✅ 원본 입력 크기 가져오기

        # ✅ 입력 채널 변환
        x = self.input_conv(x)

        # ✅ Query, Key, Value 생성
        qkv = self.qkv_conv(x).reshape(b, self.num_heads, 3, self.head_dim, h, w)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)  

        # ✅ Gumbel Softmax 적용
        attention = F.gumbel_softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, tau=0.8, hard=True, dim=-1)
        A_final = torch.matmul(attention, v)  
        A_final = A_final.reshape(b, self.out_channels, h, w)  # ✅ 원본 크기 유지

        # ✅ **출력 크기를 원본 크기 (`h, w`)로 유지**
        A_final = F.interpolate(A_final, size=(h, w), mode="bilinear", align_corners=False)

        # 최종 Projection 적용
        A_final = self.output_proj(A_final)
        A_final = self.feature_update_conv(A_final)
        A_final = self.batch_norm(A_final)
        A_final = self.activation(A_final)
        A_final = self.dropout(A_final)

        # Residual Connection 적용
        out = A_final + x  

        return out
 """


class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8, reduction=16, alpha=0.5):
        super(HardNegativeCrossAttention, self).__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads  
        self.alpha = alpha  

        # ✅ 속성 정보 학습
        self.attr_conv1 = nn.Conv2d(in_dim, in_dim // reduction, kernel_size=1)
        self.attr_relu1 = nn.ReLU(inplace=True)
        self.attr_conv2 = nn.Conv2d(in_dim // reduction, in_dim, kernel_size=1)
        self.attr_relu2 = nn.ReLU(inplace=True)

        # ✅ 텍스처 정보 학습
        self.texture_conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.texture_relu = nn.ReLU(inplace=True)
        self.texture_conv2 = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=3, padding=1)

        # ✅ 속성 정보 Concat 후 1x1 Conv 적용
        self.attr_dim_adjust = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1)

        # ✅ Multi-head Cross Attention에 필요한 Q, K, V 생성
        self.head_dim = in_dim // num_heads
        self.query_attr = nn.Linear(in_dim, in_dim)  
        self.key_attr = nn.Linear(in_dim, in_dim)  
        self.value_attr = nn.Linear(in_dim, in_dim)  

        self.query_tex = nn.Linear(in_dim, in_dim)  
        self.key_tex = nn.Linear(in_dim, in_dim)  
        self.value_tex = nn.Linear(in_dim, in_dim)  

        # ✅ 최종 Projection Layer
        self.output_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        # ✅ BatchNorm2d 적용
        self.norm = nn.BatchNorm2d(in_dim)

    def forward(self, x, hard_neg):
        b, c, h, w = x.shape  

        # ✅ 속성 정보 학습
        attr_feat = self.attr_relu1(self.attr_conv1(x))
        attr_feat = self.attr_relu2(self.attr_conv2(attr_feat))
        attr_feat = torch.cat([attr_feat, x], dim=1)  
        attr_feat = self.attr_dim_adjust(attr_feat)  

        # ✅ 텍스처 정보 학습
        texture_feat = self.texture_relu(self.texture_conv1(hard_neg))
        texture_feat = self.texture_conv2(texture_feat)
        texture_feat = texture_feat * attr_feat  # Element-wise Multiplication

        # ✅ Query, Key, Value 생성 (속성 & 텍스처 각각 따로 생성)
        attr_feat = attr_feat.view(b, c, -1).permute(0, 2, 1)  # (B, HW, C)
        texture_feat = texture_feat.view(b, c, -1).permute(0, 2, 1)  # (B, HW, C)

        Q_attr = self.query_attr(attr_feat)  
        K_attr = self.key_attr(attr_feat)  
        V_attr = self.value_attr(attr_feat)  

        Q_tex = self.query_tex(texture_feat)  
        K_tex = self.key_tex(texture_feat)  
        V_tex = self.value_tex(texture_feat)  

        # ✅ Multi-Head Cross Attention 적용
        attention_scores_attr = torch.softmax(Q_attr @ K_attr.transpose(-2, -1) / (self.in_dim ** 0.5), dim=-1)
        attn_output_attr = attention_scores_attr @ V_attr  

        attention_scores_tex = torch.softmax(Q_tex @ K_tex.transpose(-2, -1) / (self.in_dim ** 0.5), dim=-1)
        attn_output_tex = attention_scores_tex @ V_tex  

        # ✅ 속성과 텍스처 정보를 가중치 조합
        attn_output = self.alpha * attn_output_attr + (1 - self.alpha) * attn_output_tex

        # ✅ 원래 크기로 복구
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
        attn_output = self.output_proj(attn_output)

        # ✅ BatchNorm 적용 후 Residual 연결
        attn_output = self.norm(attn_output + x)  

        return attn_output

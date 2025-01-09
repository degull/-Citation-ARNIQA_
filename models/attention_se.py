""" import torch
import torch.nn as nn
import torch.nn.functional as F

class DistortionAttention(nn.Module):
    def __init__(self, in_channels):
        super(DistortionAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # Sobel Filters for Spatial Map
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # Expand Sobel filters to match input channels
        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)

    def forward(self, x, se_weights=None, spatial_map=None, channel_map=None):
        b, c, h, w = x.size()
        
        # Generate query, key, value tensors
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)
        value = self.value_conv(x).view(b, -1, h * w)

        # Attention computation
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)

        # Spatial Map generation if not provided
        if spatial_map is None:
            spatial_map = self._generate_spatial_map(x)

        # Channel Map generation if not provided
        if channel_map is None:
            channel_map = self._generate_channel_map(x)

        # Apply SE weights, spatial map, and channel map if provided
        if se_weights is not None:
            out *= se_weights.view(b, c, 1, 1)
        if spatial_map is not None:
            out *= spatial_map
        if channel_map is not None:
            out *= channel_map.view(b, c, 1, 1)

        return out + x  # Residual connection

    def _generate_spatial_map(self, x):
        # Compute gradients using Sobel filters
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Pool gradients and apply sigmoid for normalization
        pooled_gradient = torch.mean(gradient_magnitude, dim=1, keepdim=True)
        return torch.sigmoid(pooled_gradient)

    def _generate_channel_map(self, x):
        # Compute channel-wise global statistics
        channel_mean = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        return torch.sigmoid(channel_mean)  # Normalize to [0, 1]



class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(HardNegativeCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_attr, x_texture):
        b, c, h, w = x_attr.size()

        # Generate query, key, value tensors
        query = self.query_conv(x_attr).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x_texture).view(b, -1, h * w)
        value = self.value_conv(x_texture).view(b, -1, h * w).permute(0, 2, 1)

        # Attention computation
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(attention, value).permute(0, 2, 1).contiguous().view(b, c, h, w)

        return out + x_attr  # Residual connection



# Example usage
if __name__ == "__main__":
    # Random input tensor (batch_size=2, channels=2048, height=7, width=7)
    input_tensor = torch.randn(2, 2048, 7, 7)

    # Initialize DistortionAttention
    distortion_attention = DistortionAttention(in_channels=2048)

    # Forward pass with automatic generation of spatial_map and channel_map
    output = distortion_attention(input_tensor)
    print("DistortionAttention Output shape:", output.shape)


 """

# 시각화
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistortionAttention(nn.Module):
    def __init__(self, in_channels):
        super(DistortionAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # Sobel Filters for Spatial Map
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # Expand Sobel filters to match input channels
        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)

    def forward(self, x, se_weights=None, spatial_map=None, channel_map=None):
        b, c, h, w = x.size()
        
        # Generate query, key, value tensors
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)
        value = self.value_conv(x).view(b, -1, h * w)

        # Attention computation
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)

        # Spatial Map generation if not provided
        if spatial_map is None:
            spatial_map = self._generate_spatial_map(x)

        # Channel Map generation if not provided
        if channel_map is None:
            channel_map = self._generate_channel_map(x)

        # Apply SE weights, spatial map, and channel map if provided
        if se_weights is not None:
            out *= se_weights.view(b, c, 1, 1)
        if spatial_map is not None:
            out *= spatial_map
        if channel_map is not None:
            out *= channel_map.view(b, c, 1, 1)

        return out + x  # Residual connection

    def _generate_spatial_map(self, x):
        # Compute gradients using Sobel filters
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Pool gradients and apply sigmoid for normalization
        pooled_gradient = torch.mean(gradient_magnitude, dim=1, keepdim=True)
        return torch.sigmoid(pooled_gradient)

    def _generate_channel_map(self, x):
        # Compute channel-wise global statistics
        channel_mean = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        return torch.sigmoid(channel_mean)  # Normalize to [0, 1]



class HardNegativeCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(HardNegativeCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_attr, x_texture):
        b, c, h, w = x_attr.size()

        # Generate query, key, value tensors
        query = self.query_conv(x_attr).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x_texture).view(b, -1, h * w)
        value = self.value_conv(x_texture).view(b, -1, h * w).permute(0, 2, 1)

        # Attention computation
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(attention, value).permute(0, 2, 1).contiguous().view(b, c, h, w)

        return out + x_attr  # Residual connection
# ver1
""" 
import torch
import torch.nn as nn

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradients(self, grad):
        self.gradients = grad

    def forward(self, x):
        for name, module in self.model.named_children():
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradients)
                target_activation = x
        return target_activation

    def __call__(self, x):
        target_activation = self.forward(x)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * target_activation).sum(dim=1, keepdim=True)
        return cam

 """

# ver2
import torch
import matplotlib.pyplot as plt

def visualize_se_attention(features, se_weights, attention_map, title="SE Attention Visualization"):
    se_attention_map = features.mean(dim=1, keepdim=True) * se_weights.unsqueeze(2).unsqueeze(3) * attention_map
    se_attention_map = se_attention_map.squeeze().cpu().numpy()
    se_attention_map -= se_attention_map.min()
    se_attention_map /= se_attention_map.max()

    plt.figure(figsize=(10, 8))
    plt.imshow(se_attention_map, cmap='jet')
    plt.title(title)
    plt.colorbar()
    plt.show()


# visualize_gradcam.py
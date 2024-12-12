import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activations = None

        # Register forward hook
        self.hook_handle = self.target_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """Hook function to save activations and gradients"""
        self.activations = output
        output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        """Save gradients"""
        self.gradient = grad

    def generate_cam(self, input_image, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        input_image = input_image.unsqueeze(0)  # Add batch dimension
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
        
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        gradients = self.gradient[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        # Compute weights from gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, None, None] * activations, axis=0)

        # Apply ReLU to the CAM
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.size(2), input_image.size(3)))
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam

    def remove_hook(self):
        """Remove hook after Grad-CAM computation"""
        self.hook_handle.remove()


def visualize_gradcam(model, input_image, target_layer, class_idx=None):
    """Visualize Grad-CAM"""
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_image, class_idx)
    gradcam.remove_hook()

    # Apply color map to the Grad-CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Convert input image from tensor to numpy array
    input_image = input_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    input_image = np.uint8(255 * input_image)
    
    # Superimpose heatmap on original image
    result = heatmap + np.float32(input_image) / 255.0
    result = result / np.max(result)

    # Display the result
    plt.imshow(result)
    plt.axis('off')
    plt.show()

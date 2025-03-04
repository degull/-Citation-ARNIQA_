# saliency 1
""" 
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image

# ✅ VGG-16 Hook 기반 Feature Extractor
vgg_conv3_3 = vgg_conv4_3 = vgg_conv5_3 = None

def conv_3_3_hook(module, input, output):
    global vgg_conv3_3
    vgg_conv3_3 = output

def conv_4_3_hook(module, input, output):
    global vgg_conv4_3
    vgg_conv4_3 = output

def conv_5_3_hook(module, input, output):
    global vgg_conv5_3
    vgg_conv5_3 = output

class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg16 = vgg16
        self.vgg16[16].register_forward_hook(conv_3_3_hook)
        self.vgg16[23].register_forward_hook(conv_4_3_hook)
        self.vgg16[30].register_forward_hook(conv_5_3_hook)
    
    def forward(self, x):
        return self.vgg16(x)

# ✅ CPFE 모듈
class CPFE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPFE, self).__init__()
        self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_r3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv3x3_r5 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        self.conv3x3_r7 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=7, dilation=7)
        self.fuse_conv = torch.nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
    
    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_r3(x)
        feat3 = self.conv3x3_r5(x)
        feat4 = self.conv3x3_r7(x)
        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.fuse_conv(fused)

# ✅ SaliencyDetectionModel 개선 (업샘플링 적용)
class SaliencyDetectionModel(torch.nn.Module):
    def __init__(self):
        super(SaliencyDetectionModel, self).__init__()
        self.feature_extractor = VGG16FeatureExtractor()
        self.cpfe3 = CPFE(256, 64)
        self.cpfe4 = CPFE(512, 64)
        self.cpfe5 = CPFE(512, 64)
        self.final_conv = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        self.feature_extractor(x)
        feat3 = self.cpfe3(vgg_conv3_3)  # 원본 크기 (28x28)
        feat4 = self.cpfe4(vgg_conv4_3)
        feat5 = self.cpfe5(vgg_conv5_3)
        
        # ✅ 업샘플링으로 크기 맞추기
        feat4 = F.interpolate(feat4, size=feat3.shape[2:], mode='bilinear', align_corners=True)
        feat5 = F.interpolate(feat5, size=feat3.shape[2:], mode='bilinear', align_corners=True)
        
        high_feat = feat3 + feat4 + feat5
        return self.final_conv(high_feat)

# ✅ Grad-CAM 클래스
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.mean().backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Backward hook was not called. Check the target layer.")
        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()
        pooled_gradients = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(pooled_gradients * activations, axis=1)
        cam = np.maximum(cam, 0)
        cam = cam[0]
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# ✅ 이미지 로드 및 변환
image_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KONIQ10K/1024x768/5076506.jpg"
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), np.array(image)

# ✅ 모델 실행
model = SaliencyDetectionModel()
model.eval()
input_tensor, original_image = preprocess_image(image_path)
gradcam = GradCAM(model, model.cpfe5)  # ✅ CPFE5를 타겟 계층으로 지정
cam = gradcam.generate_cam(input_tensor)

# ✅ 히트맵 시각화
def overlay_heatmap(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    return overlay

heatmap_img = overlay_heatmap(original_image, cam)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.resize(original_image, (224, 224)))
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(heatmap_img)
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.show() """


# 시각화1
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        print(f"Debug: Model Output Shape = {output.shape}")  # 🔥 출력 확인

        # ✅ 출력 차원 확인 후 처리
        if output.dim() == 1:  
            output = output.unsqueeze(1)  # (batch_size, 1)로 변경
        elif output.dim() == 2 and output.shape[1] == 1:
            output = output.squeeze(dim=1)  # 🔥 차원 축소

        # ✅ target_class 설정 (배치 기반 처리)
        if target_class is None:
            target_class = output.argmax(dim=-1)  # 🔥 안전한 방식으로 변경
        if isinstance(target_class, torch.Tensor):
            target_class = target_class[0].item()  # 🔥 첫 번째 샘플만 선택

        loss = output[target_class].sum()  # 🔥 차원 문제 해결
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # ✅ NaN 값 제거
        cam = np.nan_to_num(cam)

        # ✅ 정규화 후 uint8 변환
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
        cam = np.clip(cam * 255, 0, 255).astype(np.uint8)  # 🔥 OpenCV 호환

        # ✅ OpenCV에서 지원하는 채널 크기 확인
        if len(cam.shape) == 3 and cam.shape[-1] != 3:
            cam = np.mean(cam, axis=-1)  # 3채널이 아닌 경우 평균 처리
        elif len(cam.shape) == 2:  # (H, W) 형태라면 변환 없이 그대로 사용
            pass
        else:
            raise ValueError(f"Unexpected cam shape: {cam.shape}")

        cam = cv2.resize(cam, (224, 224))

        return cam

    def visualize_cam(self, image, cam, save_path="gradcam_result.png"):
        print(f"Debug: cam shape = {cam.shape}, dtype = {cam.dtype}")  # 🔥 cam 정보 확인

        # ✅ OpenCV 컬러맵 적용 전 uint8 확인
        if cam.dtype != np.uint8:
            cam = np.clip(cam, 0, 255).astype(np.uint8)

        # ✅ 컬러맵 적용
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        superimposed = np.uint8(0.6 * heatmap + 0.4 * image)
        plt.figure(figsize=(6, 6))
        plt.imshow(superimposed)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()



# 시각화2
""" 
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output)

        loss = output[:, target_class]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
        cam = cv2.resize(cam, (224, 224))

        return cam

    def visualize_cam(self, image, cam, save_path="gradcam_result.png"):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        superimposed = np.uint8(0.6 * heatmap + 0.4 * image)
        plt.figure(figsize=(6, 6))
        plt.imshow(superimposed)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
 """
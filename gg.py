import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"cuDNN 버전: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '사용 불가'}")
print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
print(f"GPU 개수: {torch.cuda.device_count()}")

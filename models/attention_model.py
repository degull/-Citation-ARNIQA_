import torch
import torch.nn as nn
from attention_se import EnhancedDistortionDetectionModel  # 모델 불러오기

# ✅ 체크포인트 경로 설정
checkpoint_path = "E:/ARNIQA - SE - mix/ARNIQA/experiments/my_experiment/regressors/1e-4/kadid/epoch_27_srocc_0.938.pth"

# ✅ 체크포인트 로드
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# ✅ 현재 모델 초기화
model = EnhancedDistortionDetectionModel()

# ✅ 현재 모델과 체크포인트의 키 비교
model_state_keys = set(model.state_dict().keys())  # 현재 모델의 키
checkpoint_keys = set(checkpoint.keys())  # 체크포인트의 키

# ✅ 키 개수 출력
print("🚀 모델의 state_dict 키 개수:", len(model_state_keys))
print("🚀 체크포인트의 state_dict 키 개수:", len(checkpoint_keys))

# ✅ 모델에 없고, 체크포인트에만 있는 키 출력
missing_in_model = checkpoint_keys - model_state_keys
print("\n🔴 모델에 없고, 체크포인트에만 있는 키들:")
print(missing_in_model if missing_in_model else "✅ 없음")

# ✅ 체크포인트에 없고, 모델에만 있는 키 출력
missing_in_checkpoint = model_state_keys - checkpoint_keys
print("\n🔴 체크포인트에 없고, 모델에만 있는 키들:")
print(missing_in_checkpoint if missing_in_checkpoint else "✅ 없음")

# ✅ 키 크기 차이 확인
print("\n🔹 🔥 **파라미터 크기 차이 확인** 🔥 🔹")
for key in model.state_dict().keys():
    if key in checkpoint:
        model_shape = model.state_dict()[key].shape
        checkpoint_shape = checkpoint[key].shape
        if model_shape != checkpoint_shape:
            print(f"⚠️ 키 `{key}` → 모델 크기: {model_shape}, 체크포인트 크기: {checkpoint_shape}")

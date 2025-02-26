import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size

        # ✅ CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "kadid10k.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"KADID10K CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)

        # ✅ 이미지 경로 설정
        self.image_paths = [os.path.join(self.root, "images", img) for img in scores_csv["dist_img"]]
        self.mos = scores_csv["dmos"].values  # ✅ MOS 점수 (0~1 정규화됨)

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __getitem__(self, index: int):
        img_A = Image.open(self.image_paths[index]).convert("RGB")  
        img_A = self.transform(img_A)

        return {
            "img_A": img_A,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K"

    dataset = KADID10KDataset(root=dataset_path, phase="train", crop_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
    print(f"Sample MOS Scores: {sample_batch['mos']}")


""" 
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ✅ 동일한 정규화 적용
common_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ MOS 값 정규화 함수
def normalize_mos(mos_values):
    mos_values = np.array(mos_values).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(mos_values).flatten()

class KADID10KDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.root = str(root)

        # ✅ CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "kadid10k.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"KADID10K CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)

        # ✅ 이미지 경로 설정
        self.image_paths = [os.path.join(self.root, "images", img) for img in scores_csv["dist_img"]]
        self.mos = normalize_mos(scores_csv["dmos"].values)  # ✅ MOS 값 정규화

    def __getitem__(self, index: int):
        img_A = Image.open(self.image_paths[index]).convert("RGB")
        img_A = common_transforms(img_A)  # ✅ 동일한 정규화 적용

        return {
            "img_A": img_A,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K"

    dataset = KADID10KDataset(root=dataset_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"✅ KADID10K 데이터셋 크기: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"🔹 샘플 이미지 크기: {sample_batch['img_A'].shape}")
    print(f"🔹 샘플 MOS 점수: {sample_batch['mos']}")
    print(f"🔹 MOS 범위: {sample_batch['mos'].min().item()} ~ {sample_batch['mos'].max().item()}")

    # ✅ MOS 값이 0~1 범위로 정규화되었는지 확인
    assert 0 <= sample_batch["mos"].min().item() <= 1, "⚠️ MOS 값 정규화 필요!"
    assert 0 <= sample_batch["mos"].max().item() <= 1, "⚠️ MOS 값 정규화 필요!"

    print("🚀 **KADID10K 데이터셋 테스트 완료!** 🚀")
 """
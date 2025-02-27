import os
import pandas as pd
import torch
import numpy as np
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
        self.mos = scores_csv["dmos"].astype(float).values  # ✅ MOS 값을 float로 변환

        # ✅ MOS 값 정규화 (0~1 범위)
        mos_min = np.min(self.mos)
        mos_max = np.max(self.mos)
        if mos_max - mos_min == 0:
            raise ValueError("[Error] MOS 값의 최소값과 최대값이 동일하여 정규화할 수 없습니다.")

        self.mos = (self.mos - mos_min) / (mos_max - mos_min)
        print(f"[Check] MOS 최소값: {np.min(self.mos)}, 최대값: {np.max(self.mos)}")

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

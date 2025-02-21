import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SPAQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        """
        SPAQ는 Authentic 데이터셋이므로 Hard Negative를 적용하지 않고 원본 이미지만 사용합니다.
        """
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size

        # ✅ CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "Annotations", "MOS and Image attribute scores.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)

        # ✅ MOS 값 확인 및 정규화 적용
        self.mos = scores_csv["MOS"].astype(float)
        print(f"[Debug] MOS 최소값: {self.mos.min()}, MOS 최대값: {self.mos.max()}")

        # ✅ MOS 값이 10 이상이면 0~1 범위로 정규화
        if self.mos.max() > 10:
            self.mos = self.mos / 100.0  # ✅ 0~1 범위로 정규화

        self.images = scores_csv["Image name"].values
        self.image_paths = [
            os.path.join(self.root, "TestImage", img) for img in self.images
        ]

        print(f"[Debug] Phase: {self.phase}")
        print(f"[Debug] Total Records: {len(self.image_paths)}")

        # ✅ 기본 이미지 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index: int):
        """ 데이터셋에서 index 번째 샘플을 반환 """
        try:
            img = Image.open(self.image_paths[index]).convert("RGB")  # ✅ 원본 이미지
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        img_transformed = self.transform(img)

        return {
            "img_A": img_transformed,  # ✅ 원본 이미지
            "img_B": img_transformed,  # ✅ Authentic 데이터셋이므로 img_B도 동일한 원본 이미지
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.images)


# ✅ SPAQDataset 테스트 (KONIQ10K 방식 적용)
if __name__ == "__main__":
    """
    ✅ SPAQ는 Authentic 데이터셋이므로 Hard Negative 없이 원본 이미지만 사용.
    """
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ"

    authentic_dataset = SPAQDataset(root=dataset_path, phase="training", crop_size=224)
    authentic_dataloader = DataLoader(authentic_dataset, batch_size=4, shuffle=True)

    print(f"Authentic Dataset size: {len(authentic_dataset)}")

    # ✅ Authentic 데이터셋의 첫 번째 배치 확인
    sample_batch_authentic = next(iter(authentic_dataloader))
    print(f"\n[Authentic] 샘플 확인:")
    for i in range(4):  
        print(f"  Sample {i+1} - MOS: {sample_batch_authentic['mos'][i]}")

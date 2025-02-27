import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class SPAQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        """
        ✅ DistortionDetectionModel에 적합하도록 데이터셋 수정
        - `img_A`(왜곡된 이미지)만 반환
        - `mos`(Mean Opinion Score) 점수 반환 (0~1 정규화)
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

        # ✅ MOS 값 로드
        self.mos = scores_csv["MOS"].astype(float)

        # ✅ MOS 값 검사 및 정리 (NaN/Inf 처리)
        print(f"[Check] 총 MOS 값 개수: {len(self.mos)}")
        print(f"[Check] NaN 개수: {np.isnan(self.mos).sum()}, Inf 개수: {np.isinf(self.mos).sum()}")

        if np.isnan(self.mos).sum() > 0 or np.isinf(self.mos).sum() > 0:
            self.mos = np.nan_to_num(self.mos, nan=0.5, posinf=1.0, neginf=0.0)  # NaN을 0.5로 대체

        # ✅ MOS 값 정규화 (0~1 범위)
        mos_min = np.min(self.mos)
        mos_max = np.max(self.mos)
        if mos_max - mos_min == 0:
            raise ValueError("[Error] MOS 값의 최소값과 최대값이 동일하여 정규화할 수 없습니다.")

        self.mos = (self.mos - mos_min) / (mos_max - mos_min)
        print(f"[Check] MOS 최소값: {np.min(self.mos)}, 최대값: {np.max(self.mos)}")

        # ✅ 이미지 파일 경로 설정
        self.images = scores_csv["Image name"].values
        self.image_paths = [os.path.join(self.root, "TestImage", img) for img in self.images]

        print(f"[Debug] Phase: {self.phase}")
        print(f"[Debug] Total Records: {len(self.image_paths)}")

        # ✅ 기본 이미지 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index: int):
        """ ✅ `img_A`(왜곡된 이미지)와 `mos`(Mean Opinion Score)만 반환 """
        try:
            img_A = Image.open(self.image_paths[index]).convert("RGB")  # ✅ 원본 이미지
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        img_A_transformed = self.transform(img_A)

        return {
            "img_A": img_A_transformed,  # ✅ 원본 이미지
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.images)


# ✅ SPAQDataset 테스트 코드
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

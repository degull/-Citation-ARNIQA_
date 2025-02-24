import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class KONIQ10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "all", crop_size: int = 224):
        super().__init__()
        self.root = root
        self.phase = phase.lower()
        self.crop_size = crop_size

        # ✅ CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "meta_info_KonIQ10kDataset.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)

        self.images = scores_csv["image_name"].values
        self.mos = scores_csv["MOS"].values.astype(np.float32)  # MOS 값을 float으로 변환
        self.sets = scores_csv["set"].values

        # ✅ MOS 값 검사 및 정리
        print(f"[Check] 총 MOS 값 개수: {len(self.mos)}")
        print(f"[Check] NaN 개수: {np.isnan(self.mos).sum()}, Inf 개수: {np.isinf(self.mos).sum()}")

        if np.isnan(self.mos).sum() > 0 or np.isinf(self.mos).sum() > 0:
            self.mos = np.nan_to_num(self.mos, nan=0.5, posinf=1.0, neginf=0.0)  # NaN을 0.5로 대체

        # ✅ MOS 값 정규화 (0~1 범위)
        self.mos = (self.mos - np.min(self.mos)) / (np.max(self.mos) - np.min(self.mos))
        print(f"[Check] MOS 최소값: {np.min(self.mos)}, 최대값: {np.max(self.mos)}")

        # ✅ 데이터 필터링
        if self.phase != "all":
            indices = [i for i, s in enumerate(self.sets) if s.strip().lower() == self.phase]
            self.images = self.images[indices]
            self.mos = self.mos[indices]

        # ✅ 이미지 경로 생성
        self.image_paths = [os.path.join(self.root, "1024x768", img) for img in self.images]

        print(f"[Debug] Phase: {self.phase}")
        print(f"[Debug] Total Records: {len(self.image_paths)}")

        # ✅ 기본 이미지 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mos = self.mos[index]

        try:
            img_A = Image.open(image_path).convert("RGB")  # ✅ 원본 이미지 로드
        except Exception as e:
            print(f"[Error] 이미지 로드 실패: {image_path}: {e}")
            return None

        img_A_transformed = self.transform(img_A)

        return {
            "img_A": img_A_transformed,
            "mos": torch.tensor(mos, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)

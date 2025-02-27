import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class KONIQ10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "all", crop_size: int = 224):
        super().__init__()
        self.root = root
        self.phase = phase.strip().lower()  # 🔹 공백 제거 후 소문자로 변환
        self.crop_size = crop_size

        # ✅ CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "meta_info_KonIQ10kDataset.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)

        self.images = scores_csv["image_name"].values
        self.mos = scores_csv["MOS"].astype(float).values  # ✅ MOS 값을 float으로 변환

        # ✅ set 컬럼 변환 (소문자로 변환 후 'training' → 'train', 'validation' → 'val')
        self.sets = scores_csv["set"].astype(str).str.strip().str.lower().replace({
            "training": "train",
            "validation": "val"
        })

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

        # ✅ 데이터 필터링 (train/val/test 선택)
        if self.phase != "all":
            indices = [i for i, s in enumerate(self.sets) if s == self.phase]  # ✅ 비교 방식 수정

            if len(indices) == 0:
                print(f"[Error] '{self.phase}'에 해당하는 데이터가 없습니다. 'set' 컬럼 값 확인 필요.")
                print(f"✅ set 컬럼 유니크 값: {self.sets.unique()}")  # 유니크한 값 출력
                raise ValueError(f"[Error] '{self.phase}'에 해당하는 데이터가 없습니다.")

            self.images = self.images[indices]
            self.mos = self.mos[indices]

        # ✅ 이미지 경로 생성
        self.image_paths = [os.path.join(self.root, "1024x768", img) for img in self.images]

        print(f"[Debug] Phase: {self.phase}")
        print(f"[Debug] Total Records: {len(self.image_paths)}")  # ✅ 데이터 개수 확인

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

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


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KONIQ10K"

    dataset = KONIQ10KDataset(root=dataset_path, phase="train", crop_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
    print(f"Sample MOS Scores: {sample_batch['mos']}")

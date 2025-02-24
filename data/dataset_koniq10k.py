import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class KONIQ10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "all", crop_size: int = 224):
        """
        ✅ DistortionDetectionModel에 적합하도록 데이터셋 수정
        - `img_A`(왜곡된 이미지)만 반환
        - `img_B`(참조 이미지) 제거
        - `mos`(Mean Opinion Score) 점수 반환
        """
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
        self.mos = scores_csv["MOS"].values
        self.sets = scores_csv["set"].values

        # ✅ 데이터 필터링 (훈련, 검증, 테스트 세트 구분)
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
        """
        ✅ `img_A`(왜곡된 이미지)와 `mos`(Mean Opinion Score)만 반환
        """
        image_path = self.image_paths[index]
        mos = self.mos[index]

        try:
            img_A = Image.open(image_path).convert("RGB")  # ✅ 원본 이미지 로드
        except Exception as e:
            print(f"[Error] Loading image: {image_path}: {e}")
            return None

        img_A_transformed = self.transform(img_A)  # ✅ 변환 적용

        return {
            "img_A": img_A_transformed,  # ✅ 원본 이미지
            "mos": torch.tensor(mos, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    """
    ✅ KONIQ10K는 Authentic 데이터셋이므로 Hard Negative 없이 원본 이미지만 사용.
    """
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KONIQ10K"

    authentic_dataset = KONIQ10KDataset(root=dataset_path, phase="training", crop_size=224)
    authentic_dataloader = DataLoader(authentic_dataset, batch_size=4, shuffle=True)

    print(f"Authentic Dataset size: {len(authentic_dataset)}")

    # ✅ Authentic 데이터셋의 첫 번째 배치 확인
    sample_batch_authentic = next(iter(authentic_dataloader))
    print(f"\n[Authentic] 샘플 확인:")
    for i in range(4):  
        print(f"  Sample {i+1} - MOS: {sample_batch_authentic['mos'][i]}")

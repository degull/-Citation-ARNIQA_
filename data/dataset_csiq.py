import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class CSIQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224, use_hard_negative=True):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.use_hard_negative = use_hard_negative  # ✅ Hard Negative 적용 여부

        # ✅ CSIQ.txt 파일 확인 및 로드
        scores_csv_path = os.path.join(self.root, "CSIQ.txt")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"[Error] CSIQ 데이터셋 CSV 파일이 {scores_csv_path}에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)

        # ✅ 이미지 경로 설정
        self.image_paths = []
        self.reference_paths = []
        self.mos = []

        for _, row in scores_csv.iterrows():
            distortion_type = row["dis_type"].strip().lower()  # ✅ 왜곡 유형
            distorted_filename = row["dis_img_path"].split("/")[-1]  # ✅ 파일명만 가져옴
            reference_filename = row["ref_img_path"].split("/")[-1]  # ✅ 원본 파일명

            # ✅ "contrast dist." → "contrast" 자동 변환
            if "contrast dist." in distortion_type:
                distortion_type = "contrast"

            distorted_path = os.path.normpath(os.path.join(self.root, "dst_imgs", distortion_type, distorted_filename))
            reference_path = os.path.normpath(os.path.join(self.root, "src_imgs", reference_filename))

            # ✅ 존재하는 파일만 추가
            if os.path.exists(distorted_path) and os.path.exists(reference_path):
                self.image_paths.append(distorted_path)
                self.reference_paths.append(reference_path)
                self.mos.append(row["score"])
            else:
                print(f"[Warning] 파일 없음: {distorted_path} 또는 {reference_path}, 스킵됨.")

        # ✅ Hard Negative 적용을 위한 왜곡 유형 리스트
        self.distortion_types = ["motion_blur", "noise", "brightness", "contrast", "downsampling"]
        self.distortion_levels = {
            "motion_blur": [ImageFilter.GaussianBlur(radius) for radius in [1, 2, 3]],
            "noise": [0.01, 0.05, 0.1],
            "brightness": [0.5, 0.75, 1.25, 1.5],
            "contrast": [0.5, 0.75, 1.25, 1.5],
            "downsampling": [0.5, 0.75]
        }

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image: Image, distortion_type: str, level):
        image_np = np.array(image)  # ✅ PIL 이미지를 NumPy 배열로 변환
        if distortion_type == "motion_blur":
            return image.filter(level)
        elif distortion_type == "noise":
            img_array = torch.tensor(image_np, dtype=torch.float32) / 255.0  # ✅ NumPy → Tensor 변환 시 dtype 지정
            noise = torch.randn_like(img_array) * level
            noisy_img = torch.clamp(img_array + noise, 0, 1) * 255
            return Image.fromarray(noisy_img.byte().numpy())  # ✅ NumPy → PIL 변환
        elif distortion_type == "brightness":
            return ImageEnhance.Brightness(image).enhance(level)
        elif distortion_type == "contrast":
            return ImageEnhance.Contrast(image).enhance(level)
        elif distortion_type == "downsampling":
            small_img = image.resize((int(image.width * level), int(image.height * level)))
            return small_img.resize((image.width, image.height))
        return image

    def __getitem__(self, index: int):
        img_A = Image.open(self.image_paths[index]).convert("RGB")  
        img_B_orig = Image.open(self.reference_paths[index]).convert("RGB")  

        if self.use_hard_negative:
            distortion_type = random.choice(self.distortion_types)
            level = random.choice(self.distortion_levels[distortion_type])
            img_B = self.apply_distortion(img_B_orig, distortion_type, level)
        else:
            img_B = img_B_orig

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {
            "img_A": img_A,
            "img_B": img_B,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)

# ✅ `if __name__ == "__main__":`에서 데이터셋 테스트 코드 추가
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ"
    dataset = CSIQDataset(root=dataset_path, phase="train", crop_size=224, use_hard_negative=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    # ✅ 첫 번째 배치 데이터 확인
    sample_batch = next(iter(dataloader))
    
    print(f"Sample batch shapes:")
    print(f"  img_A: {sample_batch['img_A'].shape}")  # (batch_size, 3, 224, 224)
    print(f"  img_B: {sample_batch['img_B'].shape}")  # (batch_size, 3, 224, 224)
    print(f"  MOS: {sample_batch['mos']}")

    # ✅ 첫 번째 샘플 확인
    index = 0
    sample = dataset[index]
    print("\nFirst Sample in Dataset:")
    print(f"  img_A shape: {sample['img_A'].shape}")
    print(f"  img_B shape: {sample['img_B'].shape}")
    print(f"  MOS Score: {sample['mos']}")

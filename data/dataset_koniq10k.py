import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import random
import os
from PIL import ImageEnhance, ImageFilter, Image
import io

# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "awgn",
    2: "blur",
    3: "contrast",
    4: "jpeg",
    5: "jpeg2000",
    6: "fnoise"
}

# 강도 레벨 정의
def get_distortion_levels():
    return {
        'awgn': [5, 10, 15, 20, 25],
        'blur': [1, 2, 3, 4, 5],
        'contrast': [0.5, 0.7, 0.9, 1.1, 1.3],
        'jpeg': [10, 20, 30, 40, 50],
        'jpeg2000': [10, 20, 30, 40, 50],
        'fnoise': [1, 2, 3, 4, 5]
    }


class KONIQ10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "all", crop_size: int = 224):
        super().__init__()
        self.root = root
        self.phase = phase.lower()
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "meta_info_KonIQ10kDataset.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)
        self.images = scores_csv["image_name"].values
        self.mos = scores_csv["MOS"].values
        self.sets = scores_csv["set"].values

        # 데이터 필터링
        if self.phase != "all":
            indices = [
                i for i, s in enumerate(self.sets) if s.strip().lower() == self.phase
            ]
            self.images = self.images[indices]
            self.mos = self.mos[indices]

        # 이미지 경로 생성
        self.image_paths = [
            os.path.join(self.root, "1024x768", img) for img in self.images
        ]

        print(f"[Debug] Phase: {self.phase}")
        print(f"[Debug] Total Records: {len(self.image_paths)}")

        # 기본 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])


    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):
        try:
            image = image.convert("RGB")  # RGB 변환

            if distortion == "awgn":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)

            elif distortion == "jpeg2000":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))

            elif distortion == "jpeg":
                quality = max(1, min(100, int(100 - (level * 100))))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                return Image.open(buffer)

            elif distortion == "fnoise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(0, level * 255, image_array.shape).astype(np.float32)
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            else:
                print(f"[Warning] Distortion type '{distortion}' not implemented.")

        except Exception as e:
            print(f"[Error] Applying distortion {distortion} with level {level}: {e}")

        return image


    def apply_random_distortions(self, image, distortions=None, levels=None):
        if distortions is None:
            distortions = random.sample(list(self.distortion_levels.keys()), 1)
        if levels is None:
            levels = [random.choice(self.distortion_levels[distortion]) for distortion in distortions]

        for distortion, level in zip(distortions, levels):
            try:
                image = self.apply_distortion(image, distortion, level)
            except Exception as e:
                print(f"[Error] Applying distortion {distortion} with level {level}: {e}")
        return image


    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mos = self.mos[index]

        try:
            img_orig = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Loading image: {image_path}: {e}")
            return None

        distortions = random.sample(list(self.distortion_levels.keys()), 1)
        levels = [random.choice(self.distortion_levels[distortions[0]])]

        img_distorted = self.apply_random_distortions(img_orig, distortions, levels)

        img_orig = self.transform(img_orig)
        img_distorted = self.transform(img_distorted)

        return {
            "img_A": img_orig,
            "img_B": img_distorted,
            "mos": torch.tensor(mos, dtype=torch.float32),
        }

    
    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KONIQ10K"
    dataset = KONIQ10KDataset(root=dataset_path, phase="training", crop_size=224)

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"MOS score: {sample['mos']}")
        print(f"Image A shape: {sample['img_A'].shape}")
        print(f"Image B shape: {sample['img_B'].shape}")
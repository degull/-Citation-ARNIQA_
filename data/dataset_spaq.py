import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image, ImageFilter, ImageEnhance
import io


# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "brightness",    # 밝기 조절
    2: "colorfulness",  # 색감 변화
    3: "contrast",      # 대비 조절
    4: "noise",         # 노이즈
    5: "sharpness",     # 선명도
    6: "exposure"       # 노출 문제 (과노출/저노출)
}


# SPAQ 데이터셋의 왜곡 유형 매핑 및 강도 레벨 정의
def get_distortion_levels():
    return {
        'brightness': [0.5, 0.7, 0.9, 1.1, 1.3],  # 밝기 조절
        'colorfulness': [0.5, 0.7, 0.9, 1.1, 1.3],  # 색감 변화
        'contrast': [0.5, 0.7, 0.9, 1.1, 1.3],  # 대비 조절
        'noise': [5, 10, 15, 20, 25],  # 노이즈
        'sharpness': [0.5, 0.7, 0.9, 1.1, 1.3],  # 선명도
        'exposure': [0.5, 0.7, 0.9, 1.1, 1.3]  # 노출 문제 (과노출/저노출)
    }


class SPAQDataset(Dataset):
    def __init__(self, root: str, crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "Annotations", "MOS and Image attribute scores.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)
        self.images = scores_csv["Image name"].values
        self.mos = scores_csv["MOS"].values

        self.image_paths = [
            os.path.join(self.root, "TestImage", img) for img in self.images
        ]

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

            if distortion == "brightness":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(level)

            elif distortion == "colorfulness":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(level)

            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)

            elif distortion == "noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "sharpness":
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(level)

            elif distortion == "exposure":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(level)

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
            #print(f"[Debug] Applying distortion: {distortion} with level: {level}")
            try:
                image = self.apply_distortion(image, distortion, level)
            except Exception as e:
                #print(f"[Error] Applying distortion {distortion} with level {level}: {e}")
                continue
        return image
    

    def __getitem__(self, index: int):
        try:
            img_orig = Image.open(self.image_paths[index]).convert("RGB")
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        distortions = random.sample(list(self.distortion_levels.keys()), 1)
        levels = [random.choice(self.distortion_levels[distortions[0]])]

        img_distorted = self.apply_random_distortions(img_orig, distortions, levels)

        img_orig = self.transform(img_orig)
        img_distorted = self.transform(img_distorted)

        return {
            "img_A": img_orig,
            "img_B": img_distorted,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.images)


# SPAQDataset 테스트
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ"
    dataset = SPAQDataset(root=dataset_path, crop_size=224)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"MOS score: {sample['mos']}")
        print(f"Image A shape: {sample['img_A'].shape}")
        print(f"Image B shape: {sample['img_B'].shape}")
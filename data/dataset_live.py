import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random

# LIVE Dataset
class LIVEDataset(Dataset):
    def __init__(self, root: str, crop_size: int = 224):
        super().__init__()
        self.root = root
        self.crop_size = crop_size

        # CSV 파일 확인 및 로드
        csv_path = os.path.join(self.root, "LIVE_Challenge.txt")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"LIVE_Challenge.txt 파일이 {csv_path} 경로에 존재하지 않습니다.")

        # CSV 데이터 로드
        scores_csv = pd.read_csv(csv_path, sep=",")
        self.image_paths = [os.path.join(self.root, path.replace("LIVE_Challenge/", "")) for path in scores_csv["dis_img_path"].values]
        self.mos = scores_csv["score"].values

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __getitem__(self, index: int):
        try:
            img_path = self.image_paths[index]
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Loading image: {img_path}: {e}")
            return None

        # Transform 이미지
        img_transformed = self.transform(img)

        return {
            "img": img_transformed,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)

# Example Usage
if __name__ == "__main__":
    dataset = LIVEDataset(root="E:/ARNIQA - SE - mix/ARNIQA/dataset/LIVE")
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    if sample is not None:
        print(f"Sample Image Shape: {sample['img'].shape}")
        print(f"MOS Score: {sample['mos']}")

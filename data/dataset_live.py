import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random
import io

# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "jpeg",
    2: "jpeg2000",
    3: "gaussian_blur",
    4: "white_noise",
    5: "fast_fading"
}


# LIVE 데이터셋의 왜곡 유형 매핑 및 강도 레벨 정의
def get_distortion_levels():
    return {
        'jpeg': [10, 20, 30, 40, 50],
        'jpeg2000': [10, 20, 30, 40, 50],
        'gaussian_blur': [1, 2, 3, 4, 5],
        'white_noise': [5, 10, 15, 20, 25],
        'fast_fading': [1, 2, 3, 4, 5]
    }

# LIVE Dataset
class LIVEDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase  # 추가
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # CSV 파일 로드
        csv_path = os.path.join(self.root, "LIVE_Challenge.txt")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"LIVE_Challenge.txt 파일이 {csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(csv_path, sep=",")
        self.image_paths = [
            os.path.join(self.root, path.replace("LIVE_Challenge/", ""))
            for path in scores_csv["dis_img_path"].values
        ]
        self.mos = scores_csv["score"].values

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

            if distortion == "jpeg":
                quality = max(1, min(100, int(100 - level)))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                return Image.open(buffer)

            elif distortion == "jpeg2000":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))

            elif distortion == "gaussian_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "white_noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "fast_fading":
                image = image.filter(ImageFilter.BoxBlur(level))

            else:
                print(f"[Warning] Distortion type '{distortion}' not implemented.")

        except Exception as e:
            print(f"[Error] Applying distortion {distortion} with level {level}: {e}")

        return image
    
    def apply_random_distortion(self, image, distortions=None, levels=None):
        if distortions is None:
            distortions = random.sample(list(self.distortion_levels.keys()), 1)
        if levels is None:
            levels = [random.choice(self.distortion_levels[distortion]) for distortion in distortions]

        for distortion, level in zip(distortions, levels):
            print(f"[Debug] Applying distortion: {distortion} with level: {level}")
            try:
                image = self.apply_distortion(image, distortion, level)
            except Exception as e:
                print(f"[Error] Applying distortion {distortion} with level {level}: {e}")
                continue
        return image

    def __getitem__(self, index: int):
        try:
            img_A_orig = Image.open(self.image_paths[index]).convert("RGB")
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        # 동일한 왜곡 적용
        distortions = random.sample(list(self.distortion_levels.keys()), 1)
        levels = [random.choice(self.distortion_levels[distortions[0]])]

        # 디버깅 로그 추가
        print(f"[Debug] Selected Distortion: {distortions[0]}, Level: {levels[0]}")

        img_A_distorted = self.apply_random_distortion(img_A_orig, distortions, levels)

        img_A_orig = self.transform(img_A_orig)
        img_A_distorted = self.transform(img_A_distorted)

        return {
            "img_A": torch.stack([img_A_orig, img_A_distorted]),
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }


    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/LIVE/"
    dataset = LIVEDataset(root=dataset_path, phase="train", crop_size=224)

    print(f"Dataset size: {len(dataset)}")

    # 첫 번째 데이터 항목 가져오기
    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"MOS score: {sample['mos']}")
        print(f"Image A shape: {sample['img_A'].shape}")

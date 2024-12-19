""" import pandas as pd
import re
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from pathlib import Path

# Distortion types mapping
distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}

# TID2013Dataset 클래스 수정
class TID2013Dataset(Dataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size
        self.is_synthetic = True

        # Load scores from CSV
        scores_csv = pd.read_csv(self.root / "mos.csv")
        self.images = scores_csv["image_id"].values.tolist()
        self.mos = np.array(scores_csv["mean"].values.tolist())

        self.images = np.array([self.root / "distorted_images" / el for el in self.images])
        self.ref_images = [el.split("_")[0].upper() + ".BMP" for el in scores_csv["image_id"].values.tolist()]
        self.ref_images = np.array([self.root / "reference_images" / el for el in self.ref_images])

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]

        self.distortion_types = []
        for image in self.images:
            match = re.search(r'I\d+_(\d+)_(\d+)\.bmp$', str(image))
            dist_type = distortion_types_mapping[int(match.group(1))] if match else None
            self.distortion_types.append(dist_type)

    def apply_distortion(self, image):
        if random.random() > 0.5:
            pil_image = transforms.ToPILImage()(image)
            distorted_image = pil_image.filter(ImageFilter.GaussianBlur(radius=2))
            return transforms.ToTensor()(distorted_image)
        return image

    def __getitem__(self, index: int) -> dict:
        try:
            img_A_orig = Image.open(self.images[index]).convert("RGB")
            img_B_orig = Image.open(self.ref_images[index]).convert("RGB")
        except Exception as e:
            # None 반환 시 에러 원인을 로그로 출력
            print(f"Error loading image: {self.images[index]} or {self.ref_images[index]}: {e}")
            return None

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)

        crops_A = [img_A_orig]
        crops_B = [img_B_orig]

        distorted_crops_A = [self.apply_distortion(img_A_orig) for _ in range(3)]
        distorted_crops_B = [self.apply_distortion(img_B_orig) for _ in range(3)]

        crops_A += distorted_crops_A
        crops_B += distorted_crops_B

        img_A = torch.stack(crops_A)
        img_B = torch.stack(crops_B)

        img_A = img_A.unsqueeze(0)
        img_B = img_B.unsqueeze(0)

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,
            "img_B_ds": img_B,
            "mos": self.mos[index],
            "distortion_type": self.distortion_types[index],
        }

    def transform(self, image):
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __len__(self):
        return len(self.images)

# collate_fn 함수 강화
def collate_fn(batch):
    batch = [x for x in batch if x is not None]  # None 값 필터링
    if len(batch) == 0:  # 배치가 비어있을 경우 빈 딕셔너리 반환
        return {
            "img_A_orig": torch.empty(0),
            "img_B_orig": torch.empty(0),
            "img_A_ds": torch.empty(0),
            "img_B_ds": torch.empty(0),
            "mos": torch.empty(0),
            "distortion_type": []
        }
    return torch.utils.data.dataloader.default_collate(batch)


# Custom collate_fn to handle None values
def collate_fn(batch):
    batch = [x for x in batch if x is not None]  # None 값 필터링
    if len(batch) == 0:  # 배치가 비어있을 경우 빈 딕셔너리 반환
        return {
            "img_A_orig": torch.empty(0),
            "img_B_orig": torch.empty(0),
            "img_A_ds": torch.empty(0),
            "img_B_ds": torch.empty(0),
            "mos": torch.empty(0),
            "distortion_type": []
        }
    return torch.utils.data.dataloader.default_collate(batch)

# Example DataLoader
if __name__ == "__main__":
    dataset = TID2013Dataset(root="path_to_tid2013_dataset", phase="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        if batch is None:
            continue
        print(batch)
 """

# TID2013Dataset 클래스
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


# 강도 레벨 정의
def get_distortion_levels():
    return {
        'gaussian_blur': [1, 2, 3, 4, 5],
        'lens_blur': [1, 2, 3, 4, 5],
        'motion_blur': [1, 2, 3, 4, 5],
        'color_diffusion': [0.05, 0.1, 0.2, 0.3, 0.4],
        'color_shift': [10, 20, 30, 40, 50],
        'jpeg2000': [0.1, 0.2, 0.3, 0.4, 0.5],
        'jpeg': [0.1, 0.2, 0.3, 0.4, 0.5],
        'white_noise': [5, 10, 15, 20, 25],
        'impulse_noise': [0.05, 0.1, 0.2, 0.3, 0.4],
        'multiplicative_noise': [0.1, 0.2, 0.3, 0.4, 0.5]
    }

class TID2013Dataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # 정확한 MOS 경로 확인 및 로드
        scores_csv_path = os.path.join(self.root, "mos.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"mos.csv 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")
        
        scores_csv = pd.read_csv(scores_csv_path)
        self.images = scores_csv["image_id"].values
        self.mos = scores_csv["mean"].values

        self.image_paths = [
            os.path.join(self.root, "distorted_images", img) for img in self.images
        ]
        self.reference_paths = [
            os.path.join(self.root, "reference_images", img.split("_")[0] + ".BMP")
            for img in self.images
        ]

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):
        try:
            if distortion == "gaussian_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))
            elif distortion == "lens_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))
            elif distortion == "motion_blur":
                image = image.filter(ImageFilter.BoxBlur(level))
            elif distortion == "color_diffusion":
                diffused = np.array(image).astype(np.float32)
                diffused += np.random.uniform(-level, level, diffused.shape)
                image = Image.fromarray(np.clip(diffused, 0, 255).astype(np.uint8))
            elif distortion == "color_shift":
                shifted = np.array(image).astype(np.float32)
                shift_amount = np.random.randint(-level, level + 1, shifted.shape)
                shifted += shift_amount
                image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))
            elif distortion == "jpeg2000":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))
            elif distortion == "white_noise":
                noise = np.random.normal(0, level, (image.height, image.width, 3))
                noisy_image = np.array(image).astype(np.float32) + noise
                image = Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))
            elif distortion == "impulse_noise":
                image = np.array(image)
                prob = 0.1
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        if random.random() < prob:
                            image[i][j] = np.random.choice([0, 255], size=3)
                image = Image.fromarray(image)
        except Exception as e:
            print(f"[Error] Applying distortion {distortion} with level {level}: {e}")
        return image


    def apply_random_distortions(self, image, distortions=None, levels=None):
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
            img_B_orig = Image.open(self.reference_paths[index]).convert("RGB")
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]} or {self.reference_paths[index]}: {e}")
            return None

        # 동일한 왜곡 적용
        distortions = random.sample(list(self.distortion_levels.keys()), 1)
        levels = [random.choice(self.distortion_levels[distortions[0]])]

        # 디버깅 로그 추가
        print(f"[Debug] Selected Distortion: {distortions[0]}, Level: {levels[0]}")

        img_A_distorted = self.apply_random_distortions(img_A_orig, distortions, levels)
        img_B_distorted = self.apply_random_distortions(img_B_orig, distortions, levels)

        # Positive Pair 검증
        verify_positive_pairs(
            distortions_A=distortions[0],
            distortions_B=distortions[0],
            applied_distortions_A=levels[0],
            applied_distortions_B=levels[0],
        )

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)
        img_A_distorted = self.transform(img_A_distorted)
        img_B_distorted = self.transform(img_B_distorted)

        return {
            "img_A": torch.stack([img_A_orig, img_A_distorted]),
            "img_B": torch.stack([img_B_orig, img_B_distorted]),
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.images)

# ver1
""" 
import pandas as pd
import re
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
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


# 강도 레벨 정의
def get_distortion_levels():
    return {
        'gaussian_blur': [1, 2, 3, 4, 5],
        'lens_blur': [1, 2, 3, 4, 5],
        'motion_blur': [1, 2, 3, 4, 5],
        'color_diffusion': [0.05, 0.1, 0.2, 0.3, 0.4],
        'color_shift': [10, 20, 30, 40, 50],  # 양수 값만 허용
        'jpeg2000': [0.1, 0.2, 0.3, 0.4, 0.5],
        'jpeg': [0.1, 0.2, 0.3, 0.4, 0.5],
        'white_noise': [5, 10, 15, 20, 25],
        'impulse_noise': [0.05, 0.1, 0.2, 0.3, 0.4],
        'multiplicative_noise': [0.1, 0.2, 0.3, 0.4, 0.5]
    }

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        if self.root.is_file():
            csv_path = self.root
            self.dataset_root = self.root.parent
        else:
            csv_path = self.root / "kadid10k.csv"
            self.dataset_root = self.root

        scores_csv = pd.read_csv(csv_path)
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        self.images = np.array([self.dataset_root / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([self.dataset_root / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        self.distortion_types = []
        self.distortion_levels_list = []

        for img in self.images:
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img))
            if match:
                dist_type = distortion_types_mapping[int(match.group(1))]
                self.distortion_types.append(dist_type)
                self.distortion_levels_list.append(int(match.group(2)))

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels_list = np.array(self.distortion_levels_list)

        if self.phase != "all":
            split_file_path = self.dataset_root / "splits" / f"{self.phase}.npy"
            if not split_file_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_file_path}")
            split_idxs = np.load(split_file_path)[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]
            self.distortion_types = self.distortion_types[split_idxs]
            self.distortion_levels_list = self.distortion_levels_list[split_idxs]

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):
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
            image = image.resize((image.width // 2, image.height // 2))
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

        return image

    def apply_random_distortions(self, image, num_distortions=4):
        distortions = random.sample(list(self.distortion_levels.keys()), num_distortions)

        for distortion in distortions:
            level = random.choice(self.distortion_levels[distortion])

            if level <= 0:
                print(f"Skipping distortion: {distortion} with invalid level: {level}")
                continue

            print(f"Applying distortion: {distortion} with level: {level}")
            try:
                image = self.apply_distortion(image, distortion, level)
            except Exception as e:
                print(f"Error applying distortion {distortion} with level {level}: {e}")
                raise

        return image

    def __getitem__(self, index: int):
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        img_A_distorted = self.apply_random_distortions(img_A_orig)
        img_B_distorted = self.apply_random_distortions(img_B_orig)

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)
        img_A_distorted = self.transform(img_A_distorted)
        img_B_distorted = self.transform(img_B_distorted)

        return {
            "img_A": torch.stack([img_A_orig, img_A_distorted]),
            "img_B": torch.stack([img_B_orig, img_B_distorted]),
            "mos": self.mos[index],
        }

    def __len__(self):
        return len(self.images)
"""

import pandas as pd
import re
import numpy as np
from PIL import Image, ImageFilter
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


# 강도 레벨 정의
def get_distortion_levels():
    return {
        'gaussian_blur': [1, 2, 3, 4, 5],
        'lens_blur': [1, 2, 3, 4, 5],
        'motion_blur': [1, 2, 3, 4, 5],
        'color_diffusion': [0.05, 0.1, 0.2, 0.3, 0.4],
        'color_shift': [10, 20, 30, 40, 50],  # 양수 값만 허용
        'jpeg2000': [0.1, 0.2, 0.3, 0.4, 0.5],
        'jpeg': [0.1, 0.2, 0.3, 0.4, 0.5],
        'white_noise': [5, 10, 15, 20, 25],
        'impulse_noise': [0.05, 0.1, 0.2, 0.3, 0.4],
        'multiplicative_noise': [0.1, 0.2, 0.3, 0.4, 0.5]
    }

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


class KADID10KDataset(Dataset):
    # 단일
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # distortion_levels 초기화
        self.distortion_levels = get_distortion_levels()

        # CSV 파일 확인
        if self.root.is_file():
            csv_path = self.root
            self.dataset_root = self.root.parent
        else:
            csv_path = self.root / "kadid10k.csv"
            self.dataset_root = self.root

        # CSV 파일 로드
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        scores_csv = pd.read_csv(csv_path)

        # 이미지 경로 설정
        self.images = scores_csv["dist_img"].values
        self.reference_images = scores_csv["ref_img"].values
        self.mos = scores_csv["dmos"].values

        # 정확한 경로로 수정
        self.image_paths = [
            str(self.dataset_root / "images" / img) for img in self.images
        ]
        self.reference_paths = [
            str(self.dataset_root / "images" / img) for img in self.reference_images
        ]



    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):
        try:
            image = image.convert("RGB")  # Ensure the image is in RGB format

            if distortion == "gaussian_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "lens_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "motion_blur":
                image = image.filter(ImageFilter.BoxBlur(level))

            elif distortion == "color_diffusion":
                diffused = np.array(image).astype(np.float32)
                
                # 색상 확산을 위한 무작위 값 생성
                diffusion = np.random.uniform(-level * 255, level * 255, size=diffused.shape).astype(np.float32)
                
                # 색상 확산 적용
                diffused += diffusion
                
                # 값 클리핑 (0~255 범위로 제한)
                diffused = np.clip(diffused, 0, 255).astype(np.uint8)

                # 다시 PIL 이미지로 변환
                image = Image.fromarray(diffused)

            elif distortion == "color_shift":
                shifted = np.array(image).astype(np.float32)
                shift_amount = np.random.uniform(-level * 255, level * 255, shifted.shape[-1])
                shifted += shift_amount
                image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))

            elif distortion == "jpeg2000":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))

            elif distortion == "jpeg":
                # JPEG 품질 수준은 1 ~ 100의 정수로 설정
                quality = max(1, min(100, int(100 - (level * 100))))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                return Image.open(buffer)

            elif distortion == "white_noise":
                # 이미지를 numpy 배열로 변환 (float32로)
                image_array = np.array(image, dtype=np.float32)
                
                # 노이즈 생성 (가우시안 노이즈)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                
                # 원본 이미지에 노이즈 추가
                noisy_image = image_array + noise
                
                # 값 클리핑 (0~255 범위로 제한)
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                
                # 디버깅: noise와 noisy_image 값 확인
                print("White Noise Debug:")
                print("Noise min/max:", noise.min(), noise.max())
                print("Noisy Image min/max:", noisy_image.min(), noisy_image.max())
                
                # 다시 PIL 이미지로 변환
                image = Image.fromarray(noisy_image)

            elif distortion == "impulse_noise":
                image_array = np.array(image).astype(np.float32)  # NumPy 배열로 변환
                prob = level
                mask = np.random.choice([0, 1], size=image_array.shape[:2], p=[1 - prob, prob])
                random_noise = np.random.choice([0, 255], size=(image_array.shape[0], image_array.shape[1], 1))
                image_array[mask == 1] = random_noise[mask == 1]
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                return Image.fromarray(image_array)

            elif distortion == "multiplicative_noise":
                image_array = np.array(image).astype(np.float32)  # NumPy 배열로 변환
                noise = np.random.normal(1, level, image_array.shape)  # 1을 기준으로 곱셈 노이즈 생성
                noisy_image = image_array * noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # 0~255로 클리핑
                return Image.fromarray(noisy_image)

            elif distortion == "denoise":
                image = image.filter(ImageFilter.MedianFilter(size=int(level)))

            elif distortion == "brighten":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1 + level)

            elif distortion == "darken":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1 - level)

            elif distortion == "mean_shift":
                shifted_image = np.array(image).astype(np.float32) + level * 255
                image = Image.fromarray(np.clip(shifted_image, 0, 255).astype(np.uint8))

            elif distortion == "jitter":
                jitter = np.random.randint(-level * 255, level * 255, (image.height, image.width, 3))
                img_array = np.array(image).astype(np.float32) + jitter
                image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            elif distortion == "non_eccentricity_patch":
                width, height = image.size
                crop_level = int(level * min(width, height))
                image = image.crop((crop_level, crop_level, width - crop_level, height - crop_level))
                image = image.resize((width, height))

            elif distortion == "pixelate":
                width, height = image.size
                image = image.resize((width // level, height // level)).resize((width, height), Image.NEAREST)

            elif distortion == "quantization":
                quantized = (np.array(image) // int(256 / level)) * int(256 / level)
                image = Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))

            elif distortion == "color_block":
                block_size = max(1, int(image.width * level))
                img_array = np.array(image)
                for i in range(0, img_array.shape[0], block_size):
                    for j in range(0, img_array.shape[1], block_size):
                        block_color = np.random.randint(0, 256, (1, 1, 3))
                        img_array[i:i + block_size, j:j + block_size] = block_color
                image = Image.fromarray(img_array)

            elif distortion == "high_sharpen":
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(level)

            elif distortion == "contrast_change":
                enhancer = ImageEnhance.Contrast(image)
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

# dataset_kadid10k.py
#train.py → simclr.py → resnet_se.py → attention_se.py
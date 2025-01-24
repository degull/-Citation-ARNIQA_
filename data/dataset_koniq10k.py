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
        'color_shift': [10, 20, 30, 40, 50],
        'jpeg2000': [0.1, 0.2, 0.3, 0.4, 0.5],
        'jpeg': [0.1, 0.2, 0.3, 0.4, 0.5],
        'white_noise': [5, 10, 15, 20, 25],
        'impulse_noise': [0.05, 0.1, 0.2, 0.3, 0.4],
        'multiplicative_noise': [0.1, 0.2, 0.3, 0.4, 0.5],
        'denoise': [1, 2, 3, 4, 5],  # 강도 레벨 추가
        'brighten': [0.1, 0.2, 0.3, 0.4, 0.5],
        'darken': [0.1, 0.2, 0.3, 0.4, 0.5],
        'mean_shift': [0.1, 0.2, 0.3, 0.4, 0.5],
        'jitter': [1, 2, 3, 4, 5],
        'non_eccentricity_patch': [0.1, 0.2, 0.3, 0.4, 0.5],
        'pixelate': [1, 2, 3, 4, 5],
        'quantization': [1, 2, 3, 4, 5],
        'color_block': [0.1, 0.2, 0.3, 0.4, 0.5],
        'high_sharpen': [1, 2, 3, 4, 5],
        'contrast_change': [0.1, 0.2, 0.3, 0.4, 0.5]
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
            image = image.convert("RGB")  # Ensure the image is in RGB format

            if distortion == "awgn":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                noisy_image = image_array + noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
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
                noisy_image = image_array + noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            # Additional distortions from TID2013
            elif distortion == "gaussian_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "lens_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "motion_blur":
                image = image.filter(ImageFilter.BoxBlur(level))

            elif distortion == "color_diffusion":
                diffused = np.array(image).astype(np.float32)
                diffusion = np.random.uniform(-level * 255, level * 255, size=diffused.shape).astype(np.float32)
                diffused += diffusion
                diffused = np.clip(diffused, 0, 255).astype(np.uint8)
                image = Image.fromarray(diffused)

            elif distortion == "color_shift":
                shifted = np.array(image).astype(np.float32)
                shift_amount = np.random.uniform(-level * 255, level * 255, shifted.shape[-1])
                shifted += shift_amount
                image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))

            elif distortion == "white_noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                noisy_image = image_array + noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "impulse_noise":
                image_array = np.array(image).astype(np.float32)
                prob = level
                mask = np.random.choice([0, 1], size=image_array.shape[:2], p=[1 - prob, prob])
                random_noise = np.random.choice([0, 255], size=(image_array.shape[0], image_array.shape[1], 1))
                image_array[mask == 1] = random_noise[mask == 1]
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                image = Image.fromarray(image_array)

            elif distortion == "multiplicative_noise":
                image_array = np.array(image).astype(np.float32)
                noise = np.random.normal(1, level, image_array.shape)
                noisy_image = image_array * noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

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
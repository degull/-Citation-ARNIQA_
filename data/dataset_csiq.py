import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import io
from PIL import ImageEnhance, ImageFilter, Image


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
        'multiplicative_noise': [0.1, 0.2, 0.3, 0.4, 0.5]
    }

# Positive Pair 검증 함수
def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")

class CSIQDataset(Dataset):
    # 단일
    #def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
    #    super().__init__()
    #    self.root = str(root)
    #    self.phase = phase
    #    self.crop_size = crop_size
    #    self.distortion_levels = get_distortion_levels()
#
    #    # MOS 파일 확인 및 로드
    #    scores_csv_path = os.path.join(self.root, "CSIQ.txt")
    #    if not os.path.isfile(scores_csv_path):
    #        raise FileNotFoundError(f"CSIQ.txt 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")
#
    #    scores_csv = pd.read_csv(scores_csv_path, sep=",")
    #    self.image_paths = [os.path.join(self.root, img.replace("CSIQ/", "")) for img in scores_csv["dis_img_path"].values]
    #    self.reference_paths = [os.path.join(self.root, img.replace("CSIQ/", "")) for img in scores_csv["ref_img_path"].values]
    #    self.mos = scores_csv["score"].values

    # cross-dataset
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # MOS 파일 확인 및 로드
        self.root = "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ"
        scores_csv_path = os.path.join(self.root, "CSIQ.txt")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSIQ.txt 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path, sep=",")
        self.image_paths = [os.path.join(self.root, img.replace("CSIQ/", "")) for img in scores_csv["dis_img_path"].values]
        self.reference_paths = [os.path.join(self.root, img.replace("CSIQ/", "")) for img in scores_csv["ref_img_path"].values]
        self.mos = scores_csv["score"].values



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

        distortions = random.sample(list(self.distortion_levels.keys()), 1)
        levels = [random.choice(self.distortion_levels[distortions[0]])]

        print(f"[Debug] Selected Distortion: {distortions[0]}, Level: {levels[0]}")

        img_A_distorted = self.apply_random_distortions(img_A_orig, distortions, levels)
        img_B_distorted = self.apply_random_distortions(img_B_orig, distortions, levels)

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
        return len(self.image_paths)
    
# CSIQDataset 테스트
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ"
    dataset = CSIQDataset(root=dataset_path, phase="train", crop_size=224)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"MOS score: {sample['mos']}")
        print(f"Image A shape: {sample['img_A'].shape}")
        print(f"Image B shape: {sample['img_B'].shape}")
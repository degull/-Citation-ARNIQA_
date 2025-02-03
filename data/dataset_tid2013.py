
# TID2013Dataset 클래스
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import io
from PIL import ImageEnhance, ImageFilter, Image
import io
from pathlib import Path
import cv2

# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "additive_gaussian_noise",
    2: "additive_noise_in_color_components",
    3: "spatially_correlated_noise",
    4: "masked_noise",
    5: "high_frequency_noise",
    6: "impulse_noise",
    7: "quantization_noise",
    8: "gaussian_blur",
    9: "image_denoising",
    10: "jpeg_compression",
    11: "jpeg2000_compression",
    12: "jpeg_transmission_errors",
    13: "jpeg2000_transmission_errors",
    14: "non_eccentricity_pattern_noise",
    15: "local_block_wise_distortions",
    16: "mean_shift",
    17: "contrast_change",
    18: "change_of_color_saturation",
    19: "multiplicative_gaussian_noise",
    20: "comfort_noise",
    21: "lossy_compression_of_noisy_images",
    22: "image_color_quantization_with_dither",
    23: "chromatic_aberrations",
    24: "sparse_sampling_and_reconstruction"
}


# TID2013 기준 강도 레벨 정의
def get_distortion_levels():
    return {
        'additive_gaussian_noise': [5, 10, 15, 20, 25],  
        'additive_noise_in_color_components': [5, 10, 15, 20, 25],  
        'spatially_correlated_noise': [1, 2, 3, 4, 5],  
        'masked_noise': [1, 2, 3, 4, 5],  
        'high_frequency_noise': [1, 2, 3, 4, 5],  
        'impulse_noise': [0.05, 0.1, 0.2, 0.3, 0.4],  
        'quantization_noise': [1, 2, 3, 4, 5],  
        'gaussian_blur': [1, 2, 3, 4, 5],  
        'image_denoising': [1, 2, 3, 4, 5],  
        'jpeg_compression': [0.1, 0.2, 0.3, 0.4, 0.5],  
        'jpeg2000_compression': [0.1, 0.2, 0.3, 0.4, 0.5],  
        'jpeg_transmission_errors': [1, 2, 3, 4, 5],  
        'jpeg2000_transmission_errors': [1, 2, 3, 4, 5],  
        'non_eccentricity_pattern_noise': [1, 2, 3, 4, 5],  
        'local_block_wise_distortions': [1, 2, 3, 4, 5],  
        'mean_shift': [0.1, 0.2, 0.3, 0.4, 0.5],  
        'contrast_change': [0.1, 0.2, 0.3, 0.4, 0.5],  
        'change_of_color_saturation': [0.1, 0.2, 0.3, 0.4, 0.5],  
        'multiplicative_gaussian_noise': [0.1, 0.2, 0.3, 0.4, 0.5],  
        'comfort_noise': [1, 2, 3, 4, 5],  
        'lossy_compression_of_noisy_images': [1, 2, 3, 4, 5],  
        'image_color_quantization_with_dither': [1, 2, 3, 4, 5],  
        'chromatic_aberrations': [1, 2, 3, 4, 5],  
        'sparse_sampling_and_reconstruction': [1, 2, 3, 4, 5]  
    }


class TID2013Dataset(Dataset):

    # 단일
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


    # cross-dataset1
      
    #def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
    #    super().__init__()
    #    self.root = str(root)  # 정확한 파일 경로를 root로 전달
    #    self.phase = phase
    #    self.crop_size = crop_size
    #    self.distortion_levels = get_distortion_levels()
#
    #    # 정확한 MOS 경로 확인 및 로드
    #    scores_csv_path = self.root  # mos.csv 파일 경로 직접 사용
    #    if not os.path.isfile(scores_csv_path):
    #        raise FileNotFoundError(f"mos.csv 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")
    #    
    #    scores_csv = pd.read_csv(scores_csv_path)
    #    self.images = scores_csv["image_id"].values
    #    self.mos = scores_csv["mean"].values
#
    #    # 이미지 경로 생성
    #    self.image_paths = [
    #        os.path.join(os.path.dirname(self.root), "distorted_images", img) for img in self.images
    #    ]
    #    self.reference_paths = [
    #        os.path.join(os.path.dirname(self.root), "reference_images", img.split("_")[0] + ".BMP")
    #        for img in self.images
    #    ]


        
    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)
    

    def apply_distortion(self, image, distortion, level):
        try:
            image = image.convert("RGB")  # Ensure image is in RGB format

            if distortion == "additive_gaussian_noise":
                noise = np.random.normal(0, level, (image.height, image.width, 3))
                noisy_image = np.array(image).astype(np.float32) + noise
                image = Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))

            elif distortion == "additive_noise_in_color_components":
                noise = np.random.normal(0, level, (image.height, image.width, 3))
                noisy_image = np.array(image).astype(np.float32) + noise
                image = Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))

            elif distortion == "spatially_correlated_noise":
                kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
                noise = np.random.normal(0, level, (image.height, image.width, 3))
                noisy_image = cv2.filter2D(np.array(image).astype(np.float32) + noise, -1, kernel)
                image = Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))

            elif distortion == "masked_noise":
                image_array = np.array(image).astype(np.float32)

                # ✅ prob이 0~1 사이 값이 되도록 보정
                prob = max(0, min(level / 5, 1))  # level 값이 0~5라면 0~1로 정규화

                mask = np.random.choice([0, 1], size=(image_array.shape[0], image_array.shape[1], 1), p=[1 - prob, prob])

                # ✅ Mask를 RGB 채널 수에 맞게 확장
                mask = np.repeat(mask, 3, axis=2)

                random_noise = np.random.choice([0, 255], size=image_array.shape)
                image_array[mask == 1] = random_noise[mask == 1]

                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                return Image.fromarray(image_array)

            elif distortion == "high_frequency_noise":
                freq_noise = np.random.normal(0, level, (image.height, image.width, 3))
                image_array = np.array(image).astype(np.float32) + freq_noise
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            elif distortion == "impulse_noise":
                image_array = np.array(image).astype(np.float32)
                prob = level
                mask = np.random.choice([0, 1], size=image_array.shape[:2], p=[1 - prob, prob])
                random_noise = np.random.choice([0, 255], size=(image_array.shape[0], image_array.shape[1], 1))
                image_array[mask == 1] = random_noise[mask == 1]
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                return Image.fromarray(image_array)

            elif distortion == "quantization_noise":
                quantized = (np.array(image) // int(256 / level)) * int(256 / level)
                image = Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))

            elif distortion == "gaussian_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "image_denoising":
                image = image.filter(ImageFilter.MedianFilter(size=int(level)))

            elif distortion == "jpeg_compression":
                quality = max(1, min(100, int(100 - (level * 100))))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                image = Image.open(buffer).convert("RGB")  # ✅ JPEG 압축 후 RGB 변환 추가


            elif distortion == "jpeg2000_compression":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))

            elif distortion == "jpeg_transmission_errors":
                image = image.convert("L")  # Grayscale 변환
                image = image.resize((image.width // 2, image.height // 2))
                image = image.resize((image.width, image.height))
                image = image.convert("RGB")  # ✅ 다시 RGB로 변환 추가


            elif distortion == "contrast_change":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)

            elif distortion == "change_of_color_saturation":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(level)

            elif distortion == "multiplicative_gaussian_noise":
                image_array = np.array(image).astype(np.float32)
                noise = np.random.normal(1, level, image_array.shape)
                noisy_image = image_array * noise
                image = Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))

            elif distortion == "mean_shift":
                shifted_image = np.array(image).astype(np.float32) + level * 255
                image = Image.fromarray(np.clip(shifted_image, 0, 255).astype(np.uint8)).convert("RGB")  # ✅ 추가

            elif distortion == "comfort_noise":
                image = image.filter(ImageFilter.SMOOTH)

            elif distortion == "non_eccentricity_pattern_noise":
                width, height = image.size
                crop_level = int(level * min(width, height))

                # ✅ Crop 좌표가 음수가 되지 않도록 제한
                crop_level = min(crop_level, width // 2 - 1, height // 2 - 1)

                left = max(0, crop_level)
                top = max(0, crop_level)
                right = min(width, width - crop_level)
                bottom = min(height, height - crop_level)

                if right > left and bottom > top:
                    image = image.crop((left, top, right, bottom))
                else:
                    print(f"[Warning] Skipping 'non_eccentricity_pattern_noise' for level {level} due to invalid crop size.")

            elif distortion == "local_block_wise_distortions":
                image_array = np.array(image)
                block_size = max(1, int(image.width * level))
                for i in range(0, image_array.shape[0], block_size):
                    for j in range(0, image_array.shape[1], block_size):
                        image_array[i:i + block_size, j:j + block_size] = np.mean(image_array[i:i + block_size, j:j + block_size])
                image = Image.fromarray(image_array)

            elif distortion == "image_color_quantization_with_dither":
                quantized = (np.array(image) // int(256 / level)) * int(256 / level)
                image = Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))

            elif distortion == "lossy_compression_of_noisy_images":
                image = image.resize((image.width // 2, image.height // 2))
                image = image.resize((image.width, image.height))
                image = image.convert("RGB")

            elif distortion == "chromatic_aberrations":
                image = image.convert("RGB")
                r, g, b = image.split()

                shift_x = int(level * 2)
                shift_y = int(level * 2)

                # ✅ `offset` 대신 `numpy.roll()` 사용하여 채널 이동
                r_array = np.array(r)
                b_array = np.array(b)

                r_shifted = np.roll(r_array, shift=(shift_x, shift_y), axis=(0, 1))
                b_shifted = np.roll(b_array, shift=(-shift_x, -shift_y), axis=(0, 1))

                r_new = Image.fromarray(r_shifted)
                b_new = Image.fromarray(b_shifted)

                image = Image.merge("RGB", (r_new, g, b_new))

            elif distortion == "sparse_sampling_and_reconstruction":
                downsampled = image.resize((image.width // level, image.height // level))
                image = downsampled.resize((image.width, image.height), Image.BICUBIC)

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
    

    # TID2013Dataset 테스트
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/"
    dataset = TID2013Dataset(root=dataset_path, phase="train", crop_size=224)

    print(f"Dataset size: {len(dataset)}")

    # 첫 번째 데이터 항목 가져오기
    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"MOS score: {sample['mos']}")
        print(f"Image A shape: {sample['img_A'].shape}")
        print(f"Image B shape: {sample['img_B'].shape}")

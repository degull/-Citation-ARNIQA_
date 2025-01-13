
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

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


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
    

    # TID2013Dataset 테스트
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/mos.csv"
    dataset = TID2013Dataset(root=dataset_path, phase="train", crop_size=224)

    print(f"Dataset size: {len(dataset)}")

    # 첫 번째 데이터 항목 가져오기
    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"MOS score: {sample['mos']}")
        print(f"Image A shape: {sample['img_A'].shape}")
        print(f"Image B shape: {sample['img_B'].shape}")

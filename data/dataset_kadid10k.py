import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import matplotlib.pyplot as plt  # ✅ 추가: 시각적 비교를 위해 matplotlib 사용

class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224, dataset_type="synthetic"):
        """
        dataset_type: 
            "synthetic" (KADID10K, CSIQ) → Hard Negative 적용
            "authentic" (KonIQ-10k, SPAQ, LIVE-FB) → Hard Negative 적용 안함
        """
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.dataset_type = dataset_type  # ✅ 데이터셋 유형 결정

        # ✅ CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "kadid10k.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"KADID10K CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)

        # ✅ 이미지 경로 설정
        self.image_paths = [os.path.join(self.root, "images", img) for img in scores_csv["dist_img"]]
        self.reference_paths = [os.path.join(self.root, "images", img) for img in scores_csv["ref_img"]]
        self.mos = scores_csv["dmos"].values

        # ✅ KADID-10K 데이터셋의 25개 왜곡 유형 (Hard Negative를 위한 리스트)
        self.distortion_types = [
            "gaussian_blur", "lens_blur", "motion_blur", "color_diffusion", "color_shift",
            "color_quantization", "color_saturation_1", "color_saturation_2", "jpeg2000", "jpeg",
            "white_noise", "white_noise_color_component", "impulse_noise", "multiplicative_noise",
            "denoise", "brighten", "darken", "mean_shift", "jitter", "non_eccentricity_patch",
            "pixelate", "quantization", "color_block", "high_sharpen", "contrast_change"
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
                diffusion = np.random.uniform(-level * 255, level * 255, size=diffused.shape).astype(np.float32)
                diffused += diffusion
                diffused = np.clip(diffused, 0, 255).astype(np.uint8)
                image = Image.fromarray(diffused)

            elif distortion == "color_shift":
                shifted = np.array(image).astype(np.float32)
                shift_amount = np.random.uniform(-level * 255, level * 255, shifted.shape[-1])
                shifted += shift_amount
                image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))

            elif distortion == "color_quantization":
                quantized = (np.array(image) // int(256 / level)) * int(256 / level)
                image = Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))

            elif distortion == "color_saturation_1":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1 + level)

            elif distortion == "color_saturation_2":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1 - level)

            elif distortion == "jpeg2000":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))

            elif distortion == "jpeg":
                quality = max(1, min(100, int(100 - (level * 100))))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                return Image.open(buffer)

            elif distortion == "white_noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                noisy_image = image_array + noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "white_noise_color_component":
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
                return Image.fromarray(image_array)

            elif distortion == "multiplicative_noise":
                image_array = np.array(image).astype(np.float32)
                noise = np.random.normal(1, level, image_array.shape)
                noisy_image = image_array * noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_image)

            elif distortion == "denoise":
                filter_size = max(3, min(int(level * 10) * 2 + 1, 15))  # 최소 3, 최대 15의 홀수 필터 크기 설정
                if filter_size % 2 == 0:
                    filter_size += 1  # 홀수로 변환
                print(f"[Debug] Applying MedianFilter with size {filter_size}")  # 디버깅 로그 추가
                image = image.filter(ImageFilter.MedianFilter(size=filter_size))

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
                level = max(1, int(level * 10))  # float → int 변환
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

    def __getitem__(self, index: int):
        """
        ✅ 데이터셋 유형에 따라 `img_B` 처리 방식 변경 ✅
        - Synthetic 데이터셋(KADID10K, CSIQ) → Hard Negative 적용
        - Authentic 데이터셋(KonIQ-10k, SPAQ, LIVE-FB) → Hard Negative 적용 안함
        """
        img_A = Image.open(self.image_paths[index]).convert("RGB")  
        img_B = Image.open(self.reference_paths[index]).convert("RGB")  

        # ✅ Synthetic 데이터셋에만 Hard Negative 적용
        if self.dataset_type == "synthetic":
            distortion_type = random.choice(self.distortion_types)
            level = random.uniform(0.1, 0.5)
            img_B = self.apply_distortion(img_B, distortion_type, level)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {
            "img_A": img_A,
            "img_B": img_B,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    """
    ✅ Hard Negative 적용 여부를 확인하고, 이미지 비교를 수행
    """
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K"

    synthetic_dataset = KADID10KDataset(root=dataset_path, phase="train", crop_size=224, dataset_type="synthetic")
    synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=4, shuffle=True)

    print(f"Synthetic Dataset size: {len(synthetic_dataset)}")

    # ✅ Hard Negative 적용 확인
    sample_batch_synthetic = next(iter(synthetic_dataloader))
    print(f"\n[Synthetic] Hard Negative 적용 확인:")
    for i in range(4):  
        print(f"  Sample {i+1} - MOS: {sample_batch_synthetic['mos'][i]}")

    # ✅ 원본 이미지 vs Hard Negative 비교
    sample_index = 0
    img_A_np = sample_batch_synthetic['img_A'][sample_index].permute(1, 2, 0).numpy()
    img_B_np = sample_batch_synthetic['img_B'][sample_index].permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_A_np)
    ax[0].set_title("Distorted Image (img_A)")
    ax[1].imshow(img_B_np)
    ax[1].set_title("Hard Negative (img_B)")
    plt.show()

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import matplotlib.pyplot as plt

class TID2013Dataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224, dataset_type="synthetic"):
        """
        dataset_type: 
            "synthetic" (TID2013) → Hard Negative 적용
            "authentic" (KonIQ-10k, SPAQ, LIVE-FB) → Hard Negative 적용 안함
        """
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.dataset_type = dataset_type  # ✅ 데이터셋 유형 결정

        # ✅ MOS CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "mos.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"TID2013 MOS CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_data = pd.read_csv(scores_csv_path)

        # ✅ 이미지 경로 설정
        self.image_paths = [os.path.join(self.root, "distorted_images", img) for img in scores_data["image_id"]]
        self.reference_paths = [
            os.path.join(self.root, "reference_images", img.split("_")[0] + ".BMP") for img in scores_data["image_id"]
        ]
        self.mos = scores_data["mean"].astype(float).values  # MOS 값을 float로 변환

        # ✅ TID2013 데이터셋의 24개 왜곡 유형 (Hard Negative 적용 대상)
        self.distortion_types = [
            "additive_gaussian_noise", "additive_noise_in_color_components", "spatially_correlated_noise",
            "masked_noise", "high_frequency_noise", "impulse_noise", "quantization_noise",
            "gaussian_blur", "image_denoising", "jpeg_compression", "jpeg2000_compression",
            "jpeg_transmission_errors", "jpeg2000_transmission_errors", "non_eccentricity_pattern_noise",
            "local_block_wise_distortions", "mean_shift", "contrast_change", "change_of_color_saturation",
            "multiplicative_gaussian_noise", "comfort_noise", "lossy_compression_of_noisy_images",
            "image_color_quantization_with_dither", "chromatic_aberrations", "sparse_sampling_and_reconstruction"
        ]

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):
        try:
            image = image.convert("RGB")  # Ensure the image is in RGB format

            if distortion == "additive_gaussian_noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(0, level * 255, image_array.shape).astype(np.float32)
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "additive_noise_in_color_components":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(0, level * 255, image_array.shape).astype(np.float32)
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "spatially_correlated_noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(0, level * 255, image_array.shape)
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "masked_noise":
                image_array = np.array(image, dtype=np.float32)
                mask = np.random.choice([0, 1], size=image_array.shape, p=[1 - level, level])
                image_array[mask == 1] = 0  # 마스크된 영역을 검은색으로 변경
                image = Image.fromarray(image_array.astype(np.uint8))

            elif distortion == "high_frequency_noise":
                image = image.filter(ImageFilter.FIND_EDGES)

            elif distortion == "impulse_noise":
                image_array = np.array(image).astype(np.float32)
                prob = level
                mask = np.random.choice([0, 1], size=image_array.shape[:2], p=[1 - prob, prob])
                random_noise = np.random.choice([0, 255], size=(image_array.shape[0], image_array.shape[1], 1))
                image_array[mask == 1] = random_noise[mask == 1]
                image = Image.fromarray(image_array.astype(np.uint8))

            elif distortion == "quantization_noise":
                quantized = (np.array(image) // int(256 / level)) * int(256 / level)
                image = Image.fromarray(quantized.astype(np.uint8))

            elif distortion == "gaussian_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "image_denoising":
                image = image.filter(ImageFilter.MedianFilter(size=int(level)))

            elif distortion == "jpeg_compression":
                quality = max(1, min(100, int(100 - (level * 100))))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                return Image.open(buffer)

            elif distortion == "jpeg2000_compression":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))

            elif distortion == "contrast_change":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1 + level)

            elif distortion == "change_of_color_saturation":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1 + level)

            elif distortion == "multiplicative_gaussian_noise":
                image_array = np.array(image).astype(np.float32)
                noise = np.random.normal(1, level, image_array.shape)
                noisy_image = np.clip(image_array * noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "comfort_noise":
                image = image.filter(ImageFilter.EMBOSS)

            elif distortion == "lossy_compression_of_noisy_images":
                image = image.filter(ImageFilter.SMOOTH)

            elif distortion == "image_color_quantization_with_dither":
                image = image.convert("P", dither=Image.FLOYDSTEINBERG)

            elif distortion == "chromatic_aberrations":
                shift = int(level * 10)
                image_array = np.array(image)
                image_array[:, :, 0] = np.roll(image_array[:, :, 0], shift, axis=0)  # R 채널 이동
                image = Image.fromarray(image_array)

            elif distortion == "sparse_sampling_and_reconstruction":
                image = image.resize((image.width // int(10 * level), image.height // int(10 * level)))
                image = image.resize((image.width, image.height), Image.NEAREST)

            else:
                print(f"[Warning] Distortion type '{distortion}' not implemented.")

        except Exception as e:
            print(f"[Error] Applying distortion {distortion} with level {level}: {e}")

        return image

    def __getitem__(self, index: int):
        img_A = Image.open(self.image_paths[index]).convert("RGB")  
        img_B = Image.open(self.reference_paths[index]).convert("RGB")  

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
        """ ✅ 데이터셋 크기 반환 (🚨 이전 코드에서 없어서 오류 발생함) """
        return len(self.image_paths)
    
if __name__ == "__main__":
    """
    ✅ Hard Negative 적용 여부를 확인하고, 이미지 비교를 수행
    """
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013"

    synthetic_dataset = TID2013Dataset(root=dataset_path, phase="train", crop_size=224, dataset_type="synthetic")
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

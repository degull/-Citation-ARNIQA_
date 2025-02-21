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

class CSIQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224, dataset_type="synthetic"):
        """
        dataset_type: 
            "synthetic" (CSIQ) → Hard Negative 적용
            "authentic" (KonIQ-10k, SPAQ, LIVE-FB) → Hard Negative 적용 안함
        """
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.dataset_type = dataset_type  # ✅ 데이터셋 유형 결정

        # ✅ CSIQ 데이터셋 경로 설정
        scores_txt_path = os.path.join(self.root, "CSIQ.txt")
        if not os.path.isfile(scores_txt_path):
            raise FileNotFoundError(f"CSIQ TXT 파일이 {scores_txt_path} 경로에 존재하지 않습니다.")

        # ✅ CSV 파일 로드 (구분자 `,` 사용)
        scores_data = pd.read_csv(scores_txt_path, sep=',', names=["dist_img", "dist_type", "ref_img", "mos"], header=0)

        # 🔹 NaN 값 제거 후 문자열로 변환
        scores_data.dropna(inplace=True)
        scores_data = scores_data.astype(str)

        # ✅ 이미지 경로 설정
        self.image_paths = [os.path.join(self.root, img_path.replace("CSIQ/", "")) for img_path in scores_data["dist_img"]]
        self.reference_paths = [os.path.join(self.root, img_path.replace("CSIQ/", "")) for img_path in scores_data["ref_img"]]
        self.mos = scores_data["mos"].astype(float).values  # MOS 값을 float로 변환

        # ✅ CSIQ 데이터셋의 6개 왜곡 유형 (Hard Negative 적용 대상)
        self.distortion_types = ["jpeg", "jpeg2000", "blur", "awgn", "contrast", "fnoise"]

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):
        try:
            image = image.convert("RGB")  # Ensure the image is in RGB format

            if distortion == "jpeg":
                quality = max(1, min(100, int(100 - (level * 100))))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                return Image.open(buffer)

            elif distortion == "jpeg2000":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))

            elif distortion == "blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "awgn":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                noisy_image = image_array + noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1 + level)

            elif distortion == "fnoise":
                image_array = np.array(image).astype(np.float32)
                noise = np.random.normal(1, level, image_array.shape)
                noisy_image = image_array * noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            else:
                print(f"[Warning] Distortion type '{distortion}' not implemented.")

        except Exception as e:
            print(f"[Error] Applying distortion {distortion} with level {level}: {e}")

        return image

    def __getitem__(self, index: int):
        """
        ✅ 데이터셋 유형에 따라 `img_B` 처리 방식 변경 ✅
        - Synthetic 데이터셋(CSIQ) → Hard Negative 적용
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
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ"

    synthetic_dataset = CSIQDataset(root=dataset_path, phase="train", crop_size=224, dataset_type="synthetic")
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

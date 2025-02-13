""" import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import io
from PIL import ImageEnhance, ImageFilter, Image

# ✅ CSIQ 데이터셋의 왜곡 유형 매핑
distortion_types_mapping = {
    1: "jpeg",
    2: "jpeg2000",
    3: "blur",
    4: "awgn",
    5: "contrast",
    6: "fnoise"
}

# ✅ 강도 레벨 정의
def get_distortion_levels():
    return {
        'jpeg': [10, 20, 30, 40, 50],
        'jpeg2000': [10, 20, 30, 40, 50],
        'blur': [1, 2, 3, 4, 5],  # 기존 gaussian_blur
        'awgn': [5, 10, 15, 20, 25],  # 기존 white_noise
        'contrast': [0.5, 0.7, 0.9, 1.1, 1.3],  # 기존 contrast_change
        'fnoise': [1, 2, 3, 4, 5]  # 기존 foveated_compression
    }

class CSIQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # ✅ MOS 파일 확인 및 로드
        scores_csv_path = os.path.join(self.root, "CSIQ.txt")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSIQ.txt 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        # ✅ CSV 파일 로드 및 경로 변환
        scores_csv = pd.read_csv(scores_csv_path)

        # ✅ 이미지 경로 수정 (CSIQ/ 부분 제거)
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
            image = image.convert("RGB")

            if distortion == "jpeg":
                quality = max(1, min(100, 100 - level))
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
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_image)

            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)

            elif distortion == "fnoise":
                image = image.filter(ImageFilter.BoxBlur(level))

            else:
                print(f"[Warning] Distortion type '{distortion}' not implemented.")

        except Exception as e:
            print(f"[Error] Applying distortion {distortion} with level {level}: {e}")

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

        img_A_distorted = self.apply_distortion(img_A_orig, distortions[0], levels[0])
        img_B_distorted = self.apply_distortion(img_B_orig, distortions[0], levels[0])

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

# ✅ CSIQDataset 테스트
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
 """

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import io
from PIL import ImageEnhance, ImageFilter, Image

# ✅ CSIQ 데이터셋의 왜곡 유형 매핑
distortion_types_mapping = {
    "AWGN": "awgn",
    "BLUR": "blur",
    "contrast": "contrast",
    "fnoise": "fnoise",
    "JPEG": "jpeg",
    "jpeg2000": "jpeg2000"
}

# ✅ 강도 레벨 정의
def get_distortion_levels():
    return {
        'jpeg': [10, 20, 30, 40, 50],
        'jpeg2000': [10, 20, 30, 40, 50],
        'blur': [1, 2, 3, 4, 5],  
        'awgn': [5, 10, 15, 20, 25],  
        'contrast': [0.5, 0.7, 0.9, 1.1, 1.3],  
        'fnoise': [1, 2, 3, 4, 5]  
    }

class CSIQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # ✅ CSIQ.txt 파일 확인 및 로드
        scores_csv_path = os.path.join(self.root, "CSIQ.txt")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSIQ.txt 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        # ✅ CSV 파일 로드
        scores_csv = pd.read_csv(scores_csv_path)

        # ✅ 왜곡된 이미지 및 원본 이미지 경로 설정
        self.image_paths = [os.path.join(self.root, row["dis_img_path"].replace("CSIQ/", "")) for _, row in scores_csv.iterrows()]
        self.reference_paths = [os.path.join(self.root, row["ref_img_path"].replace("CSIQ/", "")) for _, row in scores_csv.iterrows()]
        self.distortion_types = [row["dis_type"] for _, row in scores_csv.iterrows()]
        self.mos = scores_csv["score"].values

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):
        """ 특정 왜곡을 이미지에 적용 """
        try:
            image = image.convert("RGB")

            if distortion == "jpeg":
                quality = max(1, min(100, 100 - level))
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
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_image)

            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)

            elif distortion == "fnoise":
                image = image.filter(ImageFilter.BoxBlur(level))

            else:
                print(f"[Warning] Distortion type '{distortion}' not implemented.")

        except Exception as e:
            print(f"[Error] Applying distortion {distortion} with level {level}: {e}")

        return image

    def __getitem__(self, index: int):
        try:
            img_A_orig = Image.open(self.image_paths[index]).convert("RGB")  # ✅ CSIQ의 기존 왜곡된 이미지
            img_B_orig = Image.open(self.reference_paths[index]).convert("RGB")  # ✅ CSIQ의 원본 이미지
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]} or {self.reference_paths[index]}: {e}")
            return None

        # ✅ img_A는 그대로 유지 (CSIQ에서 제공하는 왜곡된 이미지)
        img_A_transformed = self.transform(img_A_orig)

        # ✅ img_B: 원본 이미지에 새로운 Hard Negative 왜곡 적용 (DA 및 HNCA를 위한 추가 처리)
        distortions_B = random.sample(list(self.distortion_levels.keys()), 1)[0]
        level_B = random.choice(self.distortion_levels[distortions_B])

        print(f"[Debug] img_B Hard Negative: {distortions_B} (level: {level_B})")

        img_B_distorted = self.apply_distortion(img_B_orig, distortions_B, level_B)  # 원본에 새로운 왜곡 적용

        img_B_orig = self.transform(img_B_orig)
        img_B_distorted = self.transform(img_B_distorted)

        return {
            "img_A": img_A_transformed,  # 원래 왜곡된 이미지
            "img_B": img_B_distorted,  # Hard Negative 추가된 이미지
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }
    
    
    def __len__(self):
        return len(self.image_paths)

# ✅ CSIQDataset 테스트
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

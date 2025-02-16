""" import pandas as pd
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
    1: "low-light_noise",    # 저조도 노이즈
    2: "motion_blur",        # 움직임 블러
    3: "underexposure",      # 노출 부족
    4: "overexposure",       # 과노출
    5: "jpeg_artifacts",     # JPEG 압축 아티팩트
    6: "banding_artifacts",  # 색상 띠 노이즈
    7: "color_shift",        # 색상 이동
    8: "chromatic_aberration" # 색 수차
}


# 강도 레벨 정의
def get_distortion_levels():
    return {
        "low-light_noise": [5, 10, 15, 20, 25],
        "motion_blur": [1, 2, 3, 4, 5],
        "underexposure": [0.5, 0.7, 0.9, 1.1, 1.3],
        "overexposure": [0.5, 0.7, 0.9, 1.1, 1.3],
        "jpeg_artifacts": [10, 20, 30, 40, 50],
        "banding_artifacts": [1, 2, 3, 4, 5],
        "color_shift": [10, 20, 30, 40, 50],
        "chromatic_aberration": [1, 2, 3, 4, 5]
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
        image = image.convert("RGB")

        try:
            if distortion == "low-light_noise":
                image_array = np.array(image, dtype=np.float32) * (1 - level * 0.05)
                noise = np.random.normal(loc=0, scale=level * 10, size=image_array.shape).astype(np.float32)
                image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

            elif distortion == "motion_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "underexposure":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1 - level * 0.1)

            elif distortion == "overexposure":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1 + level * 0.1)

            elif distortion == "jpeg_artifacts":
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=max(1, min(100, 100 - level * 10)))
                buffer.seek(0)
                image = Image.open(buffer)

            elif distortion == "banding_artifacts":
                bands = np.linspace(0, 255, num=int(level * 10))
                image_array = np.array(image, dtype=np.float32)
                for i in range(image_array.shape[0]):
                    image_array[i, :, :] = bands[int((i / image_array.shape[0]) * len(bands))]
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            elif distortion == "color_shift":
                shift = np.random.randint(-level, level, (1, 1, 3))
                image_array = np.array(image).astype(np.float32) + shift
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            elif distortion == "chromatic_aberration":
                image_array = np.array(image, dtype=np.float32)
                shift = int(level * 5)
                image_array[:, :-shift, 0] = image_array[:, shift:, 0]
                image_array[:, shift:, 2] = image_array[:, :-shift, 2]
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

        except Exception as e:
            print(f"[Error] {distortion} 적용 중 오류 발생: {e}")

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
        print(f"Image B shape: {sample['img_B'].shape}") """


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
    1: "low-light_noise",    # 저조도 노이즈
    2: "motion_blur",        # 움직임 블러
    3: "underexposure",      # 노출 부족
    4: "overexposure",       # 과노출
    5: "jpeg_artifacts",     # JPEG 압축 아티팩트
    6: "banding_artifacts",  # 색상 띠 노이즈
    7: "color_shift",        # 색상 이동
    8: "chromatic_aberration" # 색 수차
}


# 강도 레벨 정의
def get_distortion_levels():
    return {
        "low-light_noise": [5, 10, 15, 20, 25],
        "motion_blur": [1, 2, 3, 4, 5],
        "underexposure": [0.5, 0.7, 0.9, 1.1, 1.3],
        "overexposure": [0.5, 0.7, 0.9, 1.1, 1.3],
        "jpeg_artifacts": [10, 20, 30, 40, 50],
        "banding_artifacts": [1, 2, 3, 4, 5],
        "color_shift": [10, 20, 30, 40, 50],
        "chromatic_aberration": [1, 2, 3, 4, 5]
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
        """ 지정된 왜곡 유형과 강도로 이미지를 변환 """
        image = image.convert("RGB")

        try:
            if distortion == "low-light_noise":
                image_array = np.array(image, dtype=np.float32) * (1 - level * 0.05)
                noise = np.random.normal(loc=0, scale=level * 10, size=image_array.shape).astype(np.float32)
                image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

            elif distortion == "motion_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "underexposure":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1 - level * 0.1)

            elif distortion == "overexposure":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1 + level * 0.1)

            elif distortion == "jpeg_artifacts":
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=max(1, min(100, 100 - level * 10)))
                buffer.seek(0)
                image = Image.open(buffer)

            elif distortion == "banding_artifacts":
                bands = np.linspace(0, 255, num=int(level * 10))
                image_array = np.array(image, dtype=np.float32)
                for i in range(image_array.shape[0]):
                    image_array[i, :, :] = bands[int((i / image_array.shape[0]) * len(bands))]
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            elif distortion == "color_shift":
                shift = np.random.randint(-level, level, (1, 1, 3))
                image_array = np.array(image).astype(np.float32) + shift
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            elif distortion == "chromatic_aberration":
                image_array = np.array(image, dtype=np.float32)
                shift = int(level * 5)
                image_array[:, :-shift, 0] = image_array[:, shift:, 0]
                image_array[:, shift:, 2] = image_array[:, :-shift, 2]
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

        except Exception as e:
            print(f"[Error] {distortion} 적용 중 오류 발생: {e}")

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
            img_A_orig = Image.open(image_path).convert("RGB")  # ✅ KONIQ10K 원본 이미지 사용
        except Exception as e:
            print(f"[Error] Loading image: {image_path}: {e}")
            return None

        # ✅ img_A는 원본 이미지 그대로 사용
        img_A_transformed = self.transform(img_A_orig)

        # ✅ img_B: Hard Negative 왜곡 추가
        distortions_B = random.sample(list(self.distortion_levels.keys()), 1)[0]
        level_B = random.choice(self.distortion_levels[distortions_B])

        print(f"[Debug] img_B Hard Negative: {distortions_B} (level: {level_B})")

        img_B_distorted = self.apply_distortion(img_A_orig, distortions_B, level_B)
        img_B_distorted = self.transform(img_B_distorted)

        return {
            "img_A": img_A_transformed,  # 원본 이미지
            "img_B": img_B_distorted,  # Hard Negative 추가된 이미지
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
""" import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image, ImageFilter, ImageEnhance
import io


# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "brightness",         # 밝기 변화 (Brightness)
    2: "exposure",           # 노출 (Exposure, 과노출 및 저노출 포함)
    3: "colorfulness",       # 색채감 변화 (Colorfulness)
    4: "contrast",           # 대비 변화 (Contrast)
    5: "noisiness",          # 노이즈 (Noisiness, Gaussian Noise, Impulse Noise 등 포함)
    6: "sharpness",          # 선명도 (Sharpness, Blur 관련 포함)
}



# SPAQ 데이터셋의 왜곡 유형 매핑 및 강도 레벨 정의
def get_distortion_levels():
    return {
        "brightness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 밝기 감소 및 증가
        "exposure": [0.5, 0.7, 0.9, 1.1, 1.3],  # 노출 감소 및 증가 (과노출/저노출)
        "colorfulness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 색채감 변화
        "contrast": [0.5, 0.7, 0.9, 1.1, 1.3],  # 대비 감소 및 증가
        "noisiness": [5, 10, 15, 20, 25],  # 노이즈 강도
        "sharpness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 선명도 감소 및 증가
    }




class SPAQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "Annotations", "MOS and Image attribute scores.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)
        self.images = scores_csv["Image name"].values
        self.mos = scores_csv["MOS"].values

        self.image_paths = [
            os.path.join(self.root, "TestImage", img) for img in self.images
        ]

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):

        try:
            image = image.convert("RGB")  # RGB 변환

            # 1. 밝기(Brightness) 조절
            if distortion == "brightness":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(level)

            # 2. 노출(Exposure) 조절
            elif distortion == "exposure":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(level)

            # 3. 색감(Colorfulness) 조절
            elif distortion == "colorfulness":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(level)

            # 4. 대비(Contrast) 조절
            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)


            # 5. 일반 노이즈(Noise) 추가
            elif distortion == "noisiness":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

            # 6. 선명도(Sharpness) 조절
            elif distortion == "sharpness":
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(level)


            else:
                print(f"[Warning] '{distortion}' 왜곡 유형이 구현되지 않았습니다.")

        except Exception as e:
            print(f"[Error] '{distortion}' 왜곡 적용 중 오류 발생: {e}")

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
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        # 📌 img_B를 reference 이미지로 설정 → img_A에서 랜덤한 왜곡 적용
        distortions_A = random.sample(list(self.distortion_levels.keys()), 1)[0]
        level_A = random.choice(self.distortion_levels[distortions_A])

        distortions_B = random.sample(list(self.distortion_levels.keys()), 1)[0]
        level_B = random.choice(self.distortion_levels[distortions_B])

        print(f"[Debug] img_A: {distortions_A} (level: {level_A}), img_B: {distortions_B} (level: {level_B})")

        img_A_distorted = self.apply_distortion(img_A_orig, distortions_A, level_A)
        img_B_distorted = self.apply_distortion(img_A_orig, distortions_B, level_B)  # 다른 왜곡 적용

        img_A_orig = self.transform(img_A_orig)
        img_A_distorted = self.transform(img_A_distorted)
        img_B_distorted = self.transform(img_B_distorted)

        return {
            "img_A": torch.stack([img_A_orig, img_A_distorted]),
            "img_B": torch.stack([img_A_orig, img_B_distorted]),
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    

    def __len__(self):
        return len(self.images)


# SPAQDataset 테스트
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ"
    dataset = SPAQDataset(root=dataset_path, crop_size=224)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"MOS score: {sample['mos']}")
        print(f"Image A shape: {sample['img_A'].shape}")
        print(f"Image B shape: {sample['img_B'].shape}") """


import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image, ImageFilter, ImageEnhance
import io


# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "brightness",         # 밝기 변화 (Brightness)
    2: "exposure",           # 노출 (Exposure, 과노출 및 저노출 포함)
    3: "colorfulness",       # 색채감 변화 (Colorfulness)
    4: "contrast",           # 대비 변화 (Contrast)
    5: "noisiness",          # 노이즈 (Noisiness, Gaussian Noise, Impulse Noise 등 포함)
    6: "sharpness",          # 선명도 (Sharpness, Blur 관련 포함)
}



# SPAQ 데이터셋의 왜곡 유형 매핑 및 강도 레벨 정의
def get_distortion_levels():
    return {
        "brightness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 밝기 감소 및 증가
        "exposure": [0.5, 0.7, 0.9, 1.1, 1.3],  # 노출 감소 및 증가 (과노출/저노출)
        "colorfulness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 색채감 변화
        "contrast": [0.5, 0.7, 0.9, 1.1, 1.3],  # 대비 감소 및 증가
        "noisiness": [5, 10, 15, 20, 25],  # 노이즈 강도
        "sharpness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 선명도 감소 및 증가
    }




class SPAQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        # CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "Annotations", "MOS and Image attribute scores.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)
        self.images = scores_csv["Image name"].values
        self.mos = scores_csv["MOS"].values

        self.image_paths = [
            os.path.join(self.root, "TestImage", img) for img in self.images
        ]

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image, distortion, level):

        try:
            image = image.convert("RGB")  # RGB 변환

            # 1. 밝기(Brightness) 조절
            if distortion == "brightness":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(level)

            # 2. 노출(Exposure) 조절
            elif distortion == "exposure":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(level)

            # 3. 색감(Colorfulness) 조절
            elif distortion == "colorfulness":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(level)

            # 4. 대비(Contrast) 조절
            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)


            # 5. 일반 노이즈(Noise) 추가
            elif distortion == "noisiness":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

            # 6. 선명도(Sharpness) 조절
            elif distortion == "sharpness":
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(level)


            else:
                print(f"[Warning] '{distortion}' 왜곡 유형이 구현되지 않았습니다.")

        except Exception as e:
            print(f"[Error] '{distortion}' 왜곡 적용 중 오류 발생: {e}")

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
        """ 데이터셋에서 index 번째 샘플을 반환 """
        try:
            img_A_orig = Image.open(self.image_paths[index]).convert("RGB")  # ✅ 원본 이미지 유지
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        # ✅ `img_A`: 원본 이미지 그대로 유지
        img_A_transformed = self.transform(img_A_orig)

        # ✅ `img_B`: Hard Negative 추가 (무작위 왜곡)
        distortion_B = random.choice(list(self.distortion_levels.keys()))
        level_B = random.choice(self.distortion_levels[distortion_B])

        print(f"[Debug] img_B: {distortion_B} (level: {level_B})")

        img_B_distorted = self.apply_distortion(img_A_orig, distortion_B, level_B)  # 원본에 새로운 왜곡 적용
        img_B_distorted = self.transform(img_B_distorted)

        return {
            "img_A": img_A_transformed,  # ✅ 원래 왜곡된 이미지
            "img_B": img_B_distorted,  # ✅ Hard Negative 추가된 이미지
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }



    def __len__(self):
        return len(self.images)


# SPAQDataset 테스트
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ"
    dataset = SPAQDataset(root=dataset_path, crop_size=224)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"MOS score: {sample['mos']}")
        print(f"Image A shape: {sample['img_A'].shape}")
        print(f"Image B shape: {sample['img_B'].shape}")
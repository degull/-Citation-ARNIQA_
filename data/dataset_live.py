import os
import random
import torch
import numpy as np
import io
from PIL import Image, ImageFilter, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# 손상된 이미지 로드 방지
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_distortion_levels():
    """ LIVE 데이터셋의 왜곡 유형과 강도 정의 """
    return {
        'jpeg': [10, 20, 30, 40, 50],
        'jpeg2000': [10, 20, 30, 40, 50],
        'gaussian_blur': [1, 2, 3, 4, 5],
        'white_noise': [5, 10, 15, 20, 25],
        'fast_fading': [1, 2, 3, 4, 5]
    }

class LIVEDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = root
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        images_txt_path = os.path.join(self.root, "Data", "AllImages_release.txt")
        mos_txt_path = os.path.join(self.root, "Data", "AllMOS_release.txt")

        if not os.path.isfile(images_txt_path):
            raise FileNotFoundError(f"{images_txt_path} 파일이 존재하지 않습니다.")
        if not os.path.isfile(mos_txt_path):
            raise FileNotFoundError(f"{mos_txt_path} 파일이 존재하지 않습니다.")

        # 이미지 파일 목록 로드
        with open(images_txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip().strip("['']") for line in f.readlines()[1:] if line.strip()]
        self.image_paths = [os.path.normpath(os.path.join(self.root, "Images", line)) for line in lines]

        # 손상된 이미지 필터링
        self.image_paths = [path for path in self.image_paths if os.path.isfile(path)]

        # MOS 점수 로드
        with open(mos_txt_path, "r", encoding="utf-8") as f:
            mos_lines = f.readlines()
        mos_scores = mos_lines[1].strip().split()
        self.mos = np.array([float(x) for x in mos_scores])

        min_size = min(len(self.image_paths), len(self.mos))
        self.image_paths = self.image_paths[:min_size]
        self.mos = self.mos[:min_size]

        print(f"[INFO] 이미지 개수: {len(self.image_paths)}, MOS 점수 개수: {len(self.mos)}")

        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    def apply_distortion(self, image, distortion, level):
        """ 지정된 왜곡 유형과 강도로 이미지를 변환 """
        image = image.convert("RGB")

        try:
            if distortion == "jpeg":
                quality = max(1, min(100, int(100 - level)))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                image = Image.open(buffer)

            elif distortion == "jpeg2000":
                image = image.resize((image.width // 2, image.height // 2)).resize((image.width, image.height))

            elif distortion == "gaussian_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "white_noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_image)

            elif distortion == "fast_fading":
                image = image.filter(ImageFilter.BoxBlur(level))

        except Exception as e:
            print(f"[Error] Applying distortion {distortion} with level {level}: {e}")

        return image

    def __getitem__(self, index: int):
        """ 데이터셋의 index 번째 샘플을 반환 """
        try:
            img_A_orig = Image.open(self.image_paths[index]).convert("RGB")
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        # 랜덤 왜곡 적용
        distortion = random.choice(list(self.distortion_levels.keys()))
        level = random.choice(self.distortion_levels[distortion])

        print(f"[Debug] Selected Distortion: {distortion}, Level: {level}")

        img_A_distorted = self.apply_distortion(img_A_orig, distortion, level)

        img_A_orig = self.transform(img_A_orig)
        img_A_distorted = self.transform(img_A_distorted)

        return {
            "img_A": img_A_orig,
            "img_B": img_A_distorted,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/LIVE/"
    dataset = LIVEDataset(root=dataset_path, phase="train", crop_size=224)

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    if sample:
        print(f"Sample keys: {sample.keys()}")
        print(f"Sample MOS score: {sample['mos']}")

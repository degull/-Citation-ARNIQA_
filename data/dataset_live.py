import os
import random
import torch
import numpy as np
import io
from PIL import Image, ImageFilter, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageFile, ImageEnhance
import pandas as pd

# 손상된 이미지 로드 방지
ImageFile.LOAD_TRUNCATED_IMAGES = True

distortion_types_mapping = {
    1: "brightness",  # 밝기 조절
    2: "exposure",    # 과노출 / 저노출
    3: "colorfulness",  # 색감 변화
    4: "color_shift",  # 색상 이동
    5: "white_balance_error", # 화이트 밸런스 오류
    6: "sharpness",   # 선명도 문제
    7: "motion_blur",  # 움직임 블러
    8: "contrast",    # 대비 문제
    9: "glare",       # 조명 반사
    10: "haze",       # 흐려짐 (안개)
    11: "noise",      # 일반 노이즈
    12: "low-light_noise", # 저조도 노이즈
    13: "color_noise", # 색상 노이즈
    14: "jpeg_artifacts", # JPEG 압축 아티팩트
    15: "banding_artifacts", # 색상 띠 노이즈
    16: "vignetting",  # 비네팅 (주변부 어두움)
    17: "chromatic_aberration", # 색 수차
    18: "distortion"  # 렌즈 기하학적 왜곡
}



def get_distortion_levels():
    """ LIVE-Challenge 데이터셋의 왜곡 유형과 강도 정의 """
    return {
        "brightness": [0.5, 0.7, 0.9, 1.1, 1.3],
        "exposure": [0.5, 0.7, 0.9, 1.1, 1.3],
        "colorfulness": [0.5, 0.7, 0.9, 1.1, 1.3],
        "color_shift": [10, 20, 30, 40, 50],
        "sharpness": [0.5, 0.7, 0.9, 1.1, 1.3],
        "motion_blur": [1, 2, 3, 4, 5],
        "contrast": [0.5, 0.7, 0.9, 1.1, 1.3],
        "glare": [1, 2, 3, 4, 5],
        "haze": [1, 2, 3, 4, 5],
        "noise": [5, 10, 15, 20, 25],
        "low-light_noise": [5, 10, 15, 20, 25],
        "color_noise": [5, 10, 15, 20, 25],
        "jpeg_artifacts": [10, 20, 30, 40, 50],
        "banding_artifacts": [1, 2, 3, 4, 5],
        "vignetting": [0.5, 0.7, 0.9, 1.1, 1.3],
        "chromatic_aberration": [1, 2, 3, 4, 5],
        "distortion": [1, 2, 3, 4, 5]
    }

class LIVEDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = root
        self.phase = phase
        self.crop_size = crop_size
        self.distortion_levels = get_distortion_levels()

        images_csv_path = os.path.join(self.root, "LIVE_Challenge.txt")
        images_dir = os.path.join(self.root, "Images")  # ✅ 올바른 이미지 폴더 경로 설정

        if not os.path.isfile(images_csv_path):
            raise FileNotFoundError(f"[Error] {images_csv_path} 파일이 존재하지 않습니다.")

        # CSV 파일 로드
        df = pd.read_csv(images_csv_path)

        # 이미지 및 MOS 점수 로드
        self.image_paths = []
        self.mos = []

        for index, row in df.iterrows():
            image_rel_path = row["dis_img_path"].strip()
            mos_score = row["score"]

            # ✅ `LIVE_Challenge/Images/` 부분 제거하고 올바른 경로로 설정
            image_name = os.path.basename(image_rel_path)  # "10.bmp" 등으로 변환
            image_path = os.path.join(images_dir, image_name)  # "E:/.../Images/10.bmp"로 변환

            if os.path.isfile(image_path):
                self.image_paths.append(image_path)
                self.mos.append(float(mos_score))
            else:
                print(f"[Warning] 이미지 파일을 찾을 수 없음: {image_path}")

        if len(self.image_paths) == 0:
            raise ValueError(f"[Error] 이미지 파일이 존재하지 않습니다. {images_csv_path} 내용을 확인하세요.")

        print(f"[INFO] 로드된 이미지 개수: {len(self.image_paths)}, MOS 점수 개수: {len(self.mos)}")

        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    def apply_distortion(self, image, distortion, level):
        """ 지정된 왜곡 유형과 강도로 이미지를 변환 """
        image = image.convert("RGB")

        try:
            if distortion == "brightness":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(level)

            elif distortion == "exposure":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(level)

            elif distortion == "colorfulness":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(level)

            elif distortion == "color_shift":
                shift = np.random.randint(-level, level, (1, 1, 3))
                image_array = np.array(image).astype(np.float32) + shift
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            elif distortion == "sharpness":
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(level)

            elif distortion == "motion_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)

            elif distortion == "glare":
                height, width = image.size
                glare_mask = np.linspace(1, level, width).reshape(1, width)  # (1, W) 형태로 변환
                glare_mask = np.tile(glare_mask, (height, 1))  # (H, W) 형태로 변환
                glare_mask = glare_mask[:, :, np.newaxis]  # (H, W, 1) 형태로 변환

                image_array = np.array(image, dtype=np.float32) * glare_mask
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))


            elif distortion == "haze":
                haze_level = level * 255
                image_array = np.array(image, dtype=np.float32) + haze_level
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            elif distortion == "noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

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

            elif distortion == "vignetting":
                width, height = image.size
                x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
                v_mask = 1 - (x**2 + y**2) * level
                v_mask = np.clip(v_mask, 0, 1)
                image_array = np.array(image, dtype=np.float32) * v_mask[:, :, np.newaxis]
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

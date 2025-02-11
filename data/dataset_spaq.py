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


# SPAQ 데이터셋의 왜곡 유형 매핑 및 강도 레벨 정의
def get_distortion_levels():
    return {
        "brightness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 밝기 조절
        "exposure": [0.5, 0.7, 0.9, 1.1, 1.3],  # 과노출 / 저노출
        "colorfulness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 색감 변화
        "color_shift": [10, 20, 30, 40, 50],  # 색 이동
        "white_balance_error": [0.5, 0.7, 0.9, 1.1, 1.3],  # 화이트 밸런스 오류
        "sharpness": [0.5, 0.7, 0.9, 1.1, 1.3],  # 선명도
        "motion_blur": [1, 2, 3, 4, 5],  # 움직임 블러
        "contrast": [0.5, 0.7, 0.9, 1.1, 1.3],  # 대비
        "glare": [1, 2, 3, 4, 5],  # 조명 반사
        "haze": [1, 2, 3, 4, 5],  # 흐려짐
        "noise": [5, 10, 15, 20, 25],  # 일반 노이즈
        "low-light_noise": [5, 10, 15, 20, 25],  # 저조도 노이즈
        "color_noise": [5, 10, 15, 20, 25],  # 색상 노이즈
        "jpeg_artifacts": [10, 20, 30, 40, 50],  # JPEG 압축 아티팩트
        "banding_artifacts": [1, 2, 3, 4, 5],  # 색상 띠 노이즈
        "vignetting": [0.5, 0.7, 0.9, 1.1, 1.3],  # 비네팅
        "chromatic_aberration": [1, 2, 3, 4, 5],  # 색 수차
        "distortion": [1, 2, 3, 4, 5]  # 렌즈 기하학적 왜곡
    }



class SPAQDataset(Dataset):
    def __init__(self, root: str, crop_size: int = 224):
        super().__init__()
        self.root = str(root)
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



    def apply_distortion(image, distortion, level):

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

            # 4. 색상 이동(Color Shift)
            elif distortion == "color_shift":
                shift = np.random.randint(-level, level, (1, 1, 3))  # RGB 채널별 무작위 이동
                image_array = np.array(image).astype(np.float32) + shift
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # 5. 화이트 밸런스 오류 (White Balance Error)
            elif distortion == "white_balance_error":
                r, g, b = np.random.uniform(0.8, 1.2, 3)  # RGB 색상 채널에 무작위 보정 적용
                image_array = np.array(image).astype(np.float32)
                image_array[:, :, 0] *= r
                image_array[:, :, 1] *= g
                image_array[:, :, 2] *= b
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # 6. 선명도(Sharpness) 조절
            elif distortion == "sharpness":
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(level)

            # 7. 움직임 블러(Motion Blur)
            elif distortion == "motion_blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=level))

            # 8. 대비(Contrast) 조절
            elif distortion == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(level)

            # 9. 조명 반사(Glare)
            elif distortion == "glare":
                glare_mask = np.linspace(1, level, image.width)
                image_array = np.array(image, dtype=np.float32) * glare_mask
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # 10. 흐려짐 (Haze)
            elif distortion == "haze":
                haze_level = level * 255  # 안개 효과를 위한 밝기 증가
                image_array = np.array(image, dtype=np.float32) + haze_level
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # 11. 일반 노이즈(Noise) 추가
            elif distortion == "noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=image_array.shape).astype(np.float32)
                image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

            # 12. 저조도 노이즈(Low-light Noise)
            elif distortion == "low-light_noise":
                image_array = np.array(image, dtype=np.float32) * (1 - level * 0.2)
                noise = np.random.normal(loc=0, scale=level * 50, size=image_array.shape).astype(np.float32)
                image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

            # 13. 색상 노이즈(Color Noise)
            elif distortion == "color_noise":
                image_array = np.array(image, dtype=np.float32)
                noise = np.random.normal(loc=0, scale=level * 255, size=(image_array.shape[0], image_array.shape[1], 1))
                image_array += noise
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # 14. JPEG 압축 아티팩트(JPEG Artifacts)
            elif distortion == "jpeg_artifacts":
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=max(1, min(100, 100 - level * 10)))
                buffer.seek(0)
                image = Image.open(buffer)

            # 15. 밴딩 노이즈(Banding Artifacts)
            elif distortion == "banding_artifacts":
                bands = np.linspace(0, 255, num=int(level * 10))
                image_array = np.array(image, dtype=np.float32)
                for i in range(image_array.shape[0]):
                    image_array[i, :, :] = bands[int((i / image_array.shape[0]) * len(bands))]
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # 16. 비네팅(Vignetting)
            elif distortion == "vignetting":
                width, height = image.size
                x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
                v_mask = 1 - (x**2 + y**2) * level
                v_mask = np.clip(v_mask, 0, 1)
                image_array = np.array(image, dtype=np.float32) * v_mask[:, :, np.newaxis]
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # 17. 색 수차(Chromatic Aberration)
            elif distortion == "chromatic_aberration":
                image_array = np.array(image, dtype=np.float32)
                shift = int(level * 5)
                image_array[:, :-shift, 0] = image_array[:, shift:, 0]  # Red 채널 이동
                image_array[:, shift:, 2] = image_array[:, :-shift, 2]  # Blue 채널 이동
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # 18. 기하학적 왜곡(Distortion)
            elif distortion == "distortion":
                width, height = image.size
                x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
                r = np.sqrt(x**2 + y**2)
                distortion_map = 1 + level * np.sin(r * np.pi)
                image_array = np.array(image, dtype=np.float32) * distortion_map[:, :, np.newaxis]
                image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

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
            #print(f"[Debug] Applying distortion: {distortion} with level: {level}")
            try:
                image = self.apply_distortion(image, distortion, level)
            except Exception as e:
                #print(f"[Error] Applying distortion {distortion} with level {level}: {e}")
                continue
        return image
    

    def __getitem__(self, index: int):
        try:
            img_orig = Image.open(self.image_paths[index]).convert("RGB")
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        distortions = random.sample(list(self.distortion_levels.keys()), 1)
        levels = [random.choice(self.distortion_levels[distortions[0]])]

        img_distorted = self.apply_random_distortions(img_orig, distortions, levels)

        img_orig = self.transform(img_orig)
        img_distorted = self.transform(img_distorted)

        return {
            "img_A": img_orig,
            "img_B": img_distorted,
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
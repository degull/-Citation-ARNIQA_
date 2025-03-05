import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class TID2013Dataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        """
        ✅ DistortionDetectionModel에 적합하도록 데이터셋 수정
        - `img_A`(왜곡된 이미지)만 반환
        - `img_B`(참조 이미지) 제거
        - `mos`(Mean Opinion Score) 점수 반환
        """
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size

        # ✅ MOS CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "mos.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"TID2013 MOS CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_data = pd.read_csv(scores_csv_path)

        # ✅ 이미지 경로 설정 (img_A만 사용)
        self.image_paths = [os.path.join(self.root, "distorted_images", img) for img in scores_data["image_id"]]
        self.mos = scores_data["mean"].astype(float).values  # MOS 값을 float로 변환

        print(f"[Debug] Phase: {self.phase}")
        print(f"[Debug] Total Records: {len(self.image_paths)}")

        # ✅ 기본 이미지 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index: int):
        """
        ✅ `img_A`(왜곡된 이미지)와 `mos`(Mean Opinion Score)만 반환
        """
        try:
            img_A = Image.open(self.image_paths[index]).convert("RGB")  # ✅ 원본 이미지 사용
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        img_A_transformed = self.transform(img_A)  # ✅ 변환 적용

        return {
            "img_A": img_A_transformed,  # ✅ 원본 이미지
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":
    """
    ✅ TID2013은 Synthetic 데이터셋이지만, `DistortionDetectionModel`과 호환되도록 Hard Negative 없이 원본 이미지만 사용.
    """
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013"

    dataset = TID2013Dataset(root=dataset_path, phase="train", crop_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"TID2013 Dataset size: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
    print(f"Sample MOS Scores: {sample_batch['mos']}")

    

# 사용 X
""" 
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import laplace


# ✅ Distortion Type을 파일명에서 추출하는 함수
def get_distortion_type(image_id):
    return int(image_id.split("_")[1])  # 예: 'i01_01_1.png' → distortion type = 01


class TID2013Dataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size

        # ✅ MOS CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "mos.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"TID2013 MOS CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_data = pd.read_csv(scores_csv_path)

        # ✅ MOS 값 로드 및 float 변환
        self.mos = scores_data["mean"].astype(float).values

        # ✅ MOS 값 정규화 (0~1 범위)
        mos_min, mos_max = np.min(self.mos), np.max(self.mos)
        if mos_max - mos_min == 0:
            raise ValueError("[Error] MOS 값의 최소값과 최대값이 동일하여 정규화할 수 없습니다.")
        self.mos = (self.mos - mos_min) / (mos_max - mos_min)

        # ✅ 이미지 파일 경로 설정
        self.image_paths = [os.path.join(self.root, "distorted_images", img) for img in scores_data["image_id"]]

        # ✅ Distortion Type 추출 (파일명 기반)
        self.distortion_types = [get_distortion_type(img) for img in scores_data["image_id"]]

        # ✅ VGG-16 Feature Extractor 정의 (Feature Similarity 기반 샘플링용)
        self.feature_extractor = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23]  # Conv4-3까지 사용
        self.feature_extractor.eval()

        # ✅ 기본 이미지 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    # ✅ Perceptual Attributes 계산 (밝기, 채도, 노이즈, 블러)
    def calculate_perceptual_attributes(self, img):
        img_np = np.array(img)

        # ✅ 밝기(Brightness)
        gray = rgb2gray(img_np)
        brightness = np.mean(gray)

        # ✅ 채도(Colorfulness)
        rg = np.absolute(img_np[:, :, 0] - img_np[:, :, 1])
        yb = np.absolute(0.5 * (img_np[:, :, 0] + img_np[:, :, 1]) - img_np[:, :, 2])
        colorfulness = np.mean(rg) + np.mean(yb)

        # ✅ 노이즈(Noisiness)
        noise = np.var(laplace(gray))

        # ✅ 블러(Sharpness)
        blur = np.mean(np.abs(laplace(gray)))

        return torch.tensor([brightness, colorfulness, noise, blur], dtype=torch.float32)

    # ✅ Feature Similarity 기반 Hard Negative 샘플링 (VGG-16 Feature Extractor 사용)
    def extract_features(self, img):
        img_tensor = img.unsqueeze(0)  # ✅ (1, C, H, W) 형태로 변환
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        return features.view(-1)  # ✅ Flatten

    def __getitem__(self, index: int):

        try:
            img_A = Image.open(self.image_paths[index]).convert("RGB")  # ✅ 왜곡된 이미지 로드
        except Exception as e:
            print(f"[Error] 이미지 로드 실패: {self.image_paths[index]}: {e}")
            return None

        img_A_transformed = self.transform(img_A)  # ✅ 변환 적용

        # ✅ Perceptual Attributes 계산
        perceptual_attrs = self.calculate_perceptual_attributes(img_A)

        # ✅ Feature Embedding 계산
        feature_embedding = self.extract_features(img_A_transformed)

        return {
            "img_A": img_A_transformed,  # ✅ 왜곡된 이미지
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),  # ✅ MOS 점수
            "perceptual_attrs": perceptual_attrs,  # ✅ Perceptual Features (밝기, 채도, 노이즈, 블러)
            "feature_embedding": feature_embedding,  # ✅ Feature Similarity 기반 벡터
            "distortion_type": torch.tensor(self.distortion_types[index], dtype=torch.int64)  # ✅ Distortion Type
        }

    def __len__(self):
        return len(self.image_paths)


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":

    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013"

    dataset = TID2013Dataset(root=dataset_path, phase="train", crop_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"TID2013 Dataset size: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
    print(f"Sample MOS Scores: {sample_batch['mos']}")
    print(f"Sample Perceptual Attributes: {sample_batch['perceptual_attrs']}")  # ✅ Perceptual Attributes 출력
    print(f"Sample Feature Embedding Shape: {sample_batch['feature_embedding'].shape}")  # ✅ Feature Embedding 출력
    print(f"Sample Distortion Type: {sample_batch['distortion_type']}")  # ✅ Distortion Type 출력
 """
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CSIQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size

        # ✅ CSIQ 데이터셋 경로 설정
        scores_txt_path = os.path.join(self.root, "CSIQ.txt")
        if not os.path.isfile(scores_txt_path):
            raise FileNotFoundError(f"CSIQ TXT 파일이 {scores_txt_path} 경로에 존재하지 않습니다.")

        # ✅ CSV 파일 로드 (구분자 `,` 사용)
        scores_data = pd.read_csv(scores_txt_path, sep=',', names=["dist_img", "dist_type", "ref_img", "mos"], header=0)

        # 🔹 NaN 값 제거 후 문자열로 변환
        scores_data.dropna(inplace=True)
        scores_data = scores_data.astype(str)

        # ✅ 이미지 경로 설정 (img_A만 사용)
        self.image_paths = [os.path.join(self.root, img_path.replace("CSIQ/", "")) for img_path in scores_data["dist_img"]]
        self.mos = scores_data["mos"].astype(float).values  # MOS 값을 float로 변환

        # ✅ MOS 값 정규화 (0~1 범위)
        mos_min = np.min(self.mos)
        mos_max = np.max(self.mos)
        if mos_max - mos_min == 0:
            raise ValueError("[Error] MOS 값의 최소값과 최대값이 동일하여 정규화할 수 없습니다.")

        self.mos = (self.mos - mos_min) / (mos_max - mos_min)
        print(f"[Check] MOS 최소값: {np.min(self.mos)}, 최대값: {np.max(self.mos)}")

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __getitem__(self, index: int):
        img_A = Image.open(self.image_paths[index]).convert("RGB")  
        img_A = self.transform(img_A)

        return {
            "img_A": img_A,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ"

    dataset = CSIQDataset(root=dataset_path, phase="train", crop_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
    print(f"Sample MOS Scores: {sample_batch['mos']}")


""" 
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ✅ 동일한 정규화 적용
common_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ MOS 값 정규화 함수
def normalize_mos(mos_values):
    mos_values = np.array(mos_values).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(mos_values).flatten()

class CSIQDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.root = str(root)

        # ✅ CSIQ 데이터셋 경로 설정
        scores_txt_path = os.path.join(self.root, "CSIQ.txt")
        if not os.path.isfile(scores_txt_path):
            raise FileNotFoundError(f"CSIQ TXT 파일이 {scores_txt_path} 경로에 존재하지 않습니다.")

        # ✅ CSV 파일 로드 (구분자 `\t` 또는 `,` 확인 필요)
        try:
            scores_data = pd.read_csv(scores_txt_path, sep=',', names=["dist_img", "dist_type", "ref_img", "mos"], header=0)
        except pd.errors.ParserError:
            print("⚠️ CSV 파일 파싱 실패. 구분자를 `\t`로 변경하여 다시 시도합니다.")
            scores_data = pd.read_csv(scores_txt_path, sep='\t', names=["dist_img", "dist_type", "ref_img", "mos"], header=0)

        scores_data.dropna(inplace=True)

        # ✅ 이미지 경로 설정 (중복된 "CSIQ/" 제거)
        self.image_paths = [os.path.join(self.root, img_path.strip().replace("CSIQ/", "")) for img_path in scores_data["dist_img"]]

        # ✅ MOS 값 정규화
        self.mos = normalize_mos(scores_data["mos"].astype(float).values)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]

        # 🔹 파일이 존재하는지 확인 (디버깅용)
        if not os.path.exists(img_path):
            print(f"⚠️ 파일이 존재하지 않음: {img_path}")

        img_A = Image.open(img_path).convert("RGB")
        img_A = common_transforms(img_A)  # ✅ 동일한 정규화 적용

        return {
            "img_A": img_A,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ"

    dataset = CSIQDataset(root=dataset_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"✅ CSIQ 데이터셋 크기: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"🔹 샘플 이미지 크기: {sample_batch['img_A'].shape}")
    print(f"🔹 샘플 MOS 점수: {sample_batch['mos']}")
    print(f"🔹 MOS 범위: {sample_batch['mos'].min().item()} ~ {sample_batch['mos'].max().item()}")

    print("🚀 **CSIQ 데이터셋 테스트 완료!** 🚀")
 """
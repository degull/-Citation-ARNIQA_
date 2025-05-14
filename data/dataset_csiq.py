import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class CSIQDataset(Dataset):
    def __init__(self, root: str, phase: str = "all", crop_size: int = 224):
        super().__init__()
        self.root = root
        self.phase = phase.lower()
        self.crop_size = crop_size

        # ✅ CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "CSIQ.txt")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_csv = pd.read_csv(scores_csv_path)

        # ✅ 이미지 이름과 MOS 값 가져오기
        self.images = scores_csv["dis_img_path"].values
        self.mos = scores_csv["score"].values.astype(np.float32)
        self.sets = scores_csv["set"].values if "set" in scores_csv.columns else ["all"] * len(scores_csv)

        # ✅ MOS 값 검사 및 정리
        print(f"[Check] 총 MOS 값 개수: {len(self.mos)}")
        print(f"[Check] NaN 개수: {np.isnan(self.mos).sum()}, Inf 개수: {np.isinf(self.mos).sum()}")

        if np.isnan(self.mos).sum() > 0 or np.isinf(self.mos).sum() > 0:
            self.mos = np.nan_to_num(self.mos, nan=0.5, posinf=1.0, neginf=0.0)

        # ✅ MOS 값 정규화 (0~1 범위)
        self.mos = (self.mos - np.min(self.mos)) / (np.max(self.mos) - np.min(self.mos))
        print(f"[Check] MOS 최소값: {np.min(self.mos)}, 최대값: {np.max(self.mos)}")

        # ✅ CSV 'set' 컬럼의 실제 값 확인
        print("CSV 'set' 컬럼에 들어 있는 값 종류:", set(self.sets))

        # ✅ 데이터 필터링 (train, test, val 구분이 없는 경우 전체 사용)
        print(f"[Debug] 데이터 필터링 전 이미지 개수: {len(self.images)}")

        if "all" in set(self.sets):  # 'all' 값이 있는 경우 모든 데이터를 사용
            print("[Info] 'set' 컬럼이 'all'이므로 모든 데이터를 사용합니다.")
        else:
            if self.phase != "all":
                indices = [i for i, s in enumerate(self.sets) if s.strip().lower() == self.phase]
                print(f"[Debug] '{self.phase}'에 해당하는 데이터 개수: {len(indices)}")

                if len(indices) == 0:
                    raise ValueError(f"'{self.phase}'에 해당하는 데이터가 없습니다. CSV 'set' 컬럼 값을 확인하세요.")

                self.images = self.images[indices]
                self.mos = self.mos[indices]

        # ✅ 올바른 이미지 경로 생성
        self.image_paths = [os.path.join(self.root, img.replace("CSIQ/", "").replace("\\", "/")) for img in self.images]

        print(f"[Debug] Phase: {self.phase}")
        print(f"[Debug] Total Records: {len(self.image_paths)}")

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mos = self.mos[index]

        try:
            img_A = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Error] 이미지 로드 실패: {image_path}: {e}")
            return None

        img_A_transformed = self.transform(img_A)

        return {
            "img_A": img_A_transformed,
            "mos": torch.tensor(mos, dtype=torch.float32),
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
    if sample_batch is not None:
        print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
        print(f"Sample MOS Scores: {sample_batch['mos']}")
    else:
        print("[Error] 데이터가 로드되지 않았습니다.")


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
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CSIQDataset(Dataset):
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

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __getitem__(self, index: int):
        """
        ✅ `img_A`(왜곡된 이미지)와 `mos`(Mean Opinion Score)만 반환
        """
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

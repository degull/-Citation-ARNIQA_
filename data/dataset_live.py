import os
import torch
import numpy as np
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LIVEDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        """
        ✅ LIVE-Challenge 데이터셋 (Authentic Dataset)  
        - img_A: 원본 이미지  
        - img_B: 원본 이미지 (Hard Negative X)  
        """
        super().__init__()
        self.root = root
        self.phase = phase
        self.crop_size = crop_size

        # ✅ LIVE-Challenge MOS 데이터 로드
        scores_csv_path = os.path.join(self.root, "LIVE_Challenge.txt")
        images_dir = os.path.join(self.root, "Images")  # ✅ 올바른 이미지 폴더 경로 설정

        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"[Error] {scores_csv_path} 파일이 존재하지 않습니다.")

        # ✅ CSV 파일 로드 (CSV 형식이 쉼표 구분자 사용)
        df = pd.read_csv(scores_csv_path, sep=",")

        # ✅ 이미지 및 MOS 점수 로드
        self.image_paths = []
        self.mos = []

        for index, row in df.iterrows():
            image_rel_path = row["dis_img_path"].strip()
            mos_score = row["score"]

            # ✅ 올바른 이미지 경로로 설정
            image_name = os.path.basename(image_rel_path)  # 예: "101.bmp"
            image_path = os.path.join(images_dir, image_name)  # 예: "E:/.../Images/101.bmp"

            if os.path.isfile(image_path):
                self.image_paths.append(image_path)
                self.mos.append(float(mos_score))
            else:
                print(f"[Warning] 이미지 파일을 찾을 수 없음: {image_path}")

        if len(self.image_paths) == 0:
            raise ValueError(f"[Error] 이미지 파일이 존재하지 않습니다. {scores_csv_path} 내용을 확인하세요.")

        print(f"[INFO] 로드된 이미지 개수: {len(self.image_paths)}, MOS 점수 개수: {len(self.mos)}")

        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index: int):
        """
        ✅ LIVE-C 데이터셋에서 index 번째 샘플을 반환
        - img_A: 원본 이미지  
        - img_B: 원본 이미지 (Authentic Dataset → Hard Negative 적용 X)  
        """
        try:
            img_A_orig = Image.open(self.image_paths[index]).convert("RGB")  # ✅ 원본 이미지 사용
        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            return None

        # ✅ img_A, img_B는 동일한 원본 이미지 사용
        img_A_transformed = self.transform(img_A_orig)
        img_B_transformed = self.transform(img_A_orig)

        return {
            "img_A": img_A_transformed,  # ✅ 원본 이미지
            "img_B": img_B_transformed,  # ✅ 원본 이미지 그대로 유지 (Hard Negative 없음)
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    """
    ✅ LIVE-C는 Authentic 데이터셋이므로 Hard Negative 없이 원본 이미지만 사용.
    """
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/LIVE"

    authentic_dataset = LIVEDataset(root=dataset_path, phase="training", crop_size=224)
    authentic_dataloader = DataLoader(authentic_dataset, batch_size=4, shuffle=True)

    print(f"Authentic Dataset size: {len(authentic_dataset)}")

    # ✅ Authentic 데이터셋의 첫 번째 배치 확인
    sample_batch_authentic = next(iter(authentic_dataloader))
    print(f"\n[Authentic] 샘플 확인:")
    for i in range(4):  
        print(f"  Sample {i+1} - MOS: {sample_batch_authentic['mos'][i]}")

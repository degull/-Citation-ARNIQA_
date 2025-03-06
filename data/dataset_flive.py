import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class FLIVEDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        """
        ✅ FLIVE 데이터셋 로드 및 MOS 점수 변환
        - `img_A`(왜곡된 이미지)만 반환
        - `mos`(Mean Opinion Score) 점수 반환
        """
        super().__init__()
        self.root = str(root)
        self.phase = phase
        self.crop_size = crop_size
        self.corrupt_images = []  # 손상된 이미지 리스트 저장

        # ✅ MOS CSV 파일 로드
        scores_csv_path = os.path.join(self.root, "ground_truth_dataset.csv")
        if not os.path.isfile(scores_csv_path):
            raise FileNotFoundError(f"FLIVE MOS CSV 파일이 {scores_csv_path} 경로에 존재하지 않습니다.")

        scores_data = pd.read_csv(scores_csv_path)

        # ✅ 'image_num'을 기반으로 이미지 경로 설정
        self.image_paths = [os.path.join(self.root, "images", f"{img_id}.jpg") for img_id in scores_data["image_num"]]

        # ✅ MOS 값 변환 (각 투표 비율을 가중 평균으로 변환)
        votes = scores_data.iloc[:, 1:].values  # 'vote_1' ~ 'vote_10' 데이터 추출
        vote_weights = np.arange(1, 11)  # 1~10의 가중치
        self.mos = np.sum(votes * vote_weights, axis=1) / np.sum(votes, axis=1)

        # ✅ MOS 정규화 (0~1 범위 조정)
        self.mos = (self.mos - self.mos.min()) / (self.mos.max() - self.mos.min())

        # ✅ 디버깅 출력 (CSV 일부 및 MOS 값 확인)
        print(scores_data.head())  # CSV의 상위 5개 행 출력
        print("\n🔹 [Debug] MOS 값 일부 확인 🔹")
        print("Raw MOS Values (Before Normalization):", self.mos[:10])

        # ✅ 데이터 분할 (FLIVE는 공식 스플릿 사용)
        if self.phase != "all":
            split_path = os.path.join(self.root, "splits", f"{self.phase}.npy")
            if not os.path.isfile(split_path):
                raise FileNotFoundError(f"FLIVE {self.phase} 스플릿 파일이 {split_path} 경로에 존재하지 않습니다.")

            split_idxs = np.load(split_path)
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # -1 패딩 제거
            self.image_paths = np.array(self.image_paths)[split_idxs]
            self.mos = self.mos[split_idxs]

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
        ✅ 손상된 이미지가 있으면 None 반환하여 배치에서 필터링
        """
        try:
            img_A = Image.open(self.image_paths[index]).convert("RGB")  # ✅ 원본 이미지 사용
            img_A_transformed = self.transform(img_A)  # ✅ 변환 적용
            mos_value = torch.tensor(self.mos[index], dtype=torch.float32)
            return {"img_A": img_A_transformed, "mos": mos_value}

        except Exception as e:
            print(f"[Error] Loading image: {self.image_paths[index]}: {e}")
            self.corrupt_images.append(self.image_paths[index])  # 손상된 이미지 저장
            return None  # ✅ 손상된 이미지가 있을 경우 None 반환

    def __len__(self):
        return len(self.image_paths)

    def save_corrupt_images(self, save_path="corrupt_images.txt"):
        """ 손상된 이미지 리스트를 파일로 저장 """
        if self.corrupt_images:
            with open(save_path, "w") as f:
                for img_path in self.corrupt_images:
                    f.write(img_path + "\n")
            print(f"🔹 [INFO] 손상된 이미지 리스트 저장 완료: {save_path}")


# ✅ DataLoader용 custom collate_fn 추가 (손상된 이미지 제거)
def custom_collate_fn(batch):
    """
    ✅ DataLoader에서 손상된 이미지를 포함한 배치를 필터링하는 collate_fn
    """
    batch = [data for data in batch if data is not None]  # ✅ None 제거
    if len(batch) == 0:
        return {"img_A": torch.empty(0), "mos": torch.empty(0)}  # ✅ 빈 텐서 반환
    return torch.utils.data.default_collate(batch)


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":
    """
    ✅ FLIVE는 Authentic 데이터셋이며, `DistortionDetectionModel`과 호환되도록 Hard Negative 없이 원본 이미지만 사용.
    """
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/FLIVE"

    dataset = FLIVEDataset(root=dataset_path, phase="train", crop_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    print(f"FLIVE Dataset size: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    if sample_batch["img_A"].shape[0] > 0:  # ✅ 빈 배치가 아닐 경우 출력
        print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
        print(f"Sample MOS Scores: {sample_batch['mos']}")
    else:
        print("[Warning] 첫 번째 배치가 비어 있음. (손상된 이미지가 많을 가능성)")

    # ✅ 손상된 이미지 리스트 저장
    dataset.save_corrupt_images("corrupt_images.txt")

    print("\n🔹 [Debug] CSV 파일 일부 확인 🔹")

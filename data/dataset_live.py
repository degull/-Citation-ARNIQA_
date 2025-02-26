import os
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class LIVEDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        """
        LIVE 데이터셋을 로드하는 클래스
        Args:
            root (str): 데이터셋 루트 경로
            phase (str): "train", "val", "test" 중 선택
            crop_size (int): 이미지 크롭 크기
        """
        super().__init__()
        self.root = root
        self.phase = phase
        self.crop_size = crop_size

        # ✅ MAT 파일 로드 (LIVE MOS 데이터)
        dmos_path = os.path.join(self.root, "dmos.mat")
        refnames_path = os.path.join(self.root, "refnames_all.mat")

        if not os.path.isfile(dmos_path) or not os.path.isfile(refnames_path):
            raise FileNotFoundError(f"LIVE 데이터셋의 dmos.mat 또는 refnames_all.mat 파일이 존재하지 않습니다.")

        # ✅ MOS 점수 로드 (0~100 범위)
        mat_data = scipy.io.loadmat(dmos_path)
        dmos = mat_data["dmos"][0]  # MOS 점수 (1D 배열)

        # ✅ MOS 점수 정규화 (0~1 범위로 변환)
        dmos = (dmos - dmos.min()) / (dmos.max() - dmos.min())

        # ✅ 참조 이미지 파일명 로드
        ref_data = scipy.io.loadmat(refnames_path)
        ref_images = [str(ref[0]) for ref in ref_data["refnames_all"][0]]  # 리스트 변환

        # ✅ 이미지 경로 매핑
        self.image_paths = []
        self.mos = []

        distortions = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]
        img_index = 0  # 이미지 인덱스

        for i, ref_img in enumerate(ref_images):  # 참조 이미지 별 반복
            for dist_type in distortions:
                for level in range(1, 6):  # 각 왜곡당 5개 강도
                    img_name = f"img{img_index + 1}.bmp"
                    img_path = os.path.join(self.root, dist_type, img_name)

                    if os.path.isfile(img_path):  # 이미지 존재 확인
                        self.image_paths.append(img_path)
                        self.mos.append(float(dmos[img_index]))  # MOS 점수 저장

                    img_index += 1  # 이미지 인덱스 증가

    def transform(self, image: Image) -> torch.Tensor:
        """이미지 변환 (크기 조정 + 텐서 변환)"""
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __getitem__(self, index: int):
        """데이터셋에서 특정 인덱스의 샘플을 가져옴"""
        img_A = Image.open(self.image_paths[index]).convert("RGB")
        img_A = self.transform(img_A)

        return {
            "img_A": img_A,
            "mos": torch.tensor(self.mos[index], dtype=torch.float32),
        }

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.image_paths)


# ✅ 데이터셋 테스트 코드
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/LIVE"

    dataset = LIVEDataset(root=dataset_path, phase="train", crop_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
    print(f"Sample MOS Scores: {sample_batch['mos']}")

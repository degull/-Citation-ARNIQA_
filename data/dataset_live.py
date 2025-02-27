import os
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class LIVEDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = root
        self.phase = phase
        self.crop_size = crop_size

        # ✅ MAT 파일 로드 (LIVE MOS 데이터)
        dmos_path = os.path.join(self.root, "dmos.mat")
        refnames_path = os.path.join(self.root, "refnames_all.mat")

        if not os.path.isfile(dmos_path) or not os.path.isfile(refnames_path):
            raise FileNotFoundError(f"LIVE 데이터셋의 dmos.mat 또는 refnames_all.mat 파일이 존재하지 않습니다.")

        # ✅ MOS 점수 로드 (총 982개)
        mat_data = scipy.io.loadmat(dmos_path)
        dmos = mat_data["dmos"][0].astype(np.float32)  # MOS 점수 (1D 배열)

        # ✅ MOS 점수 정규화 (0~1 범위로 변환)
        mos_min = np.min(dmos)
        mos_max = np.max(dmos)
        if mos_max - mos_min == 0:
            raise ValueError("[Error] MOS 값의 최소값과 최대값이 동일하여 정규화할 수 없습니다.")

        dmos = (dmos - mos_min) / (mos_max - mos_min)
        print(f"[Check] MOS 최소값: {np.min(dmos)}, 최대값: {np.max(dmos)}")

        # ✅ 참조 이미지 파일명 로드
        ref_data = scipy.io.loadmat(refnames_path)
        ref_images = [str(ref[0]) for ref in ref_data["refnames_all"][0]]  # 리스트 변환

        # ✅ LIVE 데이터셋 설정
        distortions = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]
        self.image_paths = []
        self.mos = []

        missing_files = 0  # 없는 파일 카운트

        # ✅ 이미지 파일명과 MOS 점수를 직접 매핑
        for img_index in range(len(dmos)):  # MOS 개수만큼 반복 (982개)
            distortion_type = distortions[img_index % len(distortions)]  # 🚀 수정: 안전한 인덱싱
            img_name = f"img{img_index + 1}.bmp"  # LIVE 데이터셋의 이미지명 패턴

            img_path = os.path.join(self.root, distortion_type, img_name)

            if os.path.isfile(img_path):  # ✅ 존재하는 파일만 추가
                self.image_paths.append(img_path)
                self.mos.append(float(dmos[img_index]))  # MOS 점수 저장
            else:
                missing_files += 1  # 없는 파일 개수 증가

        print(f"⚠️ {missing_files}개의 파일이 존재하지 않습니다.")  # ✅ 누락된 파일 개수 출력

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

    print(f"\n✅ 최종 Dataset 크기: {len(dataset)}")

    # ✅ 첫 번째 배치 확인
    sample_batch = next(iter(dataloader))
    print(f"Sample Image Shape: {sample_batch['img_A'].shape}")
    print(f"Sample MOS Scores: {sample_batch['mos']}")

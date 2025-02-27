import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import stats
from pathlib import Path

from models.attention_se import EnhancedDistortionDetectionModel
from data.dataset_kadid10k import KADID10KDataset
from data.dataset_tid2013 import TID2013Dataset
from data.dataset_live import LIVEDataset
from data.dataset_clive import CLIVEDataset
from data.dataset_spaq import SPAQDataset
from data.dataset_csiq import CSIQDataset 
from data.dataset_koniq10k import KONIQ10KDataset 
from utils.utils import load_config

# ✅ SROCC 및 PLCC 계산 함수
def calculate_srcc_plcc(preds, targets):
    preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    srocc, _ = stats.spearmanr(preds.flatten(), targets.flatten())
    plcc, _ = stats.pearsonr(preds.flatten(), targets.flatten())
    return srocc, plcc

# ✅ 테스트 함수
def test(model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            preds = model(img_A)
            srocc, plcc = calculate_srcc_plcc(preds, targets)

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return {
        "srocc": np.mean(srocc_values),
        "plcc": np.mean(plcc_values)
    }

# ✅ 크로스 데이터셋 테스트 실행
if __name__ == "__main__":
    # ✅ 설정 파일 로드
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # ✅ GPU 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # ✅ 모델 로드
    model = EnhancedDistortionDetectionModel().to(device)

    # ✅ 저장된 체크포인트 로드
    checkpoint_path = Path(args.checkpoint_base_path) / "E:/ARNIQA - SE - mix/ARNIQA/experiments/my_experiment/regressors/kadid/epoch_27_srocc_0.938.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # ✅ 데이터셋 경로 설정
    dataset_paths = {
        "KADID10K": "E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K",
        "LIVE": "E:/ARNIQA - SE - mix/ARNIQA/dataset/LIVE",  # ✅ LIVE 데이터셋 추가
        "TID2013": "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013",
        "SPAQ": "E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ",
        "CSIQ": "E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ",  # ✅ CSIQ 추가
        "KONIQ10K": "E:/ARNIQA - SE - mix/ARNIQA/dataset/KONIQ10K",  # ✅ Koniq-10k 추가
        "CLIVE" : "E:/ARNIQA - SE - mix/ARNIQA/dataset/CLIVE"
    }

    # ✅ 데이터셋 로드
    test_datasets = {
        "KADID10K": KADID10KDataset(root=dataset_paths["KADID10K"], phase="test", crop_size=224),
        "LIVE": LIVEDataset(root=dataset_paths["LIVE"], phase="test", crop_size=224),  # ✅ LIVE 데이터셋 추가
        "CLIVE": CLIVEDataset(root=dataset_paths["CLIVE"], phase="test", crop_size=224),  # ✅ LIVE 데이터셋 추가
        "TID2013": TID2013Dataset(root=dataset_paths["TID2013"], phase="test", crop_size=224),
        "SPAQ": SPAQDataset(root=dataset_paths["SPAQ"], phase="test", crop_size=224),
        "CSIQ": CSIQDataset(root=dataset_paths["CSIQ"], phase="test", crop_size=224),  # ✅ CSIQ 추가
        "Koniq-10k": KONIQ10KDataset(root=dataset_paths["Koniq-10k"], phase="test", crop_size=224)  # ✅ Koniq-10k 추가
    }

    # ✅ 각 데이터셋에 대해 테스트 실행
    results = {}
    for dataset_name, dataset in test_datasets.items():
        print(f"\n🔹 Testing on {dataset_name} Dataset...")

        test_dataloader = DataLoader(dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)
        test_result = test(model, test_dataloader, device)

        results[dataset_name] = test_result

    # ✅ 최종 결과 출력
    print("\n🔹 **Final Cross-Dataset Test Results:** 🔹")
    for dataset, metrics in results.items():
        print(f"📌 **{dataset}:** SROCC: {metrics['srocc']:.4f}, PLCC: {metrics['plcc']:.4f}")

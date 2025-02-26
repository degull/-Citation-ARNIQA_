import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from data.dataset_kadid10k import KADID10KDataset
from models.attention_se import EnhancedDistortionDetectionModel
from utils.utils import load_config

# ✅ 손실 함수 (MSE + Perceptual Loss)
def distortion_loss(pred, gt):
    mse_loss = nn.MSELoss()(pred, gt)
    perceptual_loss = torch.mean(torch.abs(pred - gt))
    return mse_loss + 0.1 * perceptual_loss

# ✅ SROCC 및 PLCC 계산
def calculate_srcc_plcc(preds, targets):
    preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    srocc, _ = stats.spearmanr(preds.flatten(), targets.flatten())
    plcc, _ = stats.pearsonr(preds.flatten(), targets.flatten())
    return srocc, plcc

# ✅ 학습 루프
def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    best_srocc = -1
    model.train()

    train_losses = []
    val_srocc_values, val_plcc_values = [], []

    for epoch in range(args.training.epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            optimizer.zero_grad()

            # ✅ 모델 예측
            preds = model(img_A)

            # ✅ 손실 함수 계산
            loss = distortion_loss(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        # ✅ 검증
        val_srocc, val_plcc = validate(model, val_dataloader, device)
        val_srocc_values.append(val_srocc)
        val_plcc_values.append(val_plcc)

        # ✅ 모델 저장
        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, args.checkpoint_base_path, epoch, val_srocc)

        print(f"\n🔹 Epoch {epoch+1}: Loss: {avg_loss:.6f}, Val SROCC: {val_srocc:.6f}, Val PLCC: {val_plcc:.6f}")

        lr_scheduler.step()

    print("\n✅ **Training Completed** ✅")

    return {
        "loss": train_losses,
        "srocc": val_srocc_values,
        "plcc": val_plcc_values
    }

# ✅ 검증 루프
def validate(model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            preds = model(img_A)
            srocc, plcc = calculate_srcc_plcc(preds, targets)

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

# ✅ 테스트 루프 (추가)
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
        "srocc": srocc_values,
        "plcc": plcc_values
    }

# ✅ 모델 저장 함수
def save_checkpoint(model, checkpoint_path, epoch, srocc):
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), Path(checkpoint_path) / filename)



# ✅ 메인 실행
if __name__ == "__main__":
    # ✅ 설정 파일 로드
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # ✅ GPU 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # ✅ 데이터셋 로드
    dataset_path = Path(args.data_base_path)
    dataset = KADID10KDataset(str(dataset_path), crop_size=224)


    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

    # ✅ 모델 생성
    model = EnhancedDistortionDetectionModel().to(device)

    # ✅ 옵티마이저 및 스케줄러 설정
    optimizer = optim.SGD(model.parameters(), lr=args.training.learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ✅ 학습 시작
    train_metrics = train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device)

    # ✅ 테스트 수행
    test_metrics = test(model, test_dataloader, device)

    # ✅ 최종 결과 출력
    print("\n✅ **Training Completed** ✅\n")

    print("🔹 **Final Training Metrics:** 🔹")
    for epoch, (loss, srocc, plcc) in enumerate(zip(train_metrics["loss"], train_metrics["srocc"], train_metrics["plcc"])):
        print(f"📌 **Epoch {epoch+1}:** Loss: {loss:.6f}, SROCC: {srocc:.6f}, PLCC: {plcc:.6f}")

    print("\n🔹 **Final Validation Metrics:** 🔹", {
        "srocc": train_metrics["srocc"],
        "plcc": train_metrics["plcc"]
    })

    print("🔹 **Final Test Metrics:** 🔹", test_metrics)

    
""" 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from utils.utils import load_config
from data.dataset_kadid10k import KADID10KDataset
from data.dataset_csiq import CSIQDataset
from models.attention_se import EnhancedDistortionDetectionModel  # ✅ Feature Extractor

# ✅ Regressor 추가
class IQARegressor(nn.Module):
    def __init__(self, feature_dim=512):
        super(IQARegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 최종 MOS 예측값 출력
        )

    def forward(self, x):
        return self.regressor(x)

# ✅ 모델 저장 함수
def save_checkpoint(model, checkpoint_path, epoch, srocc):
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

# ✅ SROCC 및 PLCC 계산 함수
def calculate_srcc_plcc(preds, targets):
    if torch.isnan(preds).any() or torch.isnan(targets).any():
        return np.nan, np.nan  # NaN이 포함된 경우 NaN 반환
    
    preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    srocc, _ = stats.spearmanr(preds.flatten(), targets.flatten())
    plcc, _ = stats.pearsonr(preds.flatten(), targets.flatten())
    return srocc, plcc

# ✅ 검증 루프
def validate(feature_extractor, regressor, dataloader, device):
    feature_extractor.eval()
    regressor.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            features = feature_extractor(img_A)
            preds = regressor(features).squeeze()

            # ✅ NaN 체크 및 필터링
            if torch.isnan(preds).any() or torch.isnan(targets).any():
                print("[Warning] NaN detected in validation batch. Skipping...")
                continue

            srocc, plcc = calculate_srcc_plcc(preds, targets)

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.nanmean(srocc_values), np.nanmean(plcc_values)

# ✅ 학습 루프 (lr_scheduler이 None이면 step() 호출 안함)
def train(args, feature_extractor, regressor, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics_per_epoch = []  

    for epoch in range(args.training.epochs):
        feature_extractor.train()
        regressor.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            optimizer.zero_grad()

            features = feature_extractor(img_A)
            preds = regressor(features).squeeze()

            if torch.isnan(preds).any() or torch.isnan(targets).any():
                print("[Warning] NaN detected in batch. Skipping...")
                continue

            preds = torch.clamp(preds, min=0.0, max=1.0)
            targets = torch.clamp(targets, min=0.0, max=1.0)

            loss = nn.MSELoss()(preds, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        val_srocc, val_plcc = validate(feature_extractor, regressor, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        test_metrics = test(feature_extractor, regressor, test_dataloader, device)
        test_metrics_per_epoch.append(test_metrics)

        avg_srcc = np.nanmean(test_metrics['srcc'])
        avg_plcc = np.nanmean(test_metrics['plcc'])
        print(f"Epoch {epoch + 1}: Test SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")

        # ✅ lr_scheduler이 None이 아니면 step() 실행
        if lr_scheduler is not None:
            lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(regressor, checkpoint_path, epoch, val_srocc)

    print("Training completed.")

    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics_per_epoch, 1):
        avg_srcc = np.nanmean(metrics['srcc'])
        avg_plcc = np.nanmean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")

    return train_metrics, val_metrics, test_metrics_per_epoch



# ✅ 테스트 루프
def test(feature_extractor, regressor, test_dataloader, device):
    feature_extractor.eval()
    regressor.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            features = feature_extractor(img_A)
            preds = regressor(features).squeeze()

            # ✅ NaN 체크
            if torch.isnan(preds).any() or torch.isnan(targets).any():
                print("[Warning] NaN detected in test batch. Skipping...")
                continue

            srocc, plcc = calculate_srcc_plcc(preds, targets)

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return {"srcc": np.nanmean(srocc_values), "plcc": np.nanmean(plcc_values)}


# ✅ 학습 데이터 로드
def create_dataloaders(args):
    kadid_dataset_path = Path(str(args.data_base_path_kadid))
    csiq_dataset_path = Path(str(args.data_base_path_csiq))

    kadid_dataset = KADID10KDataset(str(kadid_dataset_path))
    csiq_dataset = CSIQDataset(str(csiq_dataset_path))

    train_size = int(0.8 * len(kadid_dataset))
    val_size = len(kadid_dataset) - train_size
    train_dataset, val_dataset = random_split(kadid_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(csiq_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader

# ✅ 메인 실행
if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args)

    feature_extractor = EnhancedDistortionDetectionModel().to(device)
    regressor = IQARegressor(feature_dim=feature_extractor.feature_dim).to(device)

    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(regressor.parameters()), 
        lr=args.training.learning_rate
    )

    # ✅ lr_scheduler 추가
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    ) if hasattr(args.training, "lr_scheduler") else None  # ❗ `args.training.lr_scheduler`가 없으면 None 할당

    train_metrics, val_metrics, test_metrics = train(
        args, feature_extractor, regressor, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device
    ) """
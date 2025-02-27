import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from data.dataset_spaq import SPAQDataset
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
    dataset = SPAQDataset(str(dataset_path), crop_size=224)


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

    
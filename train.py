# 원본
""" 
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
 """

# hmm
""" 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from data.dataset_clive import CLIVEDataset
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


# ✅ 테스트 루프 (에포크별로 SROCC, PLCC 저장)
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

    return np.mean(srocc_values), np.mean(plcc_values)


# ✅ 학습 루프 (각 epoch 후 test() 실행)
def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    best_srocc = -1
    model.train()

    train_losses = []
    val_srocc_values, val_plcc_values = [], []
    test_srocc_values, test_plcc_values = [], []  # ✅ Test 결과 저장

    for epoch in range(args["training"]["epochs"]):
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args['training']['epochs']}]")

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

        # ✅ 테스트 (각 epoch 후 실행)
        test_srocc, test_plcc = test(model, test_dataloader, device)
        test_srocc_values.append(test_srocc)
        test_plcc_values.append(test_plcc)

        # ✅ 모델 저장
        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, args["checkpoint_base_path"], epoch, val_srocc)

        print(f"\n🔹 Epoch {epoch+1}: Loss: {avg_loss:.6f}, Val SROCC: {val_srocc:.6f}, Val PLCC: {val_plcc:.6f}, "
              f"Test SROCC: {test_srocc:.6f}, Test PLCC: {test_plcc:.6f}")

        lr_scheduler.step()

    print("\n✅ **Training Completed** ✅")

    return {
        "loss": train_losses,
        "val_srocc": val_srocc_values,
        "val_plcc": val_plcc_values,
        "test_srocc": test_srocc_values,  # ✅ Test 결과 추가 저장
        "test_plcc": test_plcc_values
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
    device = torch.device(f"cuda:{args['device']}" if torch.cuda.is_available() else "cpu")

    # ✅ 데이터셋 로드
    dataset_path = Path(args["data_base_path"])
    dataset = CLIVEDataset(str(dataset_path), crop_size=224)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args["training"]["batch_size"], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args["training"]["batch_size"], shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args["training"]["batch_size"], shuffle=False, num_workers=4)

    # ✅ 모델 생성
    model = EnhancedDistortionDetectionModel().to(device)

    # ✅ 옵티마이저 및 스케줄러 설정
    optimizer = optim.SGD(model.parameters(), lr=args["training"]["learning_rate"], momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ✅ 학습 시작
    train_metrics = train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device)

    # ✅ 최종 결과 출력
    print("\n✅ **Training Completed** ✅\n")

    print("🔹 **Final Training Metrics:** 🔹")
    for epoch, (loss, val_srocc, val_plcc, test_srocc, test_plcc) in enumerate(
            zip(train_metrics["loss"], train_metrics["val_srocc"], train_metrics["val_plcc"], train_metrics["test_srocc"], train_metrics["test_plcc"])):
        print(f"📌 **Epoch {epoch+1}:** Loss: {loss:.6f}, Val SROCC: {val_srocc:.6f}, Val PLCC: {val_plcc:.6f}, "
              f"Test SROCC: {test_srocc:.6f}, Test PLCC: {test_plcc:.6f}")

    print("\n🔹 **Final Validation Metrics:** 🔹", {
        "srocc": train_metrics["val_srocc"],
        "plcc": train_metrics["val_plcc"]
    })

    print("\n🔹 **Final Test Metrics:** 🔹", {
        "srocc": train_metrics["test_srocc"],
        "plcc": train_metrics["test_plcc"]
    })

 """

# HAN_IQA_PLUS
""" import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from data.dataset_koniq10k import KONIQ10KDataset
from models.attention_se import HAN_IQA_PLUS  # 🔥 모델 변경
from utils.utils import load_config

# ✅ 손실 함수 (MSE + Perceptual Loss)
def distortion_loss(pred, gt):
    mse_loss = nn.MSELoss()(pred, gt)
    perceptual_loss = torch.mean(torch.abs(pred - gt))  # L1 Loss
    return mse_loss + 0.1 * perceptual_loss  # 🔥 Perceptual Loss 반영

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
    dataset = KONIQ10KDataset(str(dataset_path), crop_size=224)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

    # ✅ 모델 생성
    model = HAN_IQA_PLUS().to(device)  # 🔥 새로운 모델 적용

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

    print("🔹 **Final Test Metrics:** 🔹", test_metrics) """





# 시각화1
""" import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from models.attention_se import EnhancedDistortionDetectionModel
from data.dataset_kadid10k import KADID10KDataset
from utils.utils import load_config
from grad_cam import GradCAM

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

# ✅ 모델 저장 함수
def save_checkpoint(model, checkpoint_path, epoch, srocc):
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), Path(checkpoint_path) / filename)

# ✅ Feature Map 시각화
def visualize_feature_maps(model, dataloader, device):
    model.eval()
    batch = next(iter(dataloader))
    img_A = batch["img_A"].to(device)

    with torch.no_grad():
        feat1, feat2, feat3, feat4, feat5 = model.vgg(img_A)
        cpfe_output = model.cpfe(feat5)
        hnca_output = model.hnca(cpfe_output)

    feature_maps = {
        "VGG_feat1": feat1,
        "VGG_feat2": feat2,
        "VGG_feat3": feat3,
        "VGG_feat4": feat4,
        "VGG_feat5": feat5,
        "CPFE_output": cpfe_output,
        "HNCA_output": hnca_output
    }

    for name, fmap in feature_maps.items():
        fmap = fmap[0].cpu().numpy()
        fmap = np.mean(fmap, axis=0)  # 채널 평균
        plt.imshow(fmap, cmap='jet')
        plt.axis('off')
        plt.title(name)
        plt.savefig(f"results/{name}.png")
        plt.close()

# ✅ 학습 루프
def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    best_srocc = -1
    model.train()

    for epoch in range(args.training.epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            optimizer.zero_grad()
            preds = model(img_A)

            loss = distortion_loss(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        val_srocc, val_plcc = validate(model, val_dataloader, device)

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, args.checkpoint_base_path, epoch, val_srocc)

        print(f"\n🔹 Epoch {epoch+1}: Loss: {avg_loss:.6f}, Val SROCC: {val_srocc:.6f}, Val PLCC: {val_plcc:.6f}")
        lr_scheduler.step()

    print("\n✅ **Training Completed** ✅")

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

# ✅ 테스트 루프
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

# ✅ 메인 실행
def main():
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(args.data_base_path)
    dataset = KADID10KDataset(str(dataset_path), crop_size=224)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

    model = EnhancedDistortionDetectionModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.training.learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ✅ Feature Map & Grad-CAM 시각화
    visualize_feature_maps(model, val_dataloader, device)
    gradcam = GradCAM(model, model.vgg.conv5_3)
    for i, batch in enumerate(val_dataloader):
        img_A = batch["img_A"].to(device)
        cam = gradcam.generate_cam(img_A)
        gradcam.visualize_cam(img_A[0].cpu().numpy().transpose(1, 2, 0), cam, save_path=f"results/gradcam_{i}.png")
        if i == 5:
            break

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device)
    test_metrics = test(model, test_dataloader, device)
    print("🔹 **Final Test Metrics:** 🔹", test_metrics)

if __name__ == "__main__":
    main()
 """

# 시각화2
""" import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from models.attention_se import EnhancedDistortionDetectionModel
from data.dataset_koniq10k import KONIQ10KDataset
from utils.utils import load_config
from grad_cam import GradCAM
import os

# ✅ 디렉토리 생성 (결과 저장)
os.makedirs("results", exist_ok=True)

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

# ✅ 모델 저장 함수
def save_checkpoint(model, checkpoint_path, epoch, srocc):
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), Path(checkpoint_path) / filename)

# ✅ Attention Map 시각화
# ✅ Attention Map 시각화 (디렉토리 생성 추가)
def visualize_attention_maps(model, dataloader, device):
    model.eval()
    batch = next(iter(dataloader))
    img_A = batch["img_A"].to(device)

    with torch.no_grad():
        feat1, feat2, feat3, feat4, feat5 = model.vgg(img_A)

        # ✅ CoordAttention 적용 전후 비교
        low_feat_before = feat1
        low_feat_after = model.coord_attn(feat1) * feat1

        # ✅ HNCA 적용 전후 비교
        high_feat_before = model.cpfe(feat5)
        high_feat_after = model.hnca(high_feat_before)

    attention_maps = {
        "Low_feat_before_CoordAttn": low_feat_before,
        "Low_feat_after_CoordAttn": low_feat_after,
        "High_feat_before_HNCA": high_feat_before,
        "High_feat_after_HNCA": high_feat_after
    }

    # ✅ 저장 디렉토리 확인 및 생성
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    for name, fmap in attention_maps.items():
        fmap = fmap[0].cpu().numpy()
        fmap = np.mean(fmap, axis=0)  # 채널 평균
        plt.imshow(fmap, cmap='jet')
        plt.axis('off')
        plt.title(name)
        plt.savefig(os.path.join(save_dir, f"{name}.png"))  # 🔥 저장 위치 지정
        plt.close()


# ✅ 학습 루프
def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    best_srocc = -1
    model.train()

    for epoch in range(args.training.epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            optimizer.zero_grad()
            preds = model(img_A)

            loss = distortion_loss(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        val_srocc, val_plcc = validate(model, val_dataloader, device)

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, args.checkpoint_base_path, epoch, val_srocc)

        print(f"\n🔹 Epoch {epoch+1}: Loss: {avg_loss:.6f}, Val SROCC: {val_srocc:.6f}, Val PLCC: {val_plcc:.6f}")
        lr_scheduler.step()

    print("\n✅ **Training Completed** ✅")

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

# ✅ 테스트 루프
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

# ✅ 메인 실행
def main():
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(args.data_base_path)
    dataset = KONIQ10KDataset(str(dataset_path), crop_size=224)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

    model = EnhancedDistortionDetectionModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.training.learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ✅ Attention Map 시각화 실행 (디렉토리 자동 생성 포함)
    visualize_attention_maps(model, val_dataloader, device)

    print("\n✅ **Attention Map Visualization Completed** ✅\n")

if __name__ == "__main__":
    main() """


"""
CoordAttention 적용 전후 비교 (feat1)
HNCA 적용 전후 비교 (CPFE(feat5))
"""

# 시각화3
""" import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from models.attention_se import EnhancedDistortionDetectionModel
from utils.utils import load_config
from PIL import Image


# ✅ 특정 이미지 경로
IMAGE_PATH = "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/distorted_images/i07_15_3.bmp"


# ✅ 손실 함수 (MSE + Perceptual Loss)
def distortion_loss(pred, gt, distortion_type_pred, distortion_type_gt):
    mse_loss = nn.MSELoss()(pred, gt)
    perceptual_loss = torch.mean(torch.abs(pred - gt))
    distortion_class_loss = nn.CrossEntropyLoss()(distortion_type_pred, distortion_type_gt)  # Distortion 분류 Loss 추가
    return mse_loss + 0.1 * perceptual_loss + 0.5 * distortion_class_loss


# ✅ SROCC 및 PLCC 계산
def calculate_srcc_plcc(preds, targets):
    preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    srocc, _ = stats.spearmanr(preds.flatten(), targets.flatten())
    plcc, _ = stats.pearsonr(preds.flatten(), targets.flatten())
    return srocc, plcc


# ✅ 모델 저장 함수
def save_checkpoint(model, checkpoint_path, epoch, srocc):
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), Path(checkpoint_path) / filename)

 """
# ✅ Attention Map Overlay 시각화
""" import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_attention_on_image(image, attention_map, save_path):

    # ✅ Attention Map 정규화 (0~1 스케일)
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
    attention_map = np.uint8(attention_map * 255)  # 0~255 변환

    # ✅ Color Map 적용 (Jet 컬러맵)
    heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR이라 RGB로 변환

    # ✅ 원본 이미지 크기로 리사이즈
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # (W, H) 맞추기

    # ✅ image가 Grayscale (H, W)일 경우 RGB 변환
    if len(image.shape) == 2 or image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # ✅ 채널 수 맞추기 (H, W, 3)
    if heatmap.shape != image.shape:
        print(f"🔥 [Error] Shape Mismatch: image {image.shape}, heatmap {heatmap.shape}")
        return

    # ✅ 이미지와 Overlay
    overlayed_img = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    # ✅ 저장 및 시각화
    plt.figure(figsize=(6, 6))
    plt.imshow(overlayed_img)
    plt.axis('off')
    plt.title(save_path.split("/")[-1])
    plt.savefig(save_path)
    plt.close()




# ✅ 특정 이미지로 Attention Map 시각화
def visualize_attention_maps(model, device):
    model.eval()

    # ✅ 특정 이미지 로드
    img = Image.open(IMAGE_PATH).convert("RGB")
    img = img.resize((224, 224))
    img_np = np.array(img)

    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        feat1, feat2, feat3, feat4, feat5 = model.vgg(img_tensor)
        low_feat_before = feat1
        low_feat_after = model.coord_attn(feat1)
        high_feat_before = model.cpfe(feat5)
        high_feat_after = model.hnca(high_feat_before)

    attention_maps = {
        "Low_feat_before_CoordAttn": low_feat_before,
        "Low_feat_after_CoordAttn": low_feat_after,
        "High_feat_before_HNCA": high_feat_before,
        "High_feat_after_HNCA": high_feat_after
    }

    for name, fmap in attention_maps.items():
        fmap = fmap[0].cpu().numpy()
        fmap = np.mean(fmap, axis=0)

        overlay_attention_on_image(img_np, fmap, f"results/{name}.png")


# ✅ 학습 루프
def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    best_srocc = -1
    model.train()

    for epoch in range(args.training.epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            optimizer.zero_grad()
            preds = model(img_A)

            loss = distortion_loss(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        val_srocc, val_plcc = validate(model, val_dataloader, device)

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, args.checkpoint_base_path, epoch, val_srocc)

        print(f"\n🔹 Epoch {epoch+1}: Loss: {avg_loss:.6f}, Val SROCC: {val_srocc:.6f}, Val PLCC: {val_plcc:.6f}")
        lr_scheduler.step()

    print("\n✅ **Training Completed** ✅")


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


# ✅ 테스트 루프
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


# ✅ 메인 실행
def main():
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model = EnhancedDistortionDetectionModel().to(device)

    # ✅ 특정 이미지로 Attention Map 시각화
    visualize_attention_maps(model, device)


if __name__ == "__main__":
    main()
 """

# 3/6
""" import torch
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from data.dataset_flive import FLIVEDataset
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
    dataset = FLIVEDataset(str(dataset_path), crop_size=224)


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


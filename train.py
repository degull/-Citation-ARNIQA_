#KADID
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


def verify_hard_negatives(original_shape, downscaled_shape):
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


def debug_ridge_regressor(embeddings, mos_scores):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], mos_scores, alpha=0.7)
    plt.xlabel('Embedding Feature 0')
    plt.ylabel('MOS Scores')
    plt.title('Embedding vs MOS Scores')
    plt.grid()
    plt.show()

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B, *_ = model(inputs_A, inputs_B)
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(args.checkpoint_base_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5).to(device)

            optimizer.zero_grad()

            # Forward pass with SE weights
            proj_A, proj_B, se_weights_A, se_weights_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1)
            proj_B = F.normalize(proj_B, dim=1)

            # train 함수에서 features_negatives 처리 부분 수정
            features_negatives = model.backbone(hard_negatives)

            # features_negatives의 차원을 확인
            if isinstance(features_negatives, tuple):  # ResNet에서 여러 값이 반환될 경우 첫 번째 값만 사용
                features_negatives = features_negatives[0]

            if features_negatives.dim() == 4:
                features_negatives = features_negatives.mean([2, 3])  # GAP 적용하여 2D로 변환
            elif features_negatives.dim() != 2:
                raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

            # Normalize and project
            proj_negatives = F.normalize(model.projector(features_negatives), dim=1)

            # Compute loss
            loss = model.compute_loss(proj_A, proj_B, proj_negatives, se_weights_A, se_weights_B)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics



def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            # Model forward pass
            proj_A, proj_B, *_ = model(inputs_A, inputs_B)  # 나머지 반환 값 무시

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics


def optimize_ridge_alpha(embeddings, mos_scores):
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    grid.fit(embeddings, mos_scores)
    best_alpha = grid.best_params_['alpha']
    print(f"Optimal alpha: {best_alpha}")
    return Ridge(alpha=best_alpha).fit(embeddings, mos_scores)

def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch in train_dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # 입력이 5D 텐서일 경우 4D로 변환
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, channels, height, width]

            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            # `mos` 값 반복 (inputs_A 크기에 맞춰 조정)
            repeat_factor = features_A.shape[0] // mos.shape[0]  # inputs_A와 mos의 크기 비율 계산
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            embeddings.append(features_A)
            mos_scores.append(mos_repeated)

    # 리스트를 numpy 배열로 변환
    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # 크기 검증
    assert embeddings.shape[0] == mos_scores.shape[0], \
        f"Mismatch in embeddings ({embeddings.shape[0]}) and MOS scores ({mos_scores.shape[0]})"

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, mos_scores)
    print("Ridge Regressor Trained: Optimal alpha=1.0")
    return ridge




def evaluate_ridge_regressor(regressor, model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # Flatten crops if needed

            # Use backbone features for Ridge prediction
            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            prediction = regressor.predict(features_A)  # Use backbone features
            repeat_factor = features_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            predictions.extend(prediction)
            mos_scores.extend(mos_repeated)

    mos_scores = np.array(mos_scores)
    predictions = np.array(predictions)
    assert mos_scores.shape == predictions.shape, \
        f"Mismatch between MOS ({mos_scores.shape}) and Predictions ({predictions.shape})"

    return mos_scores, predictions




def plot_results(mos_scores, predictions):
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()

def debug_embeddings(embeddings, title="Embeddings"):
    plt.figure(figsize=(8, 6))
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7)
    plt.title(f"{title} Distribution")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # proj_A가 튜플인지 확인
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # proj_A만 사용

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics

def test_hard_negative_attention(args, model, dataloader, device):
    model.eval()
    metrics = {'srcc': [], 'plcc': []}

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize the embeddings
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            # Compute SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            metrics['srcc'].append(srocc)
            metrics['plcc'].append(plcc)

    return metrics


if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path) / "kadid10k.csv"
    dataset = KADID10KDataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    model = SimCLR(embedding_dim=128, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    )

    # Train
    train_metrics, val_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Test
    test_metrics = test(args, model, test_dataloader, device)

    print("KADID10K")
    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)
"""

#KADID 2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


def verify_hard_negatives(original_shape, downscaled_shape):
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def debug_ridge_regressor(embeddings, mos_scores):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], mos_scores, alpha=0.7)
    plt.xlabel('Embedding Feature 0')
    plt.ylabel('MOS Scores')
    plt.title('Embedding vs MOS Scores')
    plt.grid()
    plt.show()

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)


def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # Debug: Log hard_negatives shape
            print(f"[Debug] hard_negatives shape before processing: {hard_negatives.shape}")

            if hard_negatives.dim() == 5:
                hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:]).to(device)
            elif hard_negatives.dim() != 4:
                raise ValueError(f"[Error] Unexpected hard_negatives dimensions: {hard_negatives.shape}")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)

                # Normalize projections
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                # Backbone processing for hard negatives
                features_negatives = model.backbone(hard_negatives)
                print(f"[Debug] features_negatives shape after backbone: {features_negatives.shape}")

                if features_negatives.dim() == 4:
                    features_negatives = features_negatives.mean([2, 3])
                elif features_negatives.dim() != 2:
                    raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

                proj_negatives = F.normalize(model.projector(features_negatives), dim=1)

                # Compute loss
                loss = model.compute_loss(proj_A, proj_B, proj_negatives)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics




def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics


def optimize_ridge_alpha(embeddings, mos_scores):
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    grid.fit(embeddings, mos_scores)
    best_alpha = grid.best_params_['alpha']
    print(f"Optimal alpha: {best_alpha}")
    return Ridge(alpha=best_alpha).fit(embeddings, mos_scores)

def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch in train_dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # 입력이 5D 텐서일 경우 4D로 변환
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, channels, height, width]

            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            # `mos` 값 반복 (inputs_A 크기에 맞춰 조정)
            repeat_factor = features_A.shape[0] // mos.shape[0]  # inputs_A와 mos의 크기 비율 계산
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            embeddings.append(features_A)
            mos_scores.append(mos_repeated)

    # 리스트를 numpy 배열로 변환
    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # 크기 검증
    assert embeddings.shape[0] == mos_scores.shape[0], \
        f"Mismatch in embeddings ({embeddings.shape[0]}) and MOS scores ({mos_scores.shape[0]})"

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, mos_scores)
    print("Ridge Regressor Trained: Optimal alpha=1.0")
    return ridge




def evaluate_ridge_regressor(regressor, model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # Flatten crops if needed

            # Use backbone features for Ridge prediction
            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            prediction = regressor.predict(features_A)  # Use backbone features
            repeat_factor = features_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            predictions.extend(prediction)
            mos_scores.extend(mos_repeated)

    mos_scores = np.array(mos_scores)
    predictions = np.array(predictions)
    assert mos_scores.shape == predictions.shape, \
        f"Mismatch between MOS ({mos_scores.shape}) and Predictions ({predictions.shape})"

    return mos_scores, predictions




def plot_results(mos_scores, predictions):
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()

def debug_embeddings(embeddings, title="Embeddings"):
    plt.figure(figsize=(8, 6))
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7)
    plt.title(f"{title} Distribution")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # proj_A가 튜플인지 확인
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # proj_A만 사용

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics

def test_hard_negative_attention(args, model, dataloader, device):
    model.eval()
    metrics = {'srcc': [], 'plcc': []}

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize the embeddings
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            # Compute SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            metrics['srcc'].append(srocc)
            metrics['plcc'].append(plcc)

    return metrics


if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path) / "kadid10k.csv"
    dataset = KADID10KDataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    model = SimCLR(embedding_dim=128, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    )

    # Train
    train_metrics, val_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Test
    test_metrics = test(args, model, test_dataloader, device)

    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)

# Training Metrics: {'loss': [0.7686459263345189, 0.7715813556589337, 0.7580022199697473, 0.7390993061910633, 0.7330587068341387, 0.7094864248556693, 0.6790706018158329, 0.656973103570077, 0.658129592445042, 0.6539432713867042]}
# Validation Metrics: {'srcc': [0.9369970935451635, 0.9364419593681123, 0.9353765132161436, 0.936218218360584, 0.9338744120782958, 0.9348942748556499, 0.931227948057318, 0.9312314522671019, 0.9265542097272341, 
# 0.9347717103299581], 'plcc': [0.9415914918050985, 0.9409555131953475, 0.9397106435276338, 0.9406181322098353, 0.9382513346102095, 0.9398124284436276, 0.9357886273608912, 0.9362907462564223, 0.9316025511341239, 0.9390125717015866]}
# Test Metrics: {'srcc': 0.9341844429747825, 'plcc': 0.9380751799502336}
 

# TID2013
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import TID2013Dataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


def verify_hard_negatives(original_shape, downscaled_shape):
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


def debug_ridge_regressor(embeddings, mos_scores):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], mos_scores, alpha=0.7)
    plt.xlabel('Embedding Feature 0')
    plt.ylabel('MOS Scores')
    plt.title('Embedding vs MOS Scores')
    plt.grid()
    plt.show()

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B, *_ = model(inputs_A, inputs_B)
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(args.checkpoint_base_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5).to(device)

            optimizer.zero_grad()

            # Forward pass with SE weights
            proj_A, proj_B, se_weights_A, se_weights_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1)
            proj_B = F.normalize(proj_B, dim=1)

            # train 함수에서 features_negatives 처리 부분 수정
            features_negatives = model.backbone(hard_negatives)

            # features_negatives의 차원을 확인
            if isinstance(features_negatives, tuple):  # ResNet에서 여러 값이 반환될 경우 첫 번째 값만 사용
                features_negatives = features_negatives[0]

            if features_negatives.dim() == 4:
                features_negatives = features_negatives.mean([2, 3])  # GAP 적용하여 2D로 변환
            elif features_negatives.dim() != 2:
                raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

            # Normalize and project
            proj_negatives = F.normalize(model.projector(features_negatives), dim=1)

            # Compute loss
            loss = model.compute_loss(proj_A, proj_B, proj_negatives, se_weights_A, se_weights_B)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics



def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            # Model forward pass
            proj_A, proj_B, *_ = model(inputs_A, inputs_B)  # 나머지 반환 값 무시

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics


def optimize_ridge_alpha(embeddings, mos_scores):
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    grid.fit(embeddings, mos_scores)
    best_alpha = grid.best_params_['alpha']
    print(f"Optimal alpha: {best_alpha}")
    return Ridge(alpha=best_alpha).fit(embeddings, mos_scores)

def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch in train_dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # 입력이 5D 텐서일 경우 4D로 변환
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, channels, height, width]

            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            # `mos` 값 반복 (inputs_A 크기에 맞춰 조정)
            repeat_factor = features_A.shape[0] // mos.shape[0]  # inputs_A와 mos의 크기 비율 계산
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            embeddings.append(features_A)
            mos_scores.append(mos_repeated)

    # 리스트를 numpy 배열로 변환
    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # 크기 검증
    assert embeddings.shape[0] == mos_scores.shape[0], \
        f"Mismatch in embeddings ({embeddings.shape[0]}) and MOS scores ({mos_scores.shape[0]})"

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, mos_scores)
    print("Ridge Regressor Trained: Optimal alpha=1.0")
    return ridge




def evaluate_ridge_regressor(regressor, model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # Flatten crops if needed

            # Use backbone features for Ridge prediction
            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            prediction = regressor.predict(features_A)  # Use backbone features
            repeat_factor = features_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            predictions.extend(prediction)
            mos_scores.extend(mos_repeated)

    mos_scores = np.array(mos_scores)
    predictions = np.array(predictions)
    assert mos_scores.shape == predictions.shape, \
        f"Mismatch between MOS ({mos_scores.shape}) and Predictions ({predictions.shape})"

    return mos_scores, predictions




def plot_results(mos_scores, predictions):
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()

def debug_embeddings(embeddings, title="Embeddings"):
    plt.figure(figsize=(8, 6))
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7)
    plt.title(f"{title} Distribution")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # proj_A가 튜플인지 확인
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # proj_A만 사용

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics

def test_hard_negative_attention(args, model, dataloader, device):
    model.eval()
    metrics = {'srcc': [], 'plcc': []}

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize the embeddings
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            # Compute SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            metrics['srcc'].append(srocc)
            metrics['plcc'].append(plcc)

    return metrics


if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path)
    dataset = TID2013Dataset(str(dataset_path))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    model = SimCLR(embedding_dim=128, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    )

    # Train
    train_metrics, val_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Test
    test_metrics = test(args, model, test_dataloader, device)

    print("KADID10K")
    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)

 """

# TID2013
# Training Metrics: {'loss': [0.737725217685555, 0.7462933761152354, 0.7331488028620229, 0.728813994337212, 0.7262638802781249, 0.7203712382099845, 0.7052614436005101, 0.6887555192365791, 0.6921453360806812, 0.7010436071590944]}
# Validation Metrics: {'srcc': [0.9355765201783354, 0.9271368421769867, 0.933620621790592, 0.933241494979803, 0.9324303495644619, 0.9317366455243902, 0.9332166350691363, 0.933214911379564, 0.9305469230903206, 0.9266090703519732], 'plcc': [0.9397409227816345, 0.9314102412624932, 0.9379510408039685, 0.9376009360227731, 0.9360172191086796, 0.9362024132206633, 0.9365501446472317, 0.9373739779775768, 0.9339405153216597, 0.9316460471373]}
# Test Metrics: {'srcc': 0.93056694121473, 'plcc': 0.9363675209819101}


# SPAQ
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import SPAQDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


def verify_hard_negatives(original_shape, downscaled_shape):
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def debug_ridge_regressor(embeddings, mos_scores):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], mos_scores, alpha=0.7)
    plt.xlabel('Embedding Feature 0')
    plt.ylabel('MOS Scores')
    plt.title('Embedding vs MOS Scores')
    plt.grid()
    plt.show()

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)


def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # Debug: Log hard_negatives shape
            print(f"[Debug] hard_negatives shape before processing: {hard_negatives.shape}")

            if hard_negatives.dim() == 5:
                hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:]).to(device)
            elif hard_negatives.dim() != 4:
                raise ValueError(f"[Error] Unexpected hard_negatives dimensions: {hard_negatives.shape}")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)

                # Normalize projections
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                # Backbone processing for hard negatives
                features_negatives = model.backbone(hard_negatives)
                print(f"[Debug] features_negatives shape after backbone: {features_negatives.shape}")

                if features_negatives.dim() == 4:
                    features_negatives = features_negatives.mean([2, 3])
                elif features_negatives.dim() != 2:
                    raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

                proj_negatives = F.normalize(model.projector(features_negatives), dim=1)

                # Compute loss
                loss = model.compute_loss(proj_A, proj_B, proj_negatives)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics




def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics


def optimize_ridge_alpha(embeddings, mos_scores):
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    grid.fit(embeddings, mos_scores)
    best_alpha = grid.best_params_['alpha']
    print(f"Optimal alpha: {best_alpha}")
    return Ridge(alpha=best_alpha).fit(embeddings, mos_scores)

def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch in train_dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # 입력이 5D 텐서일 경우 4D로 변환
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, channels, height, width]

            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            # `mos` 값 반복 (inputs_A 크기에 맞춰 조정)
            repeat_factor = features_A.shape[0] // mos.shape[0]  # inputs_A와 mos의 크기 비율 계산
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            embeddings.append(features_A)
            mos_scores.append(mos_repeated)

    # 리스트를 numpy 배열로 변환
    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # 크기 검증
    assert embeddings.shape[0] == mos_scores.shape[0], \
        f"Mismatch in embeddings ({embeddings.shape[0]}) and MOS scores ({mos_scores.shape[0]})"

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, mos_scores)
    print("Ridge Regressor Trained: Optimal alpha=1.0")
    return ridge




def evaluate_ridge_regressor(regressor, model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # Flatten crops if needed

            # Use backbone features for Ridge prediction
            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            prediction = regressor.predict(features_A)  # Use backbone features
            repeat_factor = features_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            predictions.extend(prediction)
            mos_scores.extend(mos_repeated)

    mos_scores = np.array(mos_scores)
    predictions = np.array(predictions)
    assert mos_scores.shape == predictions.shape, \
        f"Mismatch between MOS ({mos_scores.shape}) and Predictions ({predictions.shape})"

    return mos_scores, predictions




def plot_results(mos_scores, predictions):
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()

def debug_embeddings(embeddings, title="Embeddings"):
    plt.figure(figsize=(8, 6))
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7)
    plt.title(f"{title} Distribution")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # proj_A가 튜플인지 확인
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # proj_A만 사용

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics

def test_hard_negative_attention(args, model, dataloader, device):
    model.eval()
    metrics = {'srcc': [], 'plcc': []}

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize the embeddings
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            # Compute SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            metrics['srcc'].append(srocc)
            metrics['plcc'].append(plcc)

    return metrics


if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path)
    dataset = SPAQDataset(str(dataset_path))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    model = SimCLR(embedding_dim=128, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    )

    # Train
    train_metrics, val_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Test
    test_metrics = test(args, model, test_dataloader, device)

    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)

 """  
# SPAQ
# Training Metrics: {'loss': [0.96711675691164, 0.9384566204014255, 0.9394465068649707, 0.9328271435761109, 0.9258187807622142, 0.929989348630396, 0.8976760837698864, 0.8835711794039063, 0.8895116938580233, 0.888515325328163]}
# Validation Metrics: {'srcc': [0.8977806072754907, 0.8950514637981528, 0.9047216123379043, 0.899351765701469, 0.8888553802993773, 0.9094666710732963, 0.8958504407247477, 0.9074612159540477, 0.8958127595333932, 0.9047504578053197], 'plcc': [0.903272634484041, 0.901508086740527, 0.9099186474798078, 0.9044558455655598, 0.8959942004778992, 0.9141912384646408, 0.9009320265729011, 0.9118379420133087, 0.9012383365341987, 0.9088791369229572]}
# Test Metrics: {'srcc': 0.898468345807046, 'plcc': 0.9027940528662981}

# CSIQ
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import CSIQDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


def verify_hard_negatives(original_shape, downscaled_shape):
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def debug_ridge_regressor(embeddings, mos_scores):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], mos_scores, alpha=0.7)
    plt.xlabel('Embedding Feature 0')
    plt.ylabel('MOS Scores')
    plt.title('Embedding vs MOS Scores')
    plt.grid()
    plt.show()

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)


def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # Debug: Log hard_negatives shape
            print(f"[Debug] hard_negatives shape before processing: {hard_negatives.shape}")

            if hard_negatives.dim() == 5:
                hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:]).to(device)
            elif hard_negatives.dim() != 4:
                raise ValueError(f"[Error] Unexpected hard_negatives dimensions: {hard_negatives.shape}")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)

                # Normalize projections
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                # Backbone processing for hard negatives
                features_negatives = model.backbone(hard_negatives)
                print(f"[Debug] features_negatives shape after backbone: {features_negatives.shape}")

                if features_negatives.dim() == 4:
                    features_negatives = features_negatives.mean([2, 3])
                elif features_negatives.dim() != 2:
                    raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

                proj_negatives = F.normalize(model.projector(features_negatives), dim=1)

                # Compute loss
                loss = model.compute_loss(proj_A, proj_B, proj_negatives)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics




def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics


def optimize_ridge_alpha(embeddings, mos_scores):
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    grid.fit(embeddings, mos_scores)
    best_alpha = grid.best_params_['alpha']
    print(f"Optimal alpha: {best_alpha}")
    return Ridge(alpha=best_alpha).fit(embeddings, mos_scores)

def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch in train_dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # 입력이 5D 텐서일 경우 4D로 변환
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, channels, height, width]

            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            # `mos` 값 반복 (inputs_A 크기에 맞춰 조정)
            repeat_factor = features_A.shape[0] // mos.shape[0]  # inputs_A와 mos의 크기 비율 계산
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            embeddings.append(features_A)
            mos_scores.append(mos_repeated)

    # 리스트를 numpy 배열로 변환
    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # 크기 검증
    assert embeddings.shape[0] == mos_scores.shape[0], \
        f"Mismatch in embeddings ({embeddings.shape[0]}) and MOS scores ({mos_scores.shape[0]})"

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, mos_scores)
    print("Ridge Regressor Trained: Optimal alpha=1.0")
    return ridge




def evaluate_ridge_regressor(regressor, model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # Flatten crops if needed

            # Use backbone features for Ridge prediction
            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            prediction = regressor.predict(features_A)  # Use backbone features
            repeat_factor = features_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            predictions.extend(prediction)
            mos_scores.extend(mos_repeated)

    mos_scores = np.array(mos_scores)
    predictions = np.array(predictions)
    assert mos_scores.shape == predictions.shape, \
        f"Mismatch between MOS ({mos_scores.shape}) and Predictions ({predictions.shape})"

    return mos_scores, predictions




def plot_results(mos_scores, predictions):
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()

def debug_embeddings(embeddings, title="Embeddings"):
    plt.figure(figsize=(8, 6))
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7)
    plt.title(f"{title} Distribution")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # proj_A가 튜플인지 확인
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # proj_A만 사용

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics

def test_hard_negative_attention(args, model, dataloader, device):
    model.eval()
    metrics = {'srcc': [], 'plcc': []}

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize the embeddings
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            # Compute SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            metrics['srcc'].append(srocc)
            metrics['plcc'].append(plcc)

    return metrics


if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path)
    dataset = CSIQDataset(str(dataset_path))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    model = SimCLR(embedding_dim=128, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    )

    # Train
    train_metrics, val_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Test
    test_metrics = test(args, model, test_dataloader, device)

    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)

 """
# 수정 전
# Training Metrics: {'loss': [0.9854135513305664, 1.0046942061499546, 0.9639555466802496, 0.9972633641017111, 0.9706209223521384, 0.9797979781502172, 0.9859969161058727, 1.0065353575505709, 0.9694811842943493, 
# 0.9977130215418967]}
# Validation Metrics: {'srcc': [0.9178016371160677, 0.9255205856474106, 0.9217857082294895, 0.9125314641254821, 0.9184603668818808, 0.9219749199373433, 0.9137176581021379, 0.9164256096231672, 0.9206358148005287, 0.906857331979635], 'plcc': [0.92188525200628, 0.9287617215861261, 0.9249549866929204, 0.916230252939191, 0.9222246287816857, 0.9260194456952809, 0.9182252958867384, 0.9201609753554676, 0.9244511587061747, 
# 0.912330114772392]}
# Test Metrics: {'srcc': 0.9326779531514094, 'plcc': 0.9365337252857805}

# LIVE
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import LIVEDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B, applied_distortions_A, applied_distortions_B):
    print(f"[Debug] Verifying Positive Pairs:")
    print(f" - Distortion A: {distortions_A}, Distortion B: {distortions_B}")
    print(f" - Applied Level A: {applied_distortions_A}, Applied Level B: {applied_distortions_B}")

    if distortions_A == distortions_B and applied_distortions_A == applied_distortions_B:
        print(f"[Positive Pair Verification] Success: Distortions match.")
    else:
        print(f"[Positive Pair Verification] Error: Distortions do not match.")


def verify_hard_negatives(original_shape, downscaled_shape):
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def debug_ridge_regressor(embeddings, mos_scores):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], mos_scores, alpha=0.7)
    plt.xlabel('Embedding Feature 0')
    plt.ylabel('MOS Scores')
    plt.title('Embedding vs MOS Scores')
    plt.grid()
    plt.show()

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)


def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # Debug: Log hard_negatives shape
            print(f"[Debug] hard_negatives shape before processing: {hard_negatives.shape}")

            if hard_negatives.dim() == 5:
                hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:]).to(device)
            elif hard_negatives.dim() != 4:
                raise ValueError(f"[Error] Unexpected hard_negatives dimensions: {hard_negatives.shape}")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)

                # Normalize projections
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                # Backbone processing for hard negatives
                features_negatives = model.backbone(hard_negatives)
                print(f"[Debug] features_negatives shape after backbone: {features_negatives.shape}")

                if features_negatives.dim() == 4:
                    features_negatives = features_negatives.mean([2, 3])
                elif features_negatives.dim() != 2:
                    raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

                proj_negatives = F.normalize(model.projector(features_negatives), dim=1)

                # Compute loss
                loss = model.compute_loss(proj_A, proj_B, proj_negatives)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics




def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics


def optimize_ridge_alpha(embeddings, mos_scores):
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    grid.fit(embeddings, mos_scores)
    best_alpha = grid.best_params_['alpha']
    print(f"Optimal alpha: {best_alpha}")
    return Ridge(alpha=best_alpha).fit(embeddings, mos_scores)

def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch in train_dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # 입력이 5D 텐서일 경우 4D로 변환
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, channels, height, width]

            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            # `mos` 값 반복 (inputs_A 크기에 맞춰 조정)
            repeat_factor = features_A.shape[0] // mos.shape[0]  # inputs_A와 mos의 크기 비율 계산
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            embeddings.append(features_A)
            mos_scores.append(mos_repeated)

    # 리스트를 numpy 배열로 변환
    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # 크기 검증
    assert embeddings.shape[0] == mos_scores.shape[0], \
        f"Mismatch in embeddings ({embeddings.shape[0]}) and MOS scores ({mos_scores.shape[0]})"

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, mos_scores)
    print("Ridge Regressor Trained: Optimal alpha=1.0")
    return ridge




def evaluate_ridge_regressor(regressor, model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # Flatten crops if needed

            # Use backbone features for Ridge prediction
            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()  # GAP 적용

            prediction = regressor.predict(features_A)  # Use backbone features
            repeat_factor = features_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)[:features_A.shape[0]]

            predictions.extend(prediction)
            mos_scores.extend(mos_repeated)

    mos_scores = np.array(mos_scores)
    predictions = np.array(predictions)
    assert mos_scores.shape == predictions.shape, \
        f"Mismatch between MOS ({mos_scores.shape}) and Predictions ({predictions.shape})"

    return mos_scores, predictions




def plot_results(mos_scores, predictions):
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()

def debug_embeddings(embeddings, title="Embeddings"):
    plt.figure(figsize=(8, 6))
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7)
    plt.title(f"{title} Distribution")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def evaluate_zero_shot(model, unseen_dataset, device):
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
    metrics = {'srcc': [], 'plcc': []}

    for batch in unseen_dataloader:
        inputs_A = batch["img_A"].to(device)
        mos = batch["mos"].to(device)
        
        # 모델 출력 계산
        proj_A = model(inputs_A)

        # proj_A가 튜플인지 확인
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # proj_A만 사용

        # 크기 로그 출력
        print(f"[Debug] mos size: {mos.shape}, proj_A size: {proj_A.shape}")

        # proj_A 크기 조정
        batch_size = mos.shape[0]
        proj_A_mean = proj_A.view(batch_size, -1, proj_A.shape[-1]).mean(dim=1)
        proj_A_final = proj_A_mean.mean(dim=1)  # 최종적으로 [batch_size] 형태로 축소

        # 크기 맞춤 후 로그 출력
        print(f"[Debug] proj_A_final size after adjustment: {proj_A_final.shape}")

        # Positive Pair Verification 호출
        verify_positive_pairs(
            distortions_A="distortion_type_A",
            distortions_B="distortion_type_B",
            applied_distortions_A="applied_type_A",
            applied_distortions_B="applied_type_B"
        )

        # detach()를 사용해 그래프에서 분리
        mos_np = mos.cpu().detach().numpy()
        proj_A_np = proj_A_final.cpu().detach().numpy()

        # SRCC 및 PLCC 계산
        srcc, _ = stats.spearmanr(mos_np, proj_A_np)
        plcc, _ = stats.pearsonr(mos_np, proj_A_np)

        metrics['srcc'].append(srcc)
        metrics['plcc'].append(plcc)

    return metrics

def test_hard_negative_attention(args, model, dataloader, device):
    model.eval()
    metrics = {'srcc': [], 'plcc': []}

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize the embeddings
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            # Compute SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            metrics['srcc'].append(srocc)
            metrics['plcc'].append(plcc)

    return metrics


if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path)
    dataset = LIVEDataset(str(dataset_path))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    model = SimCLR(embedding_dim=128, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min
    )

    # Train
    train_metrics, val_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Test
    test_metrics = test(args, model, test_dataloader, device)

    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)
 """

#LIVE
# Training Metrics: {'loss': [1.3416166375665104, 1.385161556449591, 1.3430749528548296, 1.344389720290315, 1.3500481271276288, 1.311995915338105, 1.362798905840107, 1.3029328467799168, 1.3513919080004972, 1.3905024236323786]}
# Validation Metrics: {'srcc': [0.8728831272297886, 0.8926784017534533, 0.8463808868438731, 0.8430774209611263, 0.8814598820151454, 0.8783217775462138, 0.8604482425330922, 0.8722422826434939, 0.8823484322066615, 0.8497454245955931], 'plcc': [0.8802752573302571, 0.8982301779839197, 0.8545239825590596, 0.8501004006982258, 0.8879407990530739, 0.8844903435674625, 0.868455748679986, 0.8784393091916264, 0.8898163848477165, 0.8597359100276125]}
# Test Metrics: {'srcc': 0.844997775637989, 'plcc': 0.8515623023230324}


# train.py

# Train(KADID) & Test(TID2013)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset, TID2013Dataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")
    tid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/mos.csv")

    # 데이터셋 로드
    kadid_dataset = KADID10KDataset(kadid_dataset_path)
    tid_dataset = TID2013Dataset(tid_dataset_path)

    # KADID10K 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(kadid_dataset))
    val_size = len(kadid_dataset) - train_size
    train_dataset, val_dataset = random_split(kadid_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        tid_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # KADID10K로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # TID2013으로 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(KADID10K) & Test(TID2013)")
    print(f"Test Results on TID2013 Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))

 """


# Train(KADID10K) & Test(TID2013)
# Test Results on TID2013 Dataset: SRCC = 0.9446, PLCC = 0.9501
# 
# Training Metrics: {'loss': [1.0261, 1.0186, 1.0079, 1.0192, 1.0054, 1.0113, 1.0078, 1.0082, 1.0043, 1.0074]}
# Validation Metrics: {'srcc': [0.9412, 0.9417, 0.9387, 0.9419, 0.9397, 0.9408, 0.9377, 0.9397, 0.9383, 0.9424], 'plcc': [0.9462, 0.9455, 0.9431, 0.9454, 0.944, 0.9449, 0.9417, 0.9446, 0.9429, 0.9462]}        




# Train(KADID) & Test(CSIQ)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset, CSIQDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")
    csiq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/CSIQ.txt")

    # 데이터셋 로드
    kadid_dataset = KADID10KDataset(kadid_dataset_path)
    csiq_dataset = CSIQDataset(csiq_dataset_path)

    # KADID10K 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(kadid_dataset))
    val_size = len(kadid_dataset) - train_size
    train_dataset, val_dataset = random_split(kadid_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        csiq_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # KADID10K로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # CSIQ으로 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(KADID10K) & Test(CSIQ)")
    print(f"Test Results on CSIQ Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))
 """


# Train(KADID10K) & Test(CSIQ)
# Test Results on CSIQ Dataset: SRCC = 0.9145, PLCC = 0.9189
# 
# Training Metrics: {'loss': [0.8817, 0.8806, 0.8767, 0.8744, 0.8801, 0.8734, 0.8682, 0.8764, 0.8712, 0.8657]}
# Validation Metrics: {'srcc': [0.9356, 0.9361, 0.9342, 0.9345, 0.9382, 0.9357, 0.9355, 0.9336, 0.9362, 0.9349], 'plcc': [0.9394, 0.9396, 0.9379, 0.938, 0.9418, 0.9396, 0.9397, 0.9372, 0.9394, 0.9387]}        


# Train(TID2013) & Test(KADID)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import TID2013Dataset, KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    tid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/mos.csv")
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")

    # 데이터셋 로드
    tid_dataset = TID2013Dataset(tid_dataset_path)
    kadid_dataset = KADID10KDataset(kadid_dataset_path)

    # tid 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(tid_dataset))
    val_size = len(tid_dataset) - train_size
    train_dataset, val_dataset = random_split(tid_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        kadid_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # KADID10K로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # KADID으로 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(tid) & Test(KADID)")
    print(f"Test Results on KADID Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))

 """

#Train(tid) & Test(KADID)
#Test Results on KADID Dataset: SRCC = 0.9325, PLCC = 0.9413
#
#Training Metrics: {'loss': [1.0203, 1.0392, 1.0246, 1.0268, 1.0129, 1.0186, 1.02, 1.0165, 1.0129, 1.03]}
#Validation Metrics: {'srcc': [0.934, 0.9335, 0.9358, 0.9361, 0.9361, 0.9369, 0.9333, 0.9361, 0.9359, 0.9339], 'plcc': [0.9423, 0.9412, 0.9429, 0.9435, 0.944, 0.9441, 0.9409, 0.9437, 0.943, 0.9421]}





# Train(TID2013) & Test(CSIQ)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import TID2013Dataset, CSIQDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    tid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/mos.csv")
    csiq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/CSIQ.txt")

    # 데이터셋 로드
    tid_dataset = TID2013Dataset(tid_dataset_path)
    csiq_dataset = CSIQDataset(csiq_dataset_path)

    # tid 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(tid_dataset))
    val_size = len(tid_dataset) - train_size
    train_dataset, val_dataset = random_split(tid_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        csiq_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # KADID10K로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # TID2013으로 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(tid) & Test(CSIQ)")
    print(f"Test Results on CSIQ Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))

 """
# Train(tid) & Test(CSIQ)
# Test Results on CSIQ Dataset: SRCC = 0.9394, PLCC = 0.9450
# 
# Training Metrics: {'loss': [1.2744, 1.2819, 1.2838, 1.2646, 1.2665, 1.2782, 1.2755, 1.2767, 1.2661, 1.2795]}
# Validation Metrics: {'srcc': [0.9461, 0.9446, 0.943, 0.9447, 0.9455, 0.9444, 0.9448, 0.947, 0.9464, 0.9462], 'plcc': [0.951, 0.9497, 0.948, 0.9494, 0.951, 0.9492, 0.9495, 0.9516, 0.9512, 0.9511]}

# Train(KADID) & Test(SPAQ)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset, SPAQDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")
    spaq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ")

    # 데이터셋 로드
    kadid_dataset = KADID10KDataset(kadid_dataset_path)
    spaq_dataset = SPAQDataset(spaq_dataset_path)

    # kadid 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(kadid_dataset))
    val_size = len(kadid_dataset) - train_size
    train_dataset, val_dataset = random_split(kadid_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        spaq_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # KADID10K로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # SPAQ 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(KADID) & Test(SPAQ)")
    print(f"Test Results on SPAQ Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))
 """

# Train(KADID10K) & Test(SPAQ)
# 
# Test Results on SPAQ Dataset: SRCC = 0.8539, PLCC = 0.8598
# 
# Training Metrics: {'loss': [0.1345, 0.1331, 0.1333, 0.1277, 0.1253, 0.1254, 0.1267, 0.123, 0.1162, 0.1187], 'srcc': [0.9144, 0.9139, 0.9134, 0.9144, 0.9148, 0.9136, 0.9142, 0.9144, 0.917, 0.9144], 'plcc': [0.9178, 0.9172, 0.9168, 0.9178, 0.918, 0.9168, 0.9175, 0.9177, 0.9202, 0.9177]}
# Validation Metrics: {'srcc': [0.9013, 0.8976, 0.8993, 0.9019, 0.9062, 0.8924, 0.9027, 0.9024, 0.9026, 0.9036], 'plcc': [0.9059, 0.9026, 0.9033, 0.9063, 0.9114, 0.8987, 0.9074, 0.9068, 0.9067, 0.9091]}       

#Train(KADID) & Test(SPAQ)
#Test Results on SPAQ Dataset: SRCC = 0.8931, PLCC = 0.8945
#
#Training Metrics: {'loss': [0.8436, 0.838, 0.8468, 0.8382, 0.8294, 0.8375, 0.8419, 0.8351, 0.8311, 0.8378]}
#Validation Metrics: {'srcc': [0.9393, 0.9402, 0.9386, 0.9369, 0.9366, 0.9373, 0.9395, 0.9371, 0.9348, 0.9384], 'plcc': [0.9415, 0.9423, 0.941, 0.9404, 0.9392, 0.94, 0.9424, 0.9404, 0.9373, 0.941]}



# Train(CSIQ) & Test(KADID)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import CSIQDataset, KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    csiq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/CSIQ.txt")
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")
    

    # 데이터셋 로드
    csiq_dataset = CSIQDataset(csiq_dataset_path)
    kadid_dataset = KADID10KDataset(kadid_dataset_path)
    

    # csiq 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(csiq_dataset))
    val_size = len(csiq_dataset) - train_size
    train_dataset, val_dataset = random_split(csiq_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        kadid_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # CSIQ로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # KADID 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(CSIQ) & Test(KADID)")
    print(f"Test Results on KADID Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))
 """

# Train(CSIQ) & Test(KADID)
# Test Results on KADID Dataset: SRCC = 0.9434, PLCC = 0.9477
# 
# Training Metrics: {'loss': [1.4063, 1.4354, 1.4264, 1.4479, 1.3957, 1.3953, 1.4025, 1.4366, 1.4204, 1.4347]}
# Validation Metrics: {'srcc': [0.9316, 0.9221, 0.9235, 0.9278, 0.9274, 0.9325, 0.9305, 0.9284, 0.9261, 0.9266], 'plcc': [0.9361, 0.9265, 0.9275, 0.932, 0.9317, 0.9368, 0.9347, 0.9329, 0.9307, 0.9304]}   



# Train(CSIQ) & Test(TID)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import CSIQDataset, TID2013Dataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    csiq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/CSIQ.txt")
    tid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/mos.csv")
    

    # 데이터셋 로드
    csiq_dataset = CSIQDataset(csiq_dataset_path)
    tid_dataset = TID2013Dataset(tid_dataset_path)
    

    # csiq 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(csiq_dataset))
    val_size = len(csiq_dataset) - train_size
    train_dataset, val_dataset = random_split(csiq_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        tid_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # CSIQ로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # tid 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(CSIQ) & Test(TID)")
    print(f"Test Results on TID Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))
 """

# Train(CSIQ) & Test(TID)
# Test Results on TID Dataset: SRCC = 0.9378, PLCC = 0.9415
# 
# Training Metrics: {'loss': [1.0927, 1.0839, 1.0915, 1.0445, 1.0896, 1.0637, 1.0367, 1.0951, 1.0847, 1.0735]}
# Validation Metrics: {'srcc': [0.9108, 0.9103, 0.9188, 0.917, 0.9169, 0.9165, 0.9144, 0.9124, 0.9146, 0.9067], 'plcc': [0.9141, 0.9142, 0.9209, 0.9203, 0.9194, 0.9195, 0.9173, 0.9151, 0.9176, 0.9096]}        





# Train(CSIQ) & Test(SPAQ)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import CSIQDataset, SPAQDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    csiq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/CSIQ.txt")
    spaq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ")
    

    # 데이터셋 로드
    csiq_dataset = CSIQDataset(csiq_dataset_path)
    spaq_dataset = SPAQDataset(spaq_dataset_path)
    

    # csiq 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(csiq_dataset))
    val_size = len(csiq_dataset) - train_size
    train_dataset, val_dataset = random_split(csiq_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        spaq_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # CSIQ로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # tid 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(CSIQ) & Test(SPAQ)")
    print(f"Test Results on SPAQ Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))
 """

#Train(CSIQ) & Test(SPAQ)
#
#Test Results on SPAQ Dataset: SRCC = 0.8831, PLCC = 0.8896
#
#Training Metrics: {'loss': [0.4106, 0.4064, 0.4082, 0.3964, 0.4204, 0.4119, 0.3864, 0.4132, 0.3988, 0.3889], 'srcc': [0.8931, 0.8936, 0.8947, 0.8956, 0.8941, 0.8934, 0.8968, 0.8923, 0.8946, 0.8969], 'plcc': 
#[0.9037, 0.9042, 0.9048, 0.9053, 0.9042, 0.9035, 0.9067, 0.9026, 0.9046, 0.9068]}
#Validation Metrics: {'srcc': [0.8605, 0.8766, 0.8751, 0.8767, 0.8814, 0.8761, 0.8749, 0.8737, 0.8793, 0.878], 'plcc': [0.8717, 0.8851, 0.8849, 0.8856, 0.8888, 0.8854, 0.8851, 0.8845, 0.8886, 0.8871]}        



# Train(CSIQ) & Test(SPAQ)
# Test Results on SPAQ Dataset: SRCC = 0.9020, PLCC = 0.9047
# 
# Training Metrics: {'loss': [1.2032, 1.1933, 1.1784, 1.2125, 1.196, 1.217, 1.1711, 1.21, 1.1944, 1.2102]}
# Validation Metrics: {'srcc': [0.9245, 0.9244, 0.9232, 0.9191, 0.9203, 0.9192, 0.9245, 0.9272, 0.9254, 0.92], 'plcc': [0.9274, 0.927, 0.9261, 0.9224, 0.9234, 0.922, 0.9279, 0.9297, 0.9283, 0.9228]}



# Train(SPAQ) & Test(TID2013)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import SPAQDataset, TID2013Dataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    spaq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ")
    tid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/mos.csv")
    

    # 데이터셋 로드
    spaq_dataset = SPAQDataset(spaq_dataset_path)
    tid_dataset = CSIQDataset(tid_dataset_path)
    
    

    # spaq 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(csiq_dataset))
    val_size = len(csiq_dataset) - train_size
    train_dataset, val_dataset = random_split(csiq_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        tid_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # spaq로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # tid 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(SPAQ) & Test(TID)")
    print(f"Test Results on TID2013 Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))

 """


# Test Results on TID2013 Dataset: SRCC = 0.9233, PLCC = 0.9270
# 
# Training Metrics: {'loss': [0.3467, 0.3222, 0.3299, 0.2938, 0.2956, 0.2939, 0.2813, 0.2725, 0.2557, 0.2528], 'srcc': [0.892, 0.8937, 0.8871, 0.8962, 0.8929, 0.8893, 0.8928, 0.8943, 0.8942, 0.8905], 'plcc': [0.8947, 0.8963, 0.8897, 0.8986, 0.8954, 0.8919, 0.8952, 0.8968, 0.8966, 0.8928]}
# Validation Metrics: {'srcc': [0.877, 0.8869, 0.8896, 0.8836, 0.888, 0.8874, 0.8995, 0.8748, 0.8898, 0.8761], 'plcc': [0.8771, 0.8873, 0.8918, 0.8854, 0.8887, 0.8889, 0.9014, 0.8768, 0.8904, 0.8775]}



# Train(SPAQ) & Test(KADID)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import SPAQDataset, KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    spaq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ")
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")
    

    # 데이터셋 로드
    spaq_dataset = SPAQDataset(spaq_dataset_path)
    kadid_dataset = CSIQDataset(kadid_dataset_path)
    
    

    # spaq 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(csiq_dataset))
    val_size = len(csiq_dataset) - train_size
    train_dataset, val_dataset = random_split(csiq_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        kadid_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # spaq로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # tid 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(SPAQ) & Test(KADID)")
    print(f"Test Results on KADID Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))

 """

# Train(SPAQ) & Test(KADID)
# Test Results on KADID Dataset: SRCC = 0.9085, PLCC = 0.9159
# 
# Training Metrics: {'loss': [1.0839, 1.1153, 1.0932, 1.0482, 1.075, 1.1316, 1.0844, 1.1003, 1.0742, 1.0943]}
# Validation Metrics: {'srcc': [0.9225, 0.9185, 0.9169, 0.9156, 0.9208, 0.9162, 0.9157, 0.9231, 0.921, 0.9218], 'plcc': [0.9285, 0.925, 0.9236, 0.922, 0.9266, 0.923, 0.9229, 0.9293, 0.9277, 0.9276]}

# Train(SPAQ) & Test(CSIQ)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import SPAQDataset, CSIQDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    spaq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ")
    csiq_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/CSIQ/CSIQ.txt")
    

    # 데이터셋 로드
    spaq_dataset = SPAQDataset(spaq_dataset_path)
    csiq_dataset = CSIQDataset(csiq_dataset_path)
    
    

    # spaq 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(csiq_dataset))
    val_size = len(csiq_dataset) - train_size
    train_dataset, val_dataset = random_split(csiq_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        csiq_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # csiq로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # tid 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(SPAQ) & Test(CSIQ)")
    print(f"Test Results on CSIQ Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics)) """


# Train(SPAQ) & Test(CSIQ)
# Test Results on CSIQ Dataset: SRCC = 0.9170, PLCC = 0.9225
# 
# Training Metrics: {'loss': [1.4105, 1.3714, 1.3723, 1.3916, 1.3368, 1.3665, 1.3561, 1.3336, 1.3587, 1.3679]}
# Validation Metrics: {'srcc': [0.9299, 0.9302, 0.9282, 0.9287, 0.9326, 0.9272, 0.9279, 0.9316, 0.9329, 0.9231], 'plcc': [0.9337, 0.9344, 0.932, 0.9328, 0.9366, 0.9314, 0.9328, 0.935, 0.9367, 0.9272]}



# Train(TID2013) & Test(SPAQ)
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import TID2013Dataset, SPAQDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc = np.mean(srocc_values)
    avg_plcc = np.mean(plcc_values)

    return avg_srocc, avg_plcc

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = 0

    train_metrics = {'loss': [], 'srcc': [], 'plcc': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        srocc_values, plcc_values = [], []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                # Process inputs_A and inputs_B through model
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                # Generate SE weights from features_A
                features_A = model.backbone(inputs_A)
                se_weights = features_A.mean(dim=[2, 3])  # SE weights for inputs_A

                # Generate hard negatives
                features_B = model.backbone(inputs_B)  # Use inputs_B for negatives
                features_B = features_B.mean([2, 3])  # GAP for negatives
                proj_negatives = model.projector(features_B)
                proj_negatives = F.normalize(proj_negatives, dim=1)

                # Compute loss with SE weights and negatives
                loss = model.compute_loss(proj_A, proj_B, proj_negatives, se_weights)

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

            # Calculate SRCC and PLCC for the batch
            srocc, _ = stats.spearmanr(proj_A.detach().cpu().numpy().flatten(), proj_B.detach().cpu().numpy().flatten())
            plcc, _ = stats.pearsonr(proj_A.detach().cpu().numpy().flatten(), proj_B.detach().cpu().numpy().flatten())
            srocc_values.append(srocc)
            plcc_values.append(plcc)

        train_metrics['loss'].append(running_loss / len(train_dataloader))
        train_metrics['srcc'].append(np.mean(srocc_values))
        train_metrics['plcc'].append(np.mean(plcc_values))

        lr_scheduler.step()

        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(avg_srocc_val)
        val_metrics['plcc'].append(avg_plcc_val)

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Training complete. Best SRCC:", best_srocc)
    return train_metrics, val_metrics


def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}



if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # TID2013 Dataset
    tid_dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/mos.csv"
    tid_dataset = TID2013Dataset(Path(tid_dataset_path))

    train_size = int(0.7 * len(tid_dataset))
    val_size = int(0.1 * len(tid_dataset))
    test_size = len(tid_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(tid_dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

    # SPAQDataset (Test)
    SPAQDataset_dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ"
    SPAQDataset_dataset = SPAQDataset(Path(SPAQDataset_dataset_path))
    test_dataloader = DataLoader(SPAQDataset_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

    # Model initialization
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # Training on TID2013
    train_metrics, val_metrics = train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    # Testing on KADID10K
    test_results = test(args, model, test_dataloader, device)
    print("Train(TID2013) & Test(SPAQ)")
    print(f"\nTest Results on SPAQ Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))

 """
# Train(TID2013) & Test(SPAQ)
# 
# Test Results on SPAQ Dataset: SRCC = 0.8795, PLCC = 0.8852
# 
# Training Metrics: {'loss': [0.2234, 0.2313, 0.232, 0.225, 0.2274, 0.2188, 0.22, 0.2346, 0.2212, 0.2267], 'srcc': [0.9252, 0.9212, 0.921, 0.9229, 0.9226, 0.9222, 0.9233, 0.92, 0.9233, 0.921], 'plcc': [0.9327, 0.9287, 0.9286, 0.9305, 0.9301, 0.9299, 0.9308, 0.9275, 0.9308, 0.9287]}
# Validation Metrics: {'srcc': [0.9162, 0.9071, 0.9079, 0.9072, 0.909, 0.9076, 0.905, 0.91, 0.905, 0.8996], 'plcc': [0.9237, 0.9158, 0.9166, 0.9154, 0.917, 0.9166, 0.9128, 0.9182, 0.9139, 0.9076]}


# Train(KADID) & Test(LIVE)
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset, LIVEDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            # 모델의 입력 채널 크기에 맞게 변환
            if hard_negatives.shape[1] != inputs_A.shape[1]:  # 입력 채널 확인
                hard_negatives = hard_negatives.view(-1, inputs_A.shape[1], *hard_negatives.shape[2:]).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_A = model.backbone(inputs_A).mean([2, 3])
                proj_negatives = F.normalize(model.projector(model.backbone(hard_negatives).mean([2, 3])), dim=1)

                loss = model.compute_loss(proj_A, proj_B, proj_negatives, features_A)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss. Skipping batch.")
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        # Epoch 평균 손실 기록
        avg_loss = running_loss / len(train_dataloader)
        train_metrics['loss'].append(avg_loss)

        # Validation
        val_srocc, val_plcc = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(val_srocc)
        val_metrics['plcc'].append(val_plcc)

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics

def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (5D -> 4D)
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            if inputs_B.dim() == 5:
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc_test = np.mean(srocc_values)
    avg_plcc_test = np.mean(plcc_values)
    return {'srcc': avg_srocc_test, 'plcc': avg_plcc_test}

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # 디바이스 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")
    live_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/LIVE")
    

    # 데이터셋 로드
    
    kadid_dataset = KADID10KDataset(kadid_dataset_path)
    live_dataset = LIVEDataset(live_dataset_path)

    # tid 데이터셋을 Train/Validation으로 분할
    train_size = int(0.8 * len(live_dataset))
    val_size = len(live_dataset) - train_size
    train_dataset, val_dataset = random_split(live_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        live_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # SimCLR 모델 초기화
    embedding_dim = args.model.get("embedding_dim", 128)  # 기본값 128
    temperature = args.model.get("temperature", 0.07)  # 기본값 0.07

    model = SimCLR(embedding_dim=embedding_dim, temperature=temperature).to(device)

    # Optimizer 및 Learning Rate Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.training.lr_scheduler.T_0, T_mult=args.training.lr_scheduler.T_mult
    )
    scaler = torch.amp.GradScaler()

    # KADID10K로 학습
    train_metrics, val_metrics = train(
        args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device
    )

    # LIVE 테스트
    test_results = test(args, model, test_dataloader, device)
    print("Train(kadid) & Test(LIVE)")
    print(f"Test Results on LIVE Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    # 메트릭 출력 함수
    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))

 """
# Regressor 추출 코드 (KADID)
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import joblib
import yaml
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

# Save model checkpoint
def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {filename}")

# Save Ridge Regressor
def save_ridge_regressor(regressor, output_path: Path) -> None:
    filename = output_path / "ridge_regressor.pkl"
    joblib.dump(regressor, filename)
    print(f"Ridge Regressor saved: {filename}")

# Validation function
def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)
            proj_A = F.normalize(proj_A, dim=1).cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).cpu().numpy()

            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc = np.mean(srocc_values)
    avg_plcc = np.mean(plcc_values)

    return avg_srocc, avg_plcc

# Training function
def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = 0

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)  # [batch_size, num_crops, channels, height, width]
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed (reshape 5D to 4D)
            if inputs_A.dim() == 5:
                batch_size, num_crops, channels, height, width = inputs_A.shape
                inputs_A = inputs_A.view(-1, channels, height, width)  # [batch_size * num_crops, channels, height, width]
                inputs_B = inputs_B.view(-1, channels, height, width)

            # Generate hard negatives (proj_negatives)
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)
            hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:])
            proj_negatives = model.backbone(hard_negatives).mean([2, 3])  # Apply GAP
            proj_negatives = F.normalize(model.projector(proj_negatives), dim=1)

            # Calculate SE weights (se_weights)
            features_A = model.backbone(inputs_A)
            se_weights = features_A.mean(dim=[2, 3])  # Global Average Pooling

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                # Compute loss using proj_negatives and se_weights
                loss = model.compute_loss(proj_A, proj_B, proj_negatives, se_weights)

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / (i + 1))

        lr_scheduler.step()

        # Validation metrics
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Finished training")


# Train Ridge Regressor
def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch in train_dataloader:
            inputs_A = batch["img_A"].to(device)  # [batch_size, num_crops, channels, height, width]
            mos = batch["mos"]

            # Flatten crops if needed (reshape 5D to 4D)
            if inputs_A.dim() == 5:
                batch_size, num_crops, channels, height, width = inputs_A.shape
                inputs_A = inputs_A.view(-1, channels, height, width)  # [batch_size * num_crops, channels, height, width]

            # Extract features using the model's backbone
            features_A = model.backbone(inputs_A)  # [batch_size * num_crops, embedding_dim, height, width]
            features_A = features_A.mean([2, 3]).cpu().numpy()  # Global Average Pooling to [batch_size * num_crops, embedding_dim]

            # Repeat MOS scores to match features_A shape
            repeat_factor = features_A.shape[0] // mos.shape[0]  # Determine how many times to repeat each MOS value
            mos_repeated = np.repeat(mos.cpu().numpy(), repeat_factor)

            embeddings.append(features_A)
            mos_scores.append(mos_repeated)

    # Stack all embeddings and MOS scores
    embeddings = np.vstack(embeddings)  # [total_samples, embedding_dim]
    mos_scores = np.hstack(mos_scores)  # [total_samples]

    # Train Ridge Regressor
    regressor = Ridge(alpha=1.0)
    regressor.fit(embeddings, mos_scores)
    print("Ridge Regressor trained successfully.")
    return regressor

# Evaluate Ridge Regressor
def evaluate_ridge_regressor(regressor, model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            features_A = model.backbone(inputs_A)
            features_A = features_A.mean([2, 3]).cpu().numpy()

            prediction = regressor.predict(features_A)
            mos_scores.extend(mos.cpu().numpy())
            predictions.extend(prediction)

    return np.array(mos_scores), np.array(predictions)

# Plot results
def plot_results(mos_scores, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path) / "kadid10k.csv"
    dataset = KADID10KDataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )

    model = SimCLR(encoder_params=DotMap(args.model.encoder), temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min,
    )
    scaler = torch.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, scaler, device)

    regressor = train_ridge_regressor(model, train_dataloader, device)
    save_ridge_regressor(regressor, Path(args.checkpoint_base_path))

    val_mos_scores, val_predictions = evaluate_ridge_regressor(regressor, model, val_dataloader, device)
    test_mos_scores, test_predictions = evaluate_ridge_regressor(regressor, model, test_dataloader, device)

    plot_results(test_mos_scores, test_predictions) """



# Grad-CAM
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
import yaml
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import cv2  # Grad-CAM 시각화에 필요

# Config 로더
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = 0

    # Initialize metrics
    train_metrics = {'srcc': [], 'plcc': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics = {'srcc': [], 'plcc': []}

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        srocc_values, plcc_values = [], []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            # Generate hard negatives
            print(f"[Debug] Batch {i}: Generating hard negatives...")
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)
            print(f"[Debug] Batch {i}: Hard negatives shape: {hard_negatives.shape}")

            hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:])
            #verify_hard_negatives(inputs_B.shape, hard_negatives.shape)

            # Process hard negatives through backbone and projector
            print(f"[Debug] Batch {i}: Processing hard negatives through backbone...")
            backbone_output = model.backbone(hard_negatives)
            print(f"[Debug] Batch {i}: Backbone output shape: {backbone_output.shape}")

            gap_output = backbone_output.mean([2, 3])  # Global Average Pooling
            proj_negatives = model.projector(gap_output)  # [batch_size * num_crops, embedding_dim]
            proj_negatives = F.normalize(proj_negatives, dim=1)
            print(f"[Debug] Batch {i}: Projector output (negatives) shape: {proj_negatives.shape}")

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                # Process inputs_A and inputs_B through model
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)
                print(f"[Debug] Batch {i}: proj_A shape = {proj_A.shape}, proj_B shape = {proj_B.shape}")

                # Calculate SE weights from features_A
                features_A = model.backbone(inputs_A)
                se_weights = features_A.mean(dim=[2, 3])  # Calculate SE weights
                print(f"[Debug] Batch {i}: SE weights shape: {se_weights.shape}")

                # Compute loss with SE weights
                loss = model.compute_loss(proj_A, proj_B, proj_negatives, se_weights)
                print(f"[Debug] Batch {i}: Loss value: {loss.item()}")

            # Debugging: Check for NaN/Inf values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Debug] Batch {i}: Loss is NaN or Inf. Skipping this batch.")
                continue

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            # SRCC and PLCC for the batch
            srocc, _ = stats.spearmanr(proj_A.detach().cpu().flatten(), proj_B.detach().cpu().flatten())
            plcc, _ = stats.pearsonr(proj_A.detach().cpu().flatten(), proj_B.detach().cpu().flatten())
            srocc_values.append(srocc)
            plcc_values.append(plcc)

            progress_bar.set_postfix(loss=running_loss / (i + 1))

        # Average SRCC and PLCC for the epoch
        train_metrics['srcc'].append(np.mean(srocc_values))
        train_metrics['plcc'].append(np.mean(plcc_values))

        lr_scheduler.step()

        # Validation metrics
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        val_metrics['srcc'].append(avg_srocc_val)
        val_metrics['plcc'].append(avg_plcc_val)

        # Test metrics
        avg_srocc_test, avg_plcc_test = validate(args, model, test_dataloader, device)
        test_metrics['srcc'].append(avg_srocc_test)
        test_metrics['plcc'].append(avg_plcc_test)

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Finished training")
    return train_metrics, val_metrics, test_metrics


def validate(args, model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Flatten crops if needed
            if inputs_A.dim() == 5:
                inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
                inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            # Normalize projections for SRCC/PLCC calculation
            proj_A = F.normalize(proj_A, dim=1).detach().cpu().numpy()
            proj_B = F.normalize(proj_B, dim=1).detach().cpu().numpy()

            # Calculate SRCC and PLCC
            srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
            plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    avg_srocc = np.mean(srocc_values)
    avg_plcc = np.mean(plcc_values)

    return avg_srocc, avg_plcc


## Grad-CAM 구현
# class GradCam:
#    def __init__(self, model, target_layer):
#        self.model = model
#        self.target_layer = target_layer
#        self.gradients = None
#
#        # Hook 설정
#        self.hook_layers()
#
#    def hook_layers(self):
#        def forward_hook(module, input, output):
#            self.feature_maps = output
#
#        def backward_hook(module, grad_in, grad_out):
#            self.gradients = grad_out[0]
#
#        self.target_layer.register_forward_hook(forward_hook)
#        self.target_layer.register_backward_hook(backward_hook)
#
#    def generate_heatmap(self, input_tensor):
#        # Forward 패스
#        output = self.model(input_tensor)
#
#        # Grad-CAM은 임베딩 벡터의 특정 차원에 대해 수행 (예: 평균 사용)
#        class_score = output.mean(dim=1)  # 임베딩 평균 점수 사용
#        class_score.backward(torch.ones_like(class_score))  # 역전파를 위한 더미 그래디언트
#
#        # Grad-CAM 생성
#        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
#        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0).cpu().detach().numpy()
#        cam = np.maximum(cam, 0)  # ReLU
#        cam = cam / cam.max()  # 정규화
#
#        return cam
#
#    def overlay_heatmap(self, cam, image):
#        # Grad-CAM을 OpenCV에서 사용할 수 있는 형식으로 변환
#        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
#        cam_resized = np.clip(cam_resized, 0, 1)  # [0, 1] 범위로 클리핑
#        cam_resized = np.uint8(255 * cam_resized)  # [0, 255] 범위로 변환 및 uint8 타입 변환
#
#        # OpenCV의 ColorMap 적용
#        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
#
#        # 원본 이미지가 3채널인지 확인 및 변환
#        if image.ndim == 2:  # 흑백 이미지인 경우
#            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
#        # Grad-CAM 히트맵과 원본 이미지 오버레이
#        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
#        return overlay
#
#
#

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook 설정
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor):
        # Forward 패스
        output = self.model(input_tensor)

        # Grad-CAM은 임베딩 벡터의 특정 차원에 대해 수행 (예: 평균 사용)
        class_score = output.mean(dim=1)  # 임베딩 평균 점수 사용
        class_score.backward(torch.ones_like(class_score))  # 역전파를 위한 더미 그래디언트

        # Grad-CAM 생성
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0).cpu().detach().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()  # 정규화

        return cam

    def overlay_heatmap(self, cam, image):
        # Grad-CAM을 OpenCV에서 사용할 수 있는 형식으로 변환
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))  # [H, W]로 크기 조정
        cam_resized = np.uint8(255 * cam_resized / cam_resized.max())  # [0, 255] 범위로 변환 및 uint8 타입 변환

        # OpenCV의 ColorMap 적용
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)  # 컬러맵 생성

        # 원본 이미지 정규화
        image_normalized = (image - image.min()) / (image.max() - image.min())  # [0, 1]로 정규화
        image_normalized = np.uint8(255 * image_normalized)  # [0, 255]로 변환

        # 원본 이미지와 Grad-CAM 히트맵 오버레이
        overlay = cv2.addWeighted(image_normalized, 0.6, heatmap, 0.4, 0)
        return overlay


# Grad-CAM 시각화
def visualize_gradcam(model, dataloader, device, attention_layer):
    gradcam = GradCam(model, attention_layer)  # Attention 계층 설정
    for batch in dataloader:
        inputs_A = batch["img_A"].to(device)

        # 입력 데이터 차원 확인 및 변환
        if inputs_A.dim() == 5:  # [batch_size, num_crops, C, H, W]
            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, C, H, W]

        # Grad-CAM 생성
        cam = gradcam.generate_heatmap(inputs_A)

        # 첫 번째 이미지를 시각화
        input_image = inputs_A[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # 정규화
        input_image = np.uint8(255 * input_image)  # [0, 255] 범위로 변환 및 uint8 타입 변환

        overlay = gradcam.overlay_heatmap(cam, input_image)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))  # BGR -> RGB 변환
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Grad-CAM Heatmap")
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))  # BGR -> RGB 변환
        plt.axis("off")

        plt.show()
        break



if __name__ == "__main__":
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path) / "kadid10k.csv"
    dataset = KADID10KDataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=min(args.training.num_workers, 16),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=min(args.training.num_workers, 16),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=min(args.training.num_workers, 16),
    )

    model = SimCLR(encoder_params=DotMap(args.model.encoder), temperature=args.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.learning_rate,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.training.lr_scheduler.T_0,
        T_mult=args.training.lr_scheduler.T_mult,
        eta_min=args.training.lr_scheduler.eta_min,
    )
    scaler = torch.amp.GradScaler()

    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        scaler,
        device
    )

    attention_layer = model.attention  # DistortionAttention 계층 지정
    print("\nVisualizing Grad-CAM...")
    visualize_gradcam(model, test_dataloader, device, attention_layer)
"""
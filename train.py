# KADID

""" import torch
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
            verify_hard_negatives(inputs_B.shape, hard_negatives.shape)

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



def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

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


def evaluate_hard_negative_attention(model, dataloader, device):
    model.eval()
    metrics = {'with_attention': {'srcc': [], 'plcc': []}, 'without_attention': {'srcc': [], 'plcc': []}}

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A_with, proj_B_with = model(inputs_A, inputs_B)
            model.attention = None
            proj_A_without, proj_B_without = model(inputs_A, inputs_B)

            srcc_with, _ = stats.spearmanr(proj_A_with.flatten().cpu().numpy(), proj_B_with.flatten().cpu().numpy())
            plcc_with, _ = stats.pearsonr(proj_A_with.flatten().cpu().numpy(), proj_B_with.flatten().cpu().numpy())

            srcc_without, _ = stats.spearmanr(proj_A_without.flatten().cpu().numpy(), proj_B_without.flatten().cpu().numpy())
            plcc_without, _ = stats.pearsonr(proj_A_without.flatten().cpu().numpy(), proj_B_without.flatten().cpu().numpy())

            metrics['with_attention']['srcc'].append(srcc_with)
            metrics['with_attention']['plcc'].append(plcc_with)
            metrics['without_attention']['srcc'].append(srcc_without)
            metrics['without_attention']['plcc'].append(plcc_without)

    print("\nHardNegativeCrossAttention Results:")
    print(f"With Attention: SRCC = {np.mean(metrics['with_attention']['srcc']):.4f}, PLCC = {np.mean(metrics['with_attention']['plcc']):.4f}")
    print(f"Without Attention: SRCC = {np.mean(metrics['without_attention']['srcc']):.4f}, PLCC = {np.mean(metrics['without_attention']['plcc']):.4f}")
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

    # 모델, 옵티마이저, 스케줄러 초기화
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

    # train 함수 호출
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

    # Ridge Regressor 학습
    regressor = train_ridge_regressor(model, train_dataloader, device)

    # Validation 데이터에서 Ridge Regressor 평가
    val_mos_scores, val_predictions = evaluate_ridge_regressor(regressor, model, val_dataloader, device)
    val_srcc, _ = stats.spearmanr(val_mos_scores, val_predictions)
    val_plcc, _ = stats.pearsonr(val_mos_scores, val_predictions)

    # Test 데이터에서 Ridge Regressor 평가
    test_mos_scores, test_predictions = evaluate_ridge_regressor(regressor, model, test_dataloader, device)
    test_srcc, _ = stats.spearmanr(test_mos_scores, test_predictions)
    test_plcc, _ = stats.pearsonr(test_mos_scores, test_predictions)

    # Unseen Distortion Generalization 평가
    unseen_dataset = KADID10KDataset(str(args.unseen_distortion_path))
    unseen_metrics = evaluate_zero_shot(model, unseen_dataset, device)
    avg_srcc_unseen = np.mean(unseen_metrics['srcc'])
    avg_plcc_unseen = np.mean(unseen_metrics['plcc'])

    print(f"\nUnseen Distortion Generalization Results -> SRCC: {avg_srcc_unseen:.4f}, PLCC: {avg_plcc_unseen:.4f}")

    # HardNegativeCrossAttention 확인
    print("\nVerifying HardNegativeCrossAttention...")
    hard_negative_results = test_hard_negative_attention(args, model, test_dataloader, device)
    print(f"Hard Negative Attention Results: SRCC: {np.mean(hard_negative_results['srcc']):.4f}, "
          f"PLCC: {np.mean(hard_negative_results['plcc']):.4f}")

    # 최종 결과 출력
    print(f"\nFinal Validation Results: SRCC = {val_srcc:.4f}, PLCC = {val_plcc:.4f}")
    print(f"Final Test Results: SRCC = {test_srcc:.4f}, PLCC = {test_plcc:.4f}")

    # 그래프 출력 (Test 결과)
    plot_results(test_mos_scores, test_predictions)

    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))
    print("Test Metrics:", format_metrics(test_metrics))


    # KADID10K 데이터셋에서 테스트 수행
    test_results = test(args, model, test_dataloader, device)

    # 테스트 결과 출력
    print("Train(KADID10K)")
    print(f"\nTest Results on KADID10K Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")
 """

#Train(KADID10K)
#Test Results on KADID10K Dataset: SRCC = 0.9172, PLCC = 0.9213


# TID2013
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
            verify_hard_negatives(inputs_B.shape, hard_negatives.shape)

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



def test(args, model, test_dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

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


def evaluate_hard_negative_attention(model, dataloader, device):
    model.eval()
    metrics = {'with_attention': {'srcc': [], 'plcc': []}, 'without_attention': {'srcc': [], 'plcc': []}}

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A_with, proj_B_with = model(inputs_A, inputs_B)
            model.attention = None
            proj_A_without, proj_B_without = model(inputs_A, inputs_B)

            srcc_with, _ = stats.spearmanr(proj_A_with.flatten().cpu().numpy(), proj_B_with.flatten().cpu().numpy())
            plcc_with, _ = stats.pearsonr(proj_A_with.flatten().cpu().numpy(), proj_B_with.flatten().cpu().numpy())

            srcc_without, _ = stats.spearmanr(proj_A_without.flatten().cpu().numpy(), proj_B_without.flatten().cpu().numpy())
            plcc_without, _ = stats.pearsonr(proj_A_without.flatten().cpu().numpy(), proj_B_without.flatten().cpu().numpy())

            metrics['with_attention']['srcc'].append(srcc_with)
            metrics['with_attention']['plcc'].append(plcc_with)
            metrics['without_attention']['srcc'].append(srcc_without)
            metrics['without_attention']['plcc'].append(plcc_without)

    print("\nHardNegativeCrossAttention Results:")
    print(f"With Attention: SRCC = {np.mean(metrics['with_attention']['srcc']):.4f}, PLCC = {np.mean(metrics['with_attention']['plcc']):.4f}")
    print(f"Without Attention: SRCC = {np.mean(metrics['without_attention']['srcc']):.4f}, PLCC = {np.mean(metrics['without_attention']['plcc']):.4f}")
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

    # 모델, 옵티마이저, 스케줄러 초기화
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

    # train 함수 호출
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

    # Ridge Regressor 학습
    regressor = train_ridge_regressor(model, train_dataloader, device)

    # Validation 데이터에서 Ridge Regressor 평가
    val_mos_scores, val_predictions = evaluate_ridge_regressor(regressor, model, val_dataloader, device)
    val_srcc, _ = stats.spearmanr(val_mos_scores, val_predictions)
    val_plcc, _ = stats.pearsonr(val_mos_scores, val_predictions)

    # Test 데이터에서 Ridge Regressor 평가
    test_mos_scores, test_predictions = evaluate_ridge_regressor(regressor, model, test_dataloader, device)
    test_srcc, _ = stats.spearmanr(test_mos_scores, test_predictions)
    test_plcc, _ = stats.pearsonr(test_mos_scores, test_predictions)

    # Unseen Distortion Generalization 평가
    unseen_dataset = TID2013Dataset(str(args.unseen_distortion_path))
    unseen_metrics = evaluate_zero_shot(model, unseen_dataset, device)
    avg_srcc_unseen = np.mean(unseen_metrics['srcc'])
    avg_plcc_unseen = np.mean(unseen_metrics['plcc'])

    print(f"\nUnseen Distortion Generalization Results -> SRCC: {avg_srcc_unseen:.4f}, PLCC: {avg_plcc_unseen:.4f}")

    # HardNegativeCrossAttention 확인
    print("\nVerifying HardNegativeCrossAttention...")
    hard_negative_results = test_hard_negative_attention(args, model, test_dataloader, device)
    print(f"Hard Negative Attention Results: SRCC: {np.mean(hard_negative_results['srcc']):.4f}, "
          f"PLCC: {np.mean(hard_negative_results['plcc']):.4f}")

    # 최종 결과 출력
    print(f"\nFinal Validation Results: SRCC = {val_srcc:.4f}, PLCC = {val_plcc:.4f}")
    print(f"Final Test Results: SRCC = {test_srcc:.4f}, PLCC = {test_plcc:.4f}")

    # 그래프 출력 (Test 결과)
    plot_results(test_mos_scores, test_predictions)

    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))
    print("Test Metrics:", format_metrics(test_metrics))


    # TID2013 데이터셋에서 테스트 수행
    test_results = test(args, model, test_dataloader, device)

    # 테스트 결과 출력
    print("Train(KADID10K)")
    print(f"\nTest Results on KADID10K Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

 """

 

# Training Metrics: {'srcc': [0.9292, 0.9295, 0.93, 0.9291, 0.9296, 0.9294, 0.9291, 0.9277, 0.9294, 0.9297], 'plcc': [0.9332, 0.9334, 0.9339, 0.9331, 0.9336, 0.9332, 0.9331, 0.9317, 0.9334, 0.9337]}
# Validation Metrics: {'srcc': [0.9117, 0.911, 0.911, 0.9125, 0.9151, 0.9025, 0.9104, 0.914, 0.9118, 0.9108], 'plcc': [0.9171, 0.9156, 0.9165, 0.9188, 0.9205, 0.9081, 0.9162, 0.919, 0.9168, 0.9165]}
# Test Metrics: {'srcc': [0.9242, 0.924, 0.9254, 0.9223, 0.9253, 0.9196, 0.9223, 0.9253, 0.925, 0.9201], 'plcc': [0.9294, 0.9284, 0.9304, 0.9275, 0.9301, 0.9245, 0.9275, 0.9299, 0.9299, 0.925]}


#-------------------



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

    # KADID10K Dataset (Training and Validation)
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")
    kadid_dataset = KADID10KDataset(kadid_dataset_path)

    train_size = int(0.7 * len(kadid_dataset))
    val_size = int(0.1 * len(kadid_dataset))
    test_size = len(kadid_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(kadid_dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

    # TID2013 Dataset (Test)
    tid_dataset_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/TID2013/mos.csv"
    tid_dataset = TID2013Dataset(Path(tid_dataset_path))
    test_dataloader = DataLoader(tid_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

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

    # Training on KADID10K
    train_metrics, val_metrics = train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    # Testing on TID2013
    test_results = test(args, model, test_dataloader, device)
    print("Train(KADID10K) & Test(TID2013)")
    print(f"\nTest Results on TID2013 Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))

 """
# Train(KADID10K) & Test(TID2013)
# Test Results on TID2013 Dataset: SRCC = 0.9280, PLCC = 0.9319
# 
# Training Metrics: {'srcc': [0.9293, 0.9302, 0.93, 0.9286, 0.9305, 0.9293, 0.9297, 0.9298, 0.9292, 0.9304], 'plcc': [0.9327, 0.9336, 0.9332, 0.9319, 0.9337, 0.9324, 0.9331, 0.9331, 0.9325, 0.9336]}
# Validation Metrics: {'srcc': [0.9237, 0.925, 0.9212, 0.9213, 0.9212, 0.9245, 0.9202, 0.9158, 0.9196, 0.9199], 'plcc': [0.9271, 0.9282, 0.9241, 0.9253, 0.9245, 0.9277, 0.9226, 0.9196, 0.923, 0.9234]}




# Train(TID2013) & Test(KADID)
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

    # KADID10K Dataset
    kadid_dataset_path = Path("E:/ARNIQA - SE - mix/ARNIQA/dataset/KADID10K/kadid10k.csv")
    kadid_dataset = KADID10KDataset(kadid_dataset_path)
    test_dataloader = DataLoader(kadid_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

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
    print("Train(TID2013) & Test(KADID)")
    print(f"\nTest Results on KADID10K Dataset: SRCC = {test_results['srcc']:.4f}, PLCC = {test_results['plcc']:.4f}")

    def format_metrics(metrics):
        return {key: [round(value, 4) for value in values] for key, values in metrics.items()}

    print("\nTraining Metrics:", format_metrics(train_metrics))
    print("Validation Metrics:", format_metrics(val_metrics))
 """

# Train(TID2013) & Test(KADID)
# Test Results on KADID10K Dataset: SRCC = 0.8999, PLCC = 0.9084
# 
# Training Metrics: {'srcc': [0.909, 0.9123, 0.9099, 0.9123, 0.9092, 0.9103, 0.9089, 0.9112, 0.9087, 0.9087], 'plcc': [0.9159, 0.9187, 0.9167, 0.9187, 0.9162, 0.9168, 0.9158, 0.9178, 0.9156, 0.9155]}
# Validation Metrics: {'srcc': [0.913, 0.9044, 0.8976, 0.9157, 0.9067, 0.9056, 0.905, 0.9113, 0.9106, 0.9028], 'plcc': [0.9183, 0.9107, 0.9053, 0.9208, 0.9129, 0.9121, 0.9112, 0.9177, 0.9172, 0.9095]}




# Regressor 추출 코드
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

    plot_results(test_mos_scores, test_predictions)

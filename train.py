# KADID
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

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


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


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics = {'srcc': [], 'plcc': []}

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

        # Test
        test_metric = test(args, model, test_dataloader, device)
        test_metrics['srcc'].append(test_metric['srcc'])
        test_metrics['plcc'].append(test_metric['plcc'])

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics, test_metrics


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

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path)
    dataset = KADID10KDataset(str(dataset_path))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

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
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)

 """

# KONIQ
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KONIQ10KDataset
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

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


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


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics = {'srcc': [], 'plcc': []}

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

        # Test
        test_metric = test(args, model, test_dataloader, device)
        test_metrics['srcc'].append(test_metric['srcc'])
        test_metrics['plcc'].append(test_metric['plcc'])

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics, test_metrics


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

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_base_path)
    dataset = KONIQ10KDataset(str(dataset_path))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

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
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)



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


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics = {'srcc': [], 'plcc': []}

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

        # Test
        test_metric = test(args, model, test_dataloader, device)
        test_metrics['srcc'].append(test_metric['srcc'])
        test_metrics['plcc'].append(test_metric['plcc'])

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics, test_metrics


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
    dataset = TID2013Dataset(str(dataset_path))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

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
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)

 """




# SPAQ
""" import torch
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


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics = {'srcc': [], 'plcc': []}

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

        # Test
        test_metric = test(args, model, test_dataloader, device)
        test_metrics['srcc'].append(test_metric['srcc'])
        test_metrics['plcc'].append(test_metric['plcc'])

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics, test_metrics


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
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

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
""" import torch
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


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics = {'srcc': [], 'plcc': []}

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

        # Test
        test_metric = test(args, model, test_dataloader, device)
        test_metrics['srcc'].append(test_metric['srcc'])
        test_metrics['plcc'].append(test_metric['plcc'])

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics, test_metrics


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
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

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


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics = {'srcc': [], 'plcc': []}

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

        # Test
        test_metric = test(args, model, test_dataloader, device)
        test_metrics['srcc'].append(test_metric['srcc'])
        test_metrics['plcc'].append(test_metric['plcc'])

        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")
    return train_metrics, val_metrics, test_metrics


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
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device
    )

    # Results
    print("\nTraining Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)

 """

#LIVE
# Training Metrics: {'loss': [1.3416166375665104, 1.385161556449591, 1.3430749528548296, 1.344389720290315, 1.3500481271276288, 1.311995915338105, 1.362798905840107, 1.3029328467799168, 1.3513919080004972, 1.3905024236323786]}
# Validation Metrics: {'srcc': [0.8728831272297886, 0.8926784017534533, 0.8463808868438731, 0.8430774209611263, 0.8814598820151454, 0.8783217775462138, 0.8604482425330922, 0.8722422826434939, 0.8823484322066615, 0.8497454245955931], 'plcc': [0.8802752573302571, 0.8982301779839197, 0.8545239825590596, 0.8501004006982258, 0.8879407990530739, 0.8844903435674625, 0.868455748679986, 0.8784393091916264, 0.8898163848477165, 0.8597359100276125]}
# Test Metrics: {'srcc': 0.844997775637989, 'plcc': 0.8515623023230324}


# ------------------------------------------- cross dataset --------------------------

# Train(KADID) & Test(TID2013)
""" import io
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

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


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


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics_per_epoch = []  # 에포크별 테스트 결과 저장

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

            if hard_negatives.dim() == 5:
                hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:]).to(device)
            elif hard_negatives.dim() != 4:
                raise ValueError(f"[Error] Unexpected hard_negatives dimensions: {hard_negatives.shape}")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_negatives = model.backbone(hard_negatives)
                if features_negatives.dim() == 4:
                    features_negatives = features_negatives.mean([2, 3])  # Apply GAP
                elif features_negatives.dim() == 2:
                    pass  # Already reduced dimensions
                else:
                    raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

                proj_negatives = F.normalize(model.projector(features_negatives), dim=1)
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

        # Test at each epoch
        test_metrics = test(args, model, test_dataloader, device)
        test_metrics_per_epoch.append(test_metrics)  # 각 에포크 결과 저장

        # 평균 SRCC, PLCC 출력
        avg_srcc = np.mean(test_metrics['srcc'])
        avg_plcc = np.mean(test_metrics['plcc'])
        print(f"Epoch {epoch + 1}: Test SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")
        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")

    # 모든 에포크의 테스트 결과 출력
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics_per_epoch, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")

    return train_metrics, val_metrics, test_metrics_per_epoch

def test(args, model, test_dataloader, device):
    if test_dataloader is None:
        raise ValueError("Test DataLoader가 None입니다. 초기화 문제를 확인하세요.")

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

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # KADID10K 경로 설정 및 로드
    kadid_dataset_path = Path(str(args.data_base_path_kadid))
    print(f"[Debug] KADID10K Dataset Path: {kadid_dataset_path}")
    kadid_dataset = KADID10KDataset(str(kadid_dataset_path))

    # TID2013 경로 설정 및 로드
    tid_dataset_path = Path(str(args.data_base_path_tid))
    print(f"[Debug] TID2013 Dataset Path: {tid_dataset_path}")
    tid_dataset = TID2013Dataset(str(tid_dataset_path))

    # 훈련 데이터 분할
    train_size = int(0.8 * len(kadid_dataset))
    val_size = len(kadid_dataset) - train_size
    train_dataset, val_dataset = random_split(kadid_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # 테스트 데이터 로드
    test_dataloader = DataLoader(
        tid_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=4
    )

    # 모델 초기화
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

    # CSIQ 훈련
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    # 최종 결과 출력
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")

 """

# Train(KADID10K) & Test(TID2013)
# Test Results on TID2013 Dataset: SRCC = 0.9446, PLCC = 0.9501
# 
# Training Metrics: {'loss': [1.0261, 1.0186, 1.0079, 1.0192, 1.0054, 1.0113, 1.0078, 1.0082, 1.0043, 1.0074]}
# Validation Metrics: {'srcc': [0.9412, 0.9417, 0.9387, 0.9419, 0.9397, 0.9408, 0.9377, 0.9397, 0.9383, 0.9424], 'plcc': [0.9462, 0.9455, 0.9431, 0.9454, 0.944, 0.9449, 0.9417, 0.9446, 0.9429, 0.9462]}        
""" 
import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset, KONIQ10KDataset
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

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


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


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler, device):
    checkpoint_path = Path(str(args.checkpoint_base_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_srocc = -1

    train_metrics = {'loss': []}
    val_metrics = {'srcc': [], 'plcc': []}
    test_metrics_per_epoch = []  # 에포크별 테스트 결과 저장

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

            if hard_negatives.dim() == 5:
                hard_negatives = hard_negatives.view(-1, *hard_negatives.shape[2:]).to(device)
            elif hard_negatives.dim() != 4:
                raise ValueError(f"[Error] Unexpected hard_negatives dimensions: {hard_negatives.shape}")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                proj_A = F.normalize(proj_A, dim=1)
                proj_B = F.normalize(proj_B, dim=1)

                features_negatives = model.backbone(hard_negatives)
                if features_negatives.dim() == 4:
                    features_negatives = features_negatives.mean([2, 3])  # Apply GAP
                elif features_negatives.dim() == 2:
                    pass  # Already reduced dimensions
                else:
                    raise ValueError(f"[Error] Unexpected features_negatives dimensions: {features_negatives.shape}")

                proj_negatives = F.normalize(model.projector(features_negatives), dim=1)
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

        # Test at each epoch
        test_metrics = test(args, model, test_dataloader, device)
        test_metrics_per_epoch.append(test_metrics)  # 각 에포크 결과 저장

        # 평균 SRCC, PLCC 출력
        avg_srcc = np.mean(test_metrics['srcc'])
        avg_plcc = np.mean(test_metrics['plcc'])
        print(f"Epoch {epoch + 1}: Test SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")
        lr_scheduler.step()

        if val_srocc > best_srocc:
            best_srocc = val_srocc
            save_checkpoint(model, checkpoint_path, epoch, val_srocc)

    print("Training completed.")

    # 모든 에포크의 테스트 결과 출력
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics_per_epoch, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")

    return train_metrics, val_metrics, test_metrics_per_epoch

def test(args, model, test_dataloader, device):
    if test_dataloader is None:
        raise ValueError("Test DataLoader가 None입니다. 초기화 문제를 확인하세요.")

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

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # KADID10K 경로 설정 및 로드
    kadid_dataset_path = Path(str(args.data_base_path_kadid))
    print(f"[Debug] KADID10K Dataset Path: {kadid_dataset_path}")
    kadid_dataset = KADID10KDataset(str(kadid_dataset_path))

    # KONIQ10KDataset 경로 설정 및 로드
    koniq_dataset_path = Path(str(args.data_base_path_koniq))
    print(f"[Debug] TID2013 Dataset Path: {koniq_dataset_path}")
    koniq_dataset = KONIQ10KDataset(str(koniq_dataset_path))

    # 훈련 데이터 분할
    train_size = int(0.8 * len(kadid_dataset))
    val_size = len(kadid_dataset) - train_size
    train_dataset, val_dataset = random_split(kadid_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # 테스트 데이터 로드
    test_dataloader = DataLoader(
        koniq_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=4
    )

    # 모델 초기화
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

    # CSIQ 훈련
    train_metrics, val_metrics, test_metrics = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    # 최종 결과 출력
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")
 """
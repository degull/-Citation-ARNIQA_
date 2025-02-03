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
""" import torch
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

 """



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


# 오버레이 시각화
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
from PIL import ImageEnhance, ImageFilter, Image
from torchvision import transforms
from models.attention_se import DistortionAttention, HardNegativeCrossAttention
from models.resnet_se import SEBlock, ResNetSEVisualizer, ResNetSE
from utils.utils_visualization import visualize_feature_maps


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
    from torchvision.models import resnet50

    # 원본 이미지 로드
    input_image_path = "E:/ARNIQA - SE - mix/ARNIQA/dataset/SPAQ/TestImage/11104.jpg"
    input_image = Image.open(input_image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(input_image).unsqueeze(0)  # (1, 3, 224, 224)

    # ResNetSE 모델 로드
    base_model = resnet50(pretrained=True)
    distortion_attentions = [DistortionAttention(256), DistortionAttention(512),
                             DistortionAttention(1024), DistortionAttention(2048)]
    se_blocks = [SEBlock(256), SEBlock(512), SEBlock(1024), SEBlock(2048)]
    hard_negative_attention = HardNegativeCrossAttention(2048)

    model = ResNetSEVisualizer(base_model, distortion_attentions, hard_negative_attention, se_blocks)
    model.eval()

    with torch.no_grad():
        activation_maps = model(input_tensor, input_image)

    # 시각화 실행
    visualize_feature_maps(activation_maps, input_image)

 """
# ------------------------------------------- cross dataset --------------------------


# KADID & CSIQ
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

    # CSIQ 경로 설정 및 로드
    csiq_dataset_path = Path(str(args.data_base_path_csiq))
    print(f"[Debug] CSIQ Dataset Path: {csiq_dataset_path}")
    csiq_dataset = CSIQDataset(str(csiq_dataset_path))

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
        csiq_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=4
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
    print("KADID & CSIQ")
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")

 """


# TID & CSIQ
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

    # TID2013Dataset 경로 설정 및 로드
    tid_dataset_path = Path(str(args.data_base_path_tid))
    print(f"[Debug] TID Dataset Path: {tid_dataset_path}")
    tid_dataset = TID2013Dataset(str(tid_dataset_path))

    # CSIQDataset 경로 설정 및 로드
    csiq_dataset_path = Path(str(args.data_base_path_csiq))
    print(f"[Debug] CSIQ Dataset Path: {csiq_dataset_path}")
    csiq_dataset = CSIQDataset(str(csiq_dataset_path))

    # 훈련 데이터 분할
    train_size = int(0.8 * len(tid_dataset))
    val_size = len(tid_dataset) - train_size
    train_dataset, val_dataset = random_split(tid_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # 테스트 데이터 로드
    test_dataloader = DataLoader(
        csiq_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=4
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

    # csiq 훈련
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
    print("TID & CSIQ")
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")
 """

# TID & KONIQ
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
from data import TID2013Dataset, KONIQ10KDataset
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

    # TID2013Dataset 경로 설정 및 로드
    tid_dataset_path = Path(str(args.data_base_path_tid))
    print(f"[Debug] TID Dataset Path: {tid_dataset_path}")
    tid_dataset = TID2013Dataset(str(tid_dataset_path))

    # KONIQ10KDataset 경로 설정 및 로드
    koniq_dataset_path = Path(str(args.data_base_path_koniq))
    print(f"[Debug] KONIQ Dataset Path: {koniq_dataset_path}")
    koniq_dataset = KONIQ10KDataset(str(koniq_dataset_path))

    # 훈련 데이터 분할
    train_size = int(0.8 * len(tid_dataset))
    val_size = len(tid_dataset) - train_size
    train_dataset, val_dataset = random_split(tid_dataset, [train_size, val_size])

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

    # csiq 훈련
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
    print("TID & KONIQ")
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")
"""


# KONIQ & KADID
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
from data import KONIQ10KDataset, KADID10KDataset
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

    # KONIQ10KDataset 경로 설정 및 로드
    koniq_dataset_path = Path(str(args.data_base_path_koniq))
    print(f"[Debug] TID Dataset Path: {koniq_dataset_path}")
    koniq_dataset = KONIQ10KDataset(str(koniq_dataset_path))

    # KADID10KDataset 경로 설정 및 로드
    kadid_dataset_path = Path(str(args.data_base_path_kadid))
    print(f"[Debug] KONIQ Dataset Path: {kadid_dataset_path}")
    kadid_dataset = KADID10KDataset(str(kadid_dataset_path))

    # 훈련 데이터 분할
    train_size = int(0.8 * len(koniq_dataset))
    val_size = len(koniq_dataset) - train_size
    train_dataset, val_dataset = random_split(koniq_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # 테스트 데이터 로드
    test_dataloader = DataLoader(
        kadid_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=4
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

    # kadid 훈련
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
    print("KONIQ & KADID")
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")

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
from data import KONIQ10KDataset, KADID10KDataset
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

    # KONIQ10KDataset 경로 설정 및 로드
    koniq_dataset_path = Path(str(args.data_base_path_koniq))
    print(f"[Debug] KONIQ Dataset Path: {koniq_dataset_path}")
    koniq_dataset = KONIQ10KDataset(str(koniq_dataset_path))

    # KADID10KDataset 경로 설정 및 로드
    kadid_dataset_path = Path(str(args.data_base_path_kadid))
    print(f"[Debug] KADID Dataset Path: {kadid_dataset_path}")
    kadid_dataset = KADID10KDataset(str(kadid_dataset_path))

    # 훈련 데이터 분할
    train_size = int(0.8 * len(koniq_dataset))
    val_size = len(koniq_dataset) - train_size
    train_dataset, val_dataset = random_split(koniq_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4
    )

    # 테스트 데이터 로드
    test_dataloader = DataLoader(
        kadid_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=4
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

    # tid 훈련
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
    print("KONIQ & KADID")
    print("\nFinal Test Metrics Per Epoch:")
    for i, metrics in enumerate(test_metrics, 1):
        avg_srcc = np.mean(metrics['srcc'])
        avg_plcc = np.mean(metrics['plcc'])
        print(f"Epoch {i}: SRCC = {avg_srcc:.4f}, PLCC = {avg_plcc:.4f}")
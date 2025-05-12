import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from pathlib import Path
import argparse
import numpy as np
import os
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from fine_tune.data_loader import SegmentPairDataset
from fine_tune.model_ext import RelativeComplexityModel

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Compute evaluation metrics for relative complexity prediction.
    
    Args:
        y_true: Ground truth delta values
        y_pred: Predicted delta values
        
    Returns:
        dict: Dictionary of metrics
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'delta_mae': mean_absolute_error(y_true, y_pred - y_true)
    }

def train_epoch(model: nn.Module, 
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               device: str) -> Tuple[float, dict]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        
    Returns:
        tuple: (avg_loss, metrics_dict)
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for prev_seg, curr_seg, deltas, masks in tqdm(dataloader, desc="Training"):
        prev_seg = prev_seg.to(device)
        curr_seg = curr_seg.to(device)
        deltas = deltas.to(device)
        masks = masks.to(device)
        
        # Forward pass
        pred_deltas = model(prev_seg, curr_seg, masks)
        
        # Compute loss
        loss = criterion(pred_deltas, deltas.unsqueeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(pred_deltas.detach())
        all_labels.append(deltas.unsqueeze(1).detach())
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    
    return total_loss / len(dataloader), metrics

def validate(model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, dict]:
    """Validate model performance.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        tuple: (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for prev_seg, curr_seg, deltas, masks in tqdm(dataloader, desc="Validation"):
            prev_seg = prev_seg.to(device)
            curr_seg = curr_seg.to(device)
            deltas = deltas.to(device)
            masks = masks.to(device)
            
            pred_deltas = model(prev_seg, curr_seg, masks)
            loss = criterion(pred_deltas, deltas.unsqueeze(1))
            
            total_loss += loss.item()
            all_preds.append(pred_deltas)
            all_labels.append(deltas.unsqueeze(1))
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    
    return total_loss / len(dataloader), metrics

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank):
    print("[DEBUG] 开始训练脚本初始化")
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train EnCodec complexity model')
    parser.add_argument('--batch_size', type=int, required=True,
                      help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, required=True,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, required=True,
                      help='Initial learning rate')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save checkpoints')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--context_ratio', type=float, default=1.0,
                      help='Ratio of context segment to include')
    args = parser.parse_args()

    # DDP setup
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    
    # Setup device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    print(f"Using device: {device} (rank {rank}/{world_size})")

    # Create and split dataset
    print("[DEBUG] 开始加载数据集...")
    full_dataset = SegmentPairDataset(device=device, context_ratio=args.context_ratio)
    print(f"[DEBUG] 数据集加载完成，共 {len(full_dataset)} 个样本")
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Create data loaders with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=SegmentPairDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=SegmentPairDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model with DDP
    print("[DEBUG] 初始化模型...")
    model = RelativeComplexityModel().to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    print("[DEBUG] 模型初始化完成")
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.L1Loss()  # Using L1 loss for delta prediction
    
    # Training loop
    best_val_loss = float('inf')
    print(f"[DEBUG] 开始训练循环，共 {args.epochs} 个epoch")
    for epoch in range(args.epochs):
        print(f"[DEBUG] 开始epoch {epoch+1}/{args.epochs}")
        train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # Validate and compute metrics
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val MAE: {val_metrics['mae']:.4f} - "
                  f"Val ΔMAE: {val_metrics['delta_mae']:.4f} - "
                  f"Val R2: {val_metrics['r2']:.4f}")
        
        # Save best model (only on rank 0)
        if val_loss < best_val_loss and rank == 0:
            best_val_loss = val_loss
            save_path = Path(args.save_dir) / "best_model.pth"
            save_path.parent.mkdir(exist_ok=True)
            torch.save(model.module.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    cleanup()

if __name__ == '__main__':
    # Use torch.multiprocessing for DDP
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(), nprocs=world_size)

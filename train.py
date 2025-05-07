import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from pathlib import Path
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from .data_loader import ComplexityDataset
from .model_ext import EncodecComplexityModel

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Compute evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def train_epoch(model: nn.Module, 
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               device: str) -> float:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for waveforms, complexities in tqdm(dataloader, desc="Training"):
        waveforms = waveforms.to(device)
        complexities = complexities.to(device)
        
        # Forward pass
        _, pred_complexities = model(waveforms)
        loss = criterion(pred_complexities, complexities.mean(dim=1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: str) -> float:
    """Validate model performance.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for waveforms, complexities in tqdm(dataloader, desc="Validation"):
            waveforms = waveforms.to(device)
            complexities = complexities.to(device)
            
            _, pred_complexities = model(waveforms)
            loss = criterion(pred_complexities, complexities.mean(dim=1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train EnCodec complexity model')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create and split dataset
    full_dataset = ComplexityDataset(device=device)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ComplexityDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ComplexityDataset.collate_fn
    )

    # Initialize model
    model = EncodecComplexityModel().to(device)
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # Validate and compute metrics
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val MAE: {val_metrics['mae']:.4f} - "
              f"Val R2: {val_metrics['r2']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.save_dir) / "best_model.pth"
            save_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == '__main__':
    main()

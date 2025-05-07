import torch
import torch.nn as nn
from encodec import EncodecModel
from typing import Tuple

class EncodecComplexityModel(nn.Module):
    """EnCodec model extended with complexity prediction head.
    
    Args:
        pretrained: Whether to load pretrained EnCodec weights
        freeze_encoder: Whether to freeze encoder weights during training
        hidden_dim: Dimension of hidden layers in prediction head
    """
    def __init__(self, 
                 pretrained: bool = True,
                 freeze_encoder: bool = True,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Load pretrained EnCodec model
        self.encodec = EncodecModel.encodec_model_24khz(pretrained=pretrained)
        self.encodec.set_target_bandwidth(6.0)  # Use 6kbps model
        
        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.encodec.parameters():
                param.requires_grad = False
        
        # Add complexity prediction head
        encoder_out_dim = 128  # Dimension of encoder output
        self.complexity_head = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predict single complexity score
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with complexity prediction.
        
        Args:
            x: Input audio tensor of shape (batch, 1, samples)
            mask: Attention mask (batch, 1, samples) where 1=valid, 0=padding
            
        Returns:
            tuple: (audio_output, complexity)
                  audio_output: Reconstructed audio
                  complexity: Predicted complexity scores
        """
        # Get encoder output
        encoded_frames = self.encodec.encode(x)
        codes = encoded_frames[0][0]  # Get first (and only) frame's codes
        
        # Get encoder's latent representation (before quantization)
        with torch.no_grad():
            emb = self.encodec.quantizer.decode(codes.transpose(0, 1))  # shape: [batch, channels, time]
        
        # Handle masked averaging
        if mask is not None:
            # Downsample mask to match feature dimension
            mask = torch.nn.functional.interpolate(
                mask.float(), 
                size=emb.size(2), 
                mode='nearest')
            mask = mask.squeeze(1)  # shape: [batch, time]
            
            # Apply mask and compute mean
            masked_emb = emb * mask.unsqueeze(1)  # shape: [batch, channels, time]
            emb = masked_emb.sum(dim=2) / (mask.sum(dim=1, keepdim=True) + 1e-6)
        else:
            # Simple average if no mask provided
            emb = emb.mean(dim=2)
        
        # Predict complexity
        complexity = self.complexity_head(emb)
        
        # Get reconstructed audio
        audio_output = self.encodec.decode(encoded_frames)
        
        return audio_output, complexity.squeeze(-1)
    
    def predict_complexity(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Predict complexity score without decoding audio.
        
        Args:
            x: Input audio tensor of shape (batch, 1, samples)
            mask: Attention mask (batch, 1, samples) where 1=valid, 0=padding
            
        Returns:
            Predicted complexity scores
        """
        encoded_frames = self.encodec.encode(x)
        codes = encoded_frames[0][0]
        emb = self.encodec.quantizer.decode(codes.transpose(0, 1))
        
        if mask is not None:
            # Downsample mask to match feature dimension
            mask = torch.nn.functional.interpolate(
                mask.float(), 
                size=emb.size(2), 
                mode='nearest')
            mask = mask.squeeze(1)
            
            # Apply mask and compute mean
            masked_emb = emb * mask.unsqueeze(1)
            emb = masked_emb.sum(dim=2) / (mask.sum(dim=1, keepdim=True) + 1e-6)
        else:
            emb = emb.mean(dim=2)
        return self.complexity_head(emb).squeeze(-1)

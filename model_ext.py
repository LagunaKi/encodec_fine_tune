import torch
import torch.nn as nn
from encodec import EncodecModel
from typing import Tuple
import torch.nn.functional as F

class RelativeComplexityModel(nn.Module):
    """EnCodec-based model for predicting relative complexity between variation segments.
    
    Args:
        pretrained: Whether to load pretrained EnCodec weights
        freeze_encoder: Whether to freeze encoder weights during training
        hidden_dim: Dimension of hidden layers
        tcn_layers: Number of TCN layers
        num_channels: Number of channels in TCN
    """
    def __init__(self, 
                 pretrained: bool = True,
                 freeze_encoder: bool = True,
                 hidden_dim: int = 256,
                 tcn_layers: int = 4,
                 num_channels: int = 64):
        super().__init__()
        
        # Load pretrained EnCodec model
        self.encodec = EncodecModel.encodec_model_24khz(pretrained=pretrained)
        self.encodec.set_target_bandwidth(6.0)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encodec.parameters():
                param.requires_grad = False
                
        # TCN for temporal feature extraction
        self.tcn = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(128, num_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(num_channels),
                nn.ReLU()
            ) for _ in range(tcn_layers)]
        )
        
        # Cross-attention for variation comparison
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=4,
            batch_first=True
        )
        
        # Complexity prediction head (original version)
        self.complexity_head = nn.Sequential(
            nn.Linear(num_channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal features using EnCodec and TCN."""
        encoded_frames = self.encodec.encode(x) # discrete encode
        codes = encoded_frames[0][0]
        emb = self.encodec.quantizer.decode(codes.transpose(0, 1)) # Dimension reduction, (B, C, T) → (B, T)
        return self.tcn(emb.mean(dim=2).transpose(1, 2)).transpose(1, 2)
        
    def forward(self, 
               prev_segment: torch.Tensor, 
               current_segment: torch.Tensor,
               masks: torch.Tensor = None) -> torch.Tensor:
        """Predict relative complexity between variation segments.
        
        Args:
            prev_segment: Previous variation segment (B, 1, T1)
            current_segment: Current variation segment (B, 1, T2)
            masks: Optional attention masks (B, 2, max(T1,T2))
            
        Returns:
            Predicted complexity delta (B, 1)
        """
        # Extract features
        prev_feat = self.extract_features(prev_segment)  # (B, T1, C)
        curr_feat = self.extract_features(current_segment)  # (B, T2, C)
        
        # Cross-attention comparison
        attn_out, _ = self.cross_attn(
            query=curr_feat,
            key=prev_feat,
            value=prev_feat
        )
        
        # Pool features
        prev_pool = prev_feat.mean(dim=1)  # (B, C)
        curr_pool = curr_feat.mean(dim=1)  # (B, C)
        attn_pool = attn_out.mean(dim=1)  # (B, C)
        
        # Predict delta
        features = torch.cat([curr_pool - prev_pool, attn_pool], dim=1)
        return self.complexity_head(features)

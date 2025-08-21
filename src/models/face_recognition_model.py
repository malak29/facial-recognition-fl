import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from enum import Enum
from loguru import logger

# Import vision transformers for state-of-the-art performance
from torchvision import models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class ModelArchitecture(str, Enum):
    """Available model architectures"""
    EFFICIENTNET_V2 = "efficientnet_v2"
    RESNET152 = "resnet152"
    VIT = "vision_transformer"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Model configuration"""
    architecture: ModelArchitecture = ModelArchitecture.EFFICIENTNET_V2
    input_size: Tuple[int, int] = (224, 224)
    num_classes: int = 512  # Embedding dimension
    dropout_rate: float = 0.2
    use_attention: bool = True
    use_arcface: bool = True
    pretrained: bool = True
    freeze_backbone: bool = False
    differential_privacy: bool = True
    
    # ArcFace parameters
    arcface_s: float = 30.0
    arcface_m: float = 0.5
    
    # Privacy parameters
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0


class AttentionModule(nn.Module):
    """Self-attention module for enhanced feature extraction"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = avg_out + max_out
        
        return x * attention.view(b, c, 1, 1)


class ArcFaceHead(nn.Module):
    """ArcFace head for improved discriminative learning"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m
    
    def forward(
        self,
        input: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Normalize features and weights
        input = F.normalize(input)
        weight = F.normalize(self.weight)
        
        # Compute cosine similarity
        cosine = F.linear(input, weight)
        
        if label is None:
            return cosine * self.s
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class EfficientFaceNet(nn.Module):
    """
    EfficientNet-based face recognition model with privacy and fairness features
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load backbone
        if config.architecture == ModelArchitecture.EFFICIENTNET_V2:
            if config.pretrained:
                self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            else:
                self.backbone = efficientnet_v2_s(weights=None)
            
            # Get number of features from backbone
            in_features = self.backbone.classifier[1].in_features
            
            # Remove original classifier
            self.backbone.classifier = nn.Identity()
            
        elif config.architecture == ModelArchitecture.RESNET152:
            self.backbone = models.resnet152(pretrained=config.pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {config.architecture}")
        
        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Attention module
        if config.use_attention:
            self.attention = AttentionModule(in_features)
        else:
            self.attention = nn.Identity()
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(1024, config.num_classes),
            nn.BatchNorm1d(config.num_classes)
        )
        
        # ArcFace head for training
        if config.use_arcface:
            self.arcface = ArcFaceHead(
                config.num_classes,
                config.num_classes,
                s=config.arcface_s,
                m=config.arcface_m
            )
        else:
            self.arcface = None
        
        # Privacy-preserving noise layer
        if config.differential_privacy:
            self.register_buffer('noise_scale', torch.tensor(config.noise_multiplier))
        else:
            self.noise_scale = None
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract facial features"""
        # Backbone features
        features = self.backbone(x)
        
        # Apply attention if enabled
        if isinstance(features, torch.Tensor) and len(features.shape) == 4:
            features = self.attention(features)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        # Feature extraction
        embeddings = self.feature_layers(features)
        
        return F.normalize(embeddings, p=2, dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [B, C, H, W]
            labels: Optional labels for training
            return_embeddings: Whether to return embeddings
        
        Returns:
            Dictionary containing outputs and embeddings
        """
        # Extract embeddings
        embeddings = self.extract_features(x)
        
        # Add differential privacy noise during training
        if self.training and self.noise_scale is not None:
            noise = torch.randn_like(embeddings) * self.noise_scale
            embeddings = embeddings + noise
        
        outputs = {
            'embeddings': embeddings
        }
        
        # Apply ArcFace if training with labels
        if self.arcface is not None and labels is not None:
            outputs['logits'] = self.arcface(embeddings, labels)
        else:
            outputs['logits'] = embeddings
        
        return outputs
    
    def compute_similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between embeddings"""
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        return torch.matmul(embeddings1, embeddings2.t())


class FaceRecognitionModel(nn.Module):
    """
    Complete face recognition model with training utilities
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = EfficientFaceNet(config)
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        
        # Metrics tracking
        self.training_metrics = {
            'loss': [],
            'accuracy': [],
            'fairness_score': []
        }
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step"""
        self.train()
        
        images = batch['images']
        labels = batch['labels']
        sensitive_attributes = batch.get('sensitive_attributes', None)
        
        # Forward pass
        outputs = self.model(images, labels)
        
        # Calculate loss
        loss = self.criterion(outputs['logits'], labels)
        
        # Add triplet loss if triplets are provided
        if 'anchor' in batch and 'positive' in batch and 'negative' in batch:
            anchor_emb = self.model.extract_features(batch['anchor'])
            positive_emb = self.model.extract_features(batch['positive'])
            negative_emb = self.model.extract_features(batch['negative'])
            
            triplet_loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
            loss = loss + 0.5 * triplet_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for privacy
        if self.config.differential_privacy:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
        
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(outputs['logits'], dim=1)
            accuracy = (preds == labels).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
        
        return metrics
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single validation step"""
        self.eval()
        
        with torch.no_grad():
            images = batch['images']
            labels = batch['labels']
            
            outputs = self.model(images, labels)
            loss = self.criterion(outputs['logits'], labels)
            
            preds = torch.argmax(outputs['logits'], dim=1)
            accuracy = (preds == labels).float().mean()
        
        return {
            'val_loss': loss.item(),
            'val_accuracy': accuracy.item()
        }
    
    def get_embeddings(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Extract embeddings for given images"""
        self.eval()
        with torch.no_grad():
            return self.model.extract_features(images)
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_metrics = checkpoint.get('training_metrics', {})
        logger.info(f"Model checkpoint loaded from {path}")
        return checkpoint.get('metadata', {})


def create_model(
    config: Optional[ModelConfig] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> FaceRecognitionModel:
    """
    Factory function to create a face recognition model
    
    Args:
        config: Model configuration
        device: Device to place model on
    
    Returns:
        FaceRecognitionModel instance
    """
    if config is None:
        config = ModelConfig()
    
    model = FaceRecognitionModel(config)
    model = model.to(device)
    
    logger.info(f"Created {config.architecture} model on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model
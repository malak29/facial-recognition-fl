import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import json
from pathlib import Path
import pickle
from datetime import datetime
from loguru import logger
import onnx
import onnxruntime as ort
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class ModelCheckpoint:
    """
    Model checkpointing with versioning and metadata
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        save_best_only: bool = True,
        monitor_metric: str = 'val_loss',
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics
            metadata: Additional metadata
        
        Returns:
            Path to saved checkpoint or None
        """
        current_metric = metrics.get(self.monitor_metric)
        
        if current_metric is None:
            logger.warning(f"Metric {self.monitor_metric} not found in metrics")
            return None
        
        # Check if should save
        should_save = False
        if self.save_best_only:
            if self.mode == 'min' and current_metric < self.best_metric:
                should_save = True
                self.best_metric = current_metric
            elif self.mode == 'max' and current_metric > self.best_metric:
                should_save = True
                self.best_metric = current_metric
        else:
            should_save = True
        
        if not should_save:
            return None
        
        # Create checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'model_hash': self._compute_model_hash(model)
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        logger.info(
            f"Saved checkpoint to {checkpoint_path} "
            f"({self.monitor_metric}={current_metric:.4f})"
        )
        
        # Remove old checkpoints if needed
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {old_checkpoint}")
        
        return checkpoint_path
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load best checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to restore state
        
        Returns:
            Checkpoint metadata
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints available")
        
        # Best checkpoint is the most recent one if save_best_only is True
        best_checkpoint = self.checkpoints[-1]
        
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {best_checkpoint}")
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint['metrics'],
            'metadata': checkpoint.get('metadata', {})
        }
    
    @staticmethod
    def _compute_model_hash(model: nn.Module) -> str:
        """Compute hash of model weights for versioning"""
        hasher = hashlib.sha256()
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]


class ModelRegistry:
    """
    Model registry for tracking and managing model versions
    """
    
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {'models': {}, 'experiments': {}}
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(
        self,
        model_name: str,
        model_path: Path,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_name: Name of the model
            model_path: Path to model checkpoint
            metrics: Model performance metrics
            metadata: Additional metadata
            tags: Model tags
        
        Returns:
            Model version ID
        """
        timestamp = datetime.now().isoformat()
        version_id = hashlib.sha256(
            f"{model_name}_{timestamp}".encode()
        ).hexdigest()[:16]
        
        model_entry = {
            'name': model_name,
            'version_id': version_id,
            'path': str(model_path),
            'metrics': metrics,
            'metadata': metadata or {},
            'tags': tags or [],
            'timestamp': timestamp,
            'status': 'registered'
        }
        
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = []
        
        self.registry['models'][model_name].append(model_entry)
        self._save_registry()
        
        logger.info(f"Registered model {model_name} version {version_id}")
        
        return version_id
    
    def get_model(
        self,
        model_name: str,
        version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model information
        
        Args:
            model_name: Name of the model
            version_id: Specific version ID (latest if None)
        
        Returns:
            Model entry
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} not found in registry")
        
        models = self.registry['models'][model_name]
        
        if version_id is None:
            # Return latest version
            return models[-1]
        
        # Find specific version
        for model in models:
            if model['version_id'] == version_id:
                return model
        
        raise ValueError(f"Version {version_id} not found for model {model_name}")
    
    def promote_model(
        self,
        model_name: str,
        version_id: str,
        stage: str = 'production'
    ):
        """
        Promote model to a specific stage
        
        Args:
            model_name: Name of the model
            version_id: Version to promote
            stage: Target stage (staging, production)
        """
        model = self.get_model(model_name, version_id)
        model['status'] = stage
        model['promoted_at'] = datetime.now().isoformat()
        
        self._save_registry()
        
        logger.info(f"Promoted {model_name} version {version_id} to {stage}")
    
    def list_models(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all registered models
        
        Args:
            tags: Filter by tags
        
        Returns:
            List of model entries
        """
        all_models = []
        
        for model_name, versions in self.registry['models'].items():
            for version in versions:
                if tags is None or any(tag in version['tags'] for tag in tags):
                    all_models.append(version)
        
        return all_models


class ModelSerializer:
    """
    Model serialization for deployment
    """
    
    @staticmethod
    def export_onnx(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ):
        """
        Export model to ONNX format
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
        """
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify the exported model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Exported model to ONNX: {output_path}")
    
    @staticmethod
    def export_torchscript(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
        optimize: bool = True
    ):
        """
        Export model to TorchScript
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            output_path: Path to save TorchScript model
            optimize: Whether to optimize the script
        """
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save the model
        traced_model.save(str(output_path))
        
        logger.info(f"Exported model to TorchScript: {output_path}")
    
    @staticmethod
    def quantize_model(
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        backend: str = 'qnnpack'
    ) -> nn.Module:
        """
        Quantize model for deployment
        
        Args:
            model: PyTorch model
            calibration_data: Calibration dataloader
            backend: Quantization backend
        
        Returns:
            Quantized model
        """
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Prepare model
        prepared_model = torch.quantization.prepare(model)
        
        # Calibrate with representative data
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, dict):
                    images = batch['images']
                else:
                    images = batch[0]
                prepared_model(images)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        logger.info("Model quantized successfully")
        
        return quantized_model


def get_model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Get model summary including parameters and FLOPs
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run model on
    
    Returns:
        Model summary dictionary
    """
    from thop import profile, clever_format
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate FLOPs
    dummy_input = torch.randn(1, *input_shape).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops, _ = clever_format([flops, 0], "%.3f")
    
    # Get model size
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'flops': flops,
        'model_size_mb': model_size_mb,
        'input_shape': input_shape
    }
    
    logger.info(f"Model Summary: {summary}")
    
    return summary

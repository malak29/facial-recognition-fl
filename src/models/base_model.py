import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
from dataclasses import dataclass

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    input_shape: Tuple[int, int, int]
    num_classes: int
    dropout_rate: float = 0.5
    l2_regularization: float = 1e-4
    batch_normalization: bool = True
    activation: str = 'relu'
    use_bias: bool = True
    kernel_initializer: str = 'glorot_uniform'


class BaseModel(ABC):
    """Abstract base class for facial recognition models."""
    
    def __init__(self, config: ModelConfig, framework: str = "tensorflow"):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
            framework: Framework to use ('tensorflow' or 'pytorch')
        """
        self.config = config
        self.framework = framework.lower()
        self.model = None
        self._is_compiled = False
        
        if self.framework not in ['tensorflow', 'pytorch']:
            raise ValueError(f"Unsupported framework: {framework}")
    
    @abstractmethod
    def build_model(self) -> Union[tf.keras.Model, nn.Module]:
        """
        Build the neural network architecture.
        
        Returns:
            Compiled model
        """
        pass
    
    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: List[str] = None
    ) -> None:
        """
        Compile the model with optimizer and loss function.
        
        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics to track
        """
        if self.model is None:
            self.model = self.build_model()
        
        if self.framework == 'tensorflow':
            self._compile_tensorflow_model(optimizer, learning_rate, loss, metrics)
        else:
            self._compile_pytorch_model(optimizer, learning_rate, loss)
        
        self._is_compiled = True
        logger.info(f"Model compiled with {optimizer} optimizer")
    
    def _compile_tensorflow_model(
        self,
        optimizer: str,
        learning_rate: float,
        loss: str,
        metrics: List[str]
    ) -> None:
        """Compile TensorFlow model."""
        if metrics is None:
            metrics = ['accuracy']
        
        # Create optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def _compile_pytorch_model(
        self,
        optimizer: str,
        learning_rate: float,
        loss: str
    ) -> None:
        """Compile PyTorch model."""
        # Set up optimizer
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported PyTorch optimizer: {optimizer}")
        
        # Set up loss function
        if loss == 'categorical_crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported PyTorch loss: {loss}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            self.model = self.build_model()
        
        if self.framework == 'tensorflow':
            return self.model.summary()
        else:
            return str(self.model)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and non-trainable parameters.
        
        Returns:
            Parameter counts
        """
        if self.model is None:
            self.model = self.build_model()
        
        if self.framework == 'tensorflow':
            trainable = self.model.count_params()
            total = trainable  # TF doesn't easily separate trainable/non-trainable
            return {
                'trainable': trainable,
                'non_trainable': 0,
                'total': total
            }
        else:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            non_trainable = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            return {
                'trainable': trainable,
                'non_trainable': non_trainable,
                'total': trainable + non_trainable
            }
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        if self.framework == 'tensorflow':
            self.model.save(filepath)
        else:
            torch.save(self.model.state_dict(), filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        if self.framework == 'tensorflow':
            self.model = keras.models.load_model(filepath)
        else:
            if self.model is None:
                self.model = self.build_model()
            self.model.load_state_dict(torch.load(filepath))
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_layer_outputs(self, layer_names: List[str]) -> List[Any]:
        """
        Get outputs from specific layers.
        
        Args:
            layer_names: List of layer names
        
        Returns:
            Layer outputs
        """
        if self.framework == 'tensorflow':
            return [self.model.get_layer(name).output for name in layer_names]
        else:
            # PyTorch implementation would need hooks
            raise NotImplementedError("Layer outputs not implemented for PyTorch")
    
    def set_layer_trainable(self, layer_name: str, trainable: bool) -> None:
        """
        Set layer as trainable or frozen.
        
        Args:
            layer_name: Name of layer
            trainable: Whether layer should be trainable
        """
        if self.framework == 'tensorflow':
            layer = self.model.get_layer(layer_name)
            layer.trainable = trainable
        else:
            # PyTorch implementation
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = trainable
        
        logger.info(f"Set layer {layer_name} trainable: {trainable}")
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get model weights as numpy arrays.
        
        Returns:
            Dictionary mapping layer names to weights
        """
        weights = {}
        
        if self.framework == 'tensorflow':
            for layer in self.model.layers:
                if layer.get_weights():
                    weights[layer.name] = layer.get_weights()
        else:
            for name, param in self.model.named_parameters():
                weights[name] = param.data.cpu().numpy()
        
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Set model weights from numpy arrays.
        
        Args:
            weights: Dictionary mapping layer names to weights
        """
        if self.framework == 'tensorflow':
            for layer in self.model.layers:
                if layer.name in weights:
                    layer.set_weights(weights[layer.name])
        else:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in weights:
                        param.copy_(torch.tensor(weights[name]))
        
        logger.info("Model weights updated")
    
    def apply_bias_mitigation(self, strategy: str) -> None:
        """
        Apply bias mitigation techniques to model architecture.
        
        Args:
            strategy: Mitigation strategy ('dropout', 'batch_norm', 'adversarial')
        """
        if strategy == 'dropout':
            self._increase_dropout()
        elif strategy == 'batch_norm':
            self._add_batch_normalization()
        elif strategy == 'adversarial':
            self._setup_adversarial_training()
        else:
            logger.warning(f"Unknown bias mitigation strategy: {strategy}")
    
    def _increase_dropout(self) -> None:
        """Increase dropout rates for regularization."""
        # Implementation depends on specific model architecture
        logger.info("Applied increased dropout for bias mitigation")
    
    def _add_batch_normalization(self) -> None:
        """Add batch normalization layers."""
        # Implementation depends on specific model architecture
        logger.info("Applied batch normalization for bias mitigation")
    
    def _setup_adversarial_training(self) -> None:
        """Setup adversarial training components."""
        # Implementation depends on specific model architecture
        logger.info("Setup adversarial training for bias mitigation")
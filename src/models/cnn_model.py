mport logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, applications
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from .base_model import BaseModel, ModelConfig
from config.settings import settings

logger = logging.getLogger(__name__)


class CNNModel(BaseModel):
    """Custom CNN architecture for facial recognition."""
    
    def __init__(
        self,
        config: ModelConfig,
        framework: str = "tensorflow",
        architecture: str = "custom"  # custom, simple, deep
    ):
        """
        Initialize CNN model.
        
        Args:
            config: Model configuration
            framework: Framework to use
            architecture: Architecture complexity
        """
        super().__init__(config, framework)
        self.architecture = architecture
    
    def build_model(self) -> Union[tf.keras.Model, nn.Module]:
        """Build CNN architecture."""
        if self.framework == 'tensorflow':
            return self._build_tensorflow_model()
        else:
            return self._build_pytorch_model()
    
    def _build_tensorflow_model(self) -> tf.keras.Model:
        """Build TensorFlow CNN model."""
        inputs = keras.Input(shape=self.config.input_shape)
        
        if self.architecture == "simple":
            x = self._simple_cnn_blocks(inputs)
        elif self.architecture == "deep":
            x = self._deep_cnn_blocks(inputs)
        else:
            x = self._custom_cnn_blocks(inputs)
        
        # Global pooling and classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(
            512,
            activation=self.config.activation,
            kernel_regularizer=keras.regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        x = layers.Dense(
            256,
            activation=self.config.activation,
            kernel_regularizer=keras.regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.dropout_rate / 2)(x)
        
        # Output layer
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        model = keras.Model(inputs, outputs, name="facial_recognition_cnn")
        return model
    
    def _simple_cnn_blocks(self, inputs):
        """Simple CNN architecture."""
        x = inputs
        
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation=self.config.activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(64, (3, 3), activation=self.config.activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(128, (3, 3), activation=self.config.activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        return x
    
    def _custom_cnn_blocks(self, inputs):
        """Custom CNN architecture with bias mitigation features."""
        x = inputs
        
        # Initial conv block
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation=self.config.activation)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.Dropout(0.1)(x)
        
        # Residual-like blocks with attention
        x = self._residual_block(x, 64, "block1")
        x = self._residual_block(x, 128, "block2", stride=2)
        x = self._attention_block(x, "attention1")
        x = self._residual_block(x, 256, "block3", stride=2)
        x = self._attention_block(x, "attention2")
        x = self._residual_block(x, 512, "block4", stride=2)
        
        return x
    
    def _deep_cnn_blocks(self, inputs):
        """Deep CNN architecture."""
        x = inputs
        
        # Progressive feature extraction
        filters = [64, 128, 256, 512, 1024]
        
        for i, f in enumerate(filters):
            x = layers.Conv2D(f, (3, 3), activation=self.config.activation, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(f, (3, 3), activation=self.config.activation, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            if i < len(filters) - 1:  # Don't pool after last layer
                x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Dropout(0.3)(x)
        
        return x
    
    def _residual_block(self, x, filters, name, stride=1):
        """Residual block with batch normalization."""
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation(self.config.activation, name=f"{name}_act1")(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, name=f"{name}_shortcut")(shortcut)
            shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)
        
        x = layers.Add(name=f"{name}_add")([x, shortcut])
        x = layers.Activation(self.config.activation, name=f"{name}_act2")(x)
        x = layers.Dropout(0.2)(x)
        
        return x
    
    def _attention_block(self, x, name):
        """Channel attention mechanism."""
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        # Shared MLP
        dense1 = layers.Dense(x.shape[-1] // 8, activation='relu')
        dense2 = layers.Dense(x.shape[-1], activation='sigmoid')
        
        avg_out = dense2(dense1(avg_pool))
        max_out = dense2(dense1(max_pool))
        
        channel_attention = layers.Add()([avg_out, max_out])
        channel_attention = layers.Reshape((1, 1, x.shape[-1]))(channel_attention)
        
        x = layers.Multiply(name=f"{name}_channel_att")([x, channel_attention])
        
        return x
    
    def _build_pytorch_model(self) -> nn.Module:
        """Build PyTorch CNN model."""
        return CNNModelPyTorch(self.config, self.architecture)


class CNNModelPyTorch(nn.Module):
    """PyTorch CNN implementation."""
    
    def __init__(self, config: ModelConfig, architecture: str = "custom"):
        super().__init__()
        self.config = config
        self.architecture = architecture
        
        if architecture == "simple":
            self._build_simple_model()
        elif architecture == "deep":
            self._build_deep_model()
        else:
            self._build_custom_model()
    
    def _build_simple_model(self):
        """Build simple CNN in PyTorch."""
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.config.dropout_rate / 2),
            nn.Linear(256, self.config.num_classes)
        )
    
    def _build_custom_model(self):
        """Build custom CNN with attention."""
        # Implementation similar to TensorFlow version
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout2d(0.1),
        )
        
        # Add residual blocks (simplified)
        self.block1 = self._make_residual_block(64, 64)
        self.block2 = self._make_residual_block(64, 128, stride=2)
        self.block3 = self._make_residual_block(128, 256, stride=2)
        self.block4 = self._make_residual_block(256, 512, stride=2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.config.dropout_rate / 2),
            nn.Linear(256, self.config.num_classes)
        )
    
    def _build_deep_model(self):
        """Build deep CNN model."""
        layers = []
        in_channels = 3
        filters = [64, 128, 256, 512, 1024]
        
        for i, f in enumerate(filters):
            layers.extend([
                nn.Conv2d(in_channels, f, 3, padding=1),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ])
            
            if i < len(filters) - 1:
                layers.append(nn.MaxPool2d(2))
            
            layers.append(nn.Dropout2d(0.3))
            in_channels = f
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, self.config.num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
    
    def forward(self, x):
        """Forward pass."""
        if self.architecture == "custom":
            x = self.features(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.classifier(x)
        else:
            x = self.features(x)
            x = self.classifier(x)
        
        return x


class ResNetModel(BaseModel):
    """ResNet-based model for facial recognition."""
    
    def __init__(
        self,
        config: ModelConfig,
        framework: str = "tensorflow",
        variant: str = "resnet50"  # resnet50, resnet101, resnet152
    ):
        super().__init__(config, framework)
        self.variant = variant
    
    def build_model(self) -> Union[tf.keras.Model, nn.Module]:
        """Build ResNet model."""
        if self.framework == 'tensorflow':
            return self._build_tensorflow_resnet()
        else:
            return self._build_pytorch_resnet()
    
    def _build_tensorflow_resnet(self) -> tf.keras.Model:
        """Build TensorFlow ResNet model."""
        # Load pre-trained ResNet
        if self.variant == "resnet50":
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.config.input_shape
            )
        elif self.variant == "resnet101":
            base_model = applications.ResNet101(
                weights='imagenet',
                include_top=False,
                input_shape=self.config.input_shape
            )
        else:
            base_model = applications.ResNet152(
                weights='imagenet',
                include_top=False,
                input_shape=self.config.input_shape
            )
        
        # Freeze early layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add custom head
        inputs = base_model.input
        x = base_model.output
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Bias mitigation layers
        x = layers.Dense(
            1024,
            activation=self.config.activation,
            kernel_regularizer=keras.regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        x = layers.Dense(
            512,
            activation=self.config.activation,
            kernel_regularizer=keras.regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.dropout_rate / 2)(x)
        
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        model = keras.Model(inputs, outputs, name=f"facial_recognition_{self.variant}")
        return model
    
    def _build_pytorch_resnet(self) -> nn.Module:
        """Build PyTorch ResNet model."""
        import torchvision.models as models
        
        if self.variant == "resnet50":
            model = models.resnet50(pretrained=True)
        elif self.variant == "resnet101":
            model = models.resnet101(pretrained=True)
        else:
            model = models.resnet152(pretrained=True)
        
        # Modify final layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.config.dropout_rate / 2),
            nn.Linear(512, self.config.num_classes)
        )
        
        return model


class EfficientNetModel(BaseModel):
    """EfficientNet-based model for facial recognition."""
    
    def __init__(
        self,
        config: ModelConfig,
        framework: str = "tensorflow",
        variant: str = "b0"  # b0, b1, b2, b3, b4, b5, b6, b7
    ):
        super().__init__(config, framework)
        self.variant = variant
    
    def build_model(self) -> Union[tf.keras.Model, nn.Module]:
        """Build EfficientNet model."""
        if self.framework == 'tensorflow':
            return self._build_tensorflow_efficientnet()
        else:
            return self._build_pytorch_efficientnet()
    
    def _build_tensorflow_efficientnet(self) -> tf.keras.Model:
        """Build TensorFlow EfficientNet model."""
        # Load pre-trained EfficientNet
        base_model = getattr(applications, f'EfficientNet{self.variant.upper()}')(
            weights='imagenet',
            include_top=False,
            input_shape=self.config.input_shape
        )
        
        # Fine-tuning setup
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add custom head
        inputs = base_model.input
        x = base_model.output
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Custom classification head
        x = layers.Dense(
            512,
            activation=self.config.activation,
            kernel_regularizer=keras.regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        model = keras.Model(inputs, outputs, name=f"facial_recognition_efficientnet_{self.variant}")
        return model
    
    def _build_pytorch_efficientnet(self) -> nn.Module:
        """Build PyTorch EfficientNet model."""
        model = EfficientNet.from_pretrained(f'efficientnet-{self.variant}')
        
        # Modify classifier
        num_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, self.config.num_classes)
        )
        
        return model


def create_model(
    model_type: str,
    config: ModelConfig,
    framework: str = "tensorflow",
    **kwargs
) -> BaseModel:
    """
    Factory function to create appropriate model.
    
    Args:
        model_type: Type of model ('cnn', 'resnet', 'efficientnet')
        config: Model configuration
        framework: Framework to use
        **kwargs: Additional model-specific arguments
    
    Returns:
        Configured model instance
    """
    if model_type == "cnn":
        return CNNModel(config, framework, **kwargs)
    elif model_type == "resnet":
        return ResNetModel(config, framework, **kwargs)
    elif model_type == "efficientnet":
        return EfficientNetModel(config, framework, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
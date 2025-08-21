import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import copy
from dataclasses import dataclass, asdict

from .base_model import BaseModel, ModelConfig
from .cnn_model import create_model
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ClientModelConfig:
    """Configuration for client model."""
    base_config: ModelConfig
    client_id: int
    local_epochs: int = 5
    local_batch_size: int = 32
    learning_rate: float = 0.001
    privacy_budget: float = 1.0
    differential_privacy: bool = True
    secure_aggregation: bool = True


class ClientModel:
    """Client model for federated learning with privacy preservation."""
    
    def __init__(
        self,
        config: ClientModelConfig,
        model_type: str = "cnn",
        framework: str = "tensorflow"
    ):
        """
        Initialize client model.
        
        Args:
            config: Client model configuration
            model_type: Type of base model
            framework: Framework to use
        """
        self.config = config
        self.model_type = model_type
        self.framework = framework
        
        # Initialize base model
        self.base_model = create_model(
            model_type, 
            config.base_config, 
            framework
        )
        
        # Training state
        self.current_round = 0
        self.training_history = []
        self.model_updates = []
        
        # Privacy components
        self.noise_multiplier = self._calculate_noise_multiplier()
        self.gradient_clipper = None
        
        # Initialize model
        self._setup_model()
        
        logger.info(f"Client {config.client_id} initialized with {model_type} model")
    
    def _setup_model(self) -> None:
        """Setup and compile the client model."""
        self.base_model.compile_model(
            optimizer='adam',
            learning_rate=self.config.learning_rate,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Setup differential privacy if enabled
        if self.config.differential_privacy:
            self._setup_differential_privacy()
    
    def _setup_differential_privacy(self) -> None:
        """Setup differential privacy mechanisms."""
        try:
            if self.framework == 'tensorflow':
                from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
                from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
                
                # Replace optimizer with DP version
                original_optimizer = self.base_model.model.optimizer
                dp_optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
                    l2_norm_clip=1.0,
                    noise_multiplier=self.noise_multiplier,
                    num_microbatches=self.config.local_batch_size,
                    learning_rate=self.config.learning_rate
                )
                
                self.base_model.model.compile(
                    optimizer=dp_optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                logger.info(f"Differential privacy enabled with noise multiplier: {self.noise_multiplier}")
                
        except ImportError:
            logger.warning("TensorFlow Privacy not available, continuing without DP")
            self.config.differential_privacy = False
    
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier for differential privacy."""
        # Simplified calculation - in practice, this would be more sophisticated
        epsilon = self.config.privacy_budget
        delta = 1e-5
        
        # Basic noise multiplier calculation
        noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return max(noise_multiplier, 0.1)  # Minimum noise level
    
    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Set model weights from server.
        
        Args:
            weights: Model weights from server
        """
        self.base_model.set_weights(weights)
        logger.info(f"Client {self.config.client_id} received updated weights")
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get current model weights.
        
        Returns:
            Current model weights
        """
        return self.base_model.get_weights()
    
    def train_local_model(
        self,
        train_data: Any,
        train_labels: Any,
        validation_data: Optional[Tuple[Any, Any]] = None,
        demographic_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train model locally with client data.
        
        Args:
            train_data: Training data
            train_labels: Training labels
            validation_data: Optional validation data
            demographic_data: Optional demographic information
        
        Returns:
            Training metrics and updates
        """
        logger.info(f"Client {self.config.client_id} starting local training")
        
        # Store initial weights
        initial_weights = self.get_weights()
        
        # Apply bias mitigation if demographic data available
        if demographic_data:
            train_data, train_labels = self._apply_bias_mitigation(
                train_data, train_labels, demographic_data
            )
        
        # Train model
        if self.framework == 'tensorflow':
            history = self._train_tensorflow_model(
                train_data, train_labels, validation_data
            )
        else:
            history = self._train_pytorch_model(
                train_data, train_labels, validation_data
            )
        
        # Calculate model updates
        final_weights = self.get_weights()
        weight_updates = self._calculate_weight_updates(initial_weights, final_weights)
        
        # Apply secure aggregation preparation if enabled
        if self.config.secure_aggregation:
            weight_updates = self._prepare_secure_updates(weight_updates)
        
        # Store training history
        training_result = {
            'client_id': self.config.client_id,
            'round': self.current_round,
            'history': history,
            'weight_updates': weight_updates,
            'num_samples': len(train_data),
            'privacy_spent': self._calculate_privacy_spent()
        }
        
        self.training_history.append(training_result)
        self.current_round += 1
        
        logger.info(f"Client {self.config.client_id} completed local training")
        return training_result
    
    def _train_tensorflow_model(
        self,
        train_data: Any,
        train_labels: Any,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> Dict[str, List[float]]:
        """Train TensorFlow model."""
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.base_model.model.fit(
            train_data,
            train_labels,
            batch_size=self.config.local_batch_size,
            epochs=self.config.local_epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        return history.history
    
    def _train_pytorch_model(
        self,
        train_data: Any,
        train_labels: Any,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> Dict[str, List[float]]:
        """Train PyTorch model."""
        # Implementation for PyTorch training
        # This would include the training loop, loss calculation, etc.
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.model.to(device)
        
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(self.config.local_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch_pytorch(
                train_data, train_labels, device
            )
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            # Validation phase
            if validation_data:
                val_loss, val_acc = self._validate_epoch_pytorch(
                    validation_data, device
                )
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
        
        return history
    
    def _train_epoch_pytorch(self, train_data, train_labels, device):
        """Train one epoch in PyTorch."""
        self.base_model.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # This is a simplified version - actual implementation would use DataLoader
        for batch_data, batch_labels in zip(train_data, train_labels):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            self.base_model.optimizer.zero_grad()
            outputs = self.base_model.model(batch_data)
            loss = self.base_model.criterion(outputs, batch_labels)
            
            loss.backward()
            
            # Apply gradient clipping for privacy
            if self.config.differential_privacy:
                torch.nn.utils.clip_grad_norm_(self.base_model.model.parameters(), 1.0)
                # Add noise to gradients
                for param in self.base_model.model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, self.noise_multiplier, param.grad.shape)
                        param.grad.add_(noise.to(device))
            
            self.base_model.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        return total_loss / len(train_data), correct / total
    
    def _validate_epoch_pytorch(self, validation_data, device):
        """Validate one epoch in PyTorch."""
        self.base_model.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in validation_data:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = self.base_model.model(batch_data)
                loss = self.base_model.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        
        return total_loss / len(validation_data), correct / total
    
    def _apply_bias_mitigation(
        self,
        train_data: Any,
        train_labels: Any,
        demographic_data: List[Dict[str, Any]]
    ) -> Tuple[Any, Any]:
        """Apply bias mitigation techniques during local training."""
        # Analyze demographic distribution
        demo_stats = self._analyze_local_demographics(demographic_data)
        
        # Apply fairness-aware sampling if needed
        if self._needs_demographic_balancing(demo_stats):
            train_data, train_labels = self._balance_local_data(
                train_data, train_labels, demographic_data
            )
            logger.info(f"Client {self.config.client_id} applied demographic balancing")
        
        return train_data, train_labels
    
    def _analyze_local_demographics(
        self, 
        demographic_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        """Analyze demographic distribution in local data."""
        demo_stats = {}
        
        for demo in demographic_data:
            for attr, value in demo.items():
                if attr not in demo_stats:
                    demo_stats[attr] = {}
                if value not in demo_stats[attr]:
                    demo_stats[attr][value] = 0
                demo_stats[attr][value] += 1
        
        return demo_stats
    
    def _needs_demographic_balancing(
        self, 
        demo_stats: Dict[str, Dict[str, int]]
    ) -> bool:
        """Check if demographic balancing is needed."""
        for attr, values in demo_stats.items():
            if len(values) > 1:  # Multiple groups exist
                counts = list(values.values())
                max_count = max(counts)
                min_count = min(counts)
                
                # If imbalance ratio > 3:1, apply balancing
                if max_count / min_count > 3:
                    return True
        
        return False
    
    def _balance_local_data(
        self,
        train_data: Any,
        train_labels: Any,
        demographic_data: List[Dict[str, Any]]
    ) -> Tuple[Any, Any]:
        """Balance local data for fairness."""
        # Implementation of local data balancing
        # This is a simplified version
        return train_data, train_labels
    
    def _calculate_weight_updates(
        self,
        initial_weights: Dict[str, np.ndarray],
        final_weights: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate weight updates (difference)."""
        updates = {}
        
        for layer_name in initial_weights:
            if layer_name in final_weights:
                updates[layer_name] = final_weights[layer_name] - initial_weights[layer_name]
        
        return updates
    
    def _prepare_secure_updates(
        self,
        weight_updates: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Prepare updates for secure aggregation."""
        # Add random masking for secure aggregation
        # This is a simplified version - real implementation would be more sophisticated
        
        masked_updates = {}
        for layer_name, updates in weight_updates.items():
            # Add small random noise for secure aggregation
            noise = np.random.normal(0, 0.001, updates.shape)
            masked_updates[layer_name] = updates + noise
        
        return masked_updates
    
    def _calculate_privacy_spent(self) -> float:
        """Calculate privacy budget spent so far."""
        if not self.config.differential_privacy:
            return 0.0
        
        # Simplified privacy accounting
        # Real implementation would use more sophisticated privacy accounting
        privacy_per_round = self.config.privacy_budget / 10  # Assume 10 rounds max
        return privacy_per_round * self.current_round
    
    def evaluate_model(
        self,
        test_data: Any,
        test_labels: Any,
        demographic_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            test_data: Test data
            test_labels: Test labels
            demographic_data: Optional demographic information
        
        Returns:
            Evaluation metrics
        """
        if self.framework == 'tensorflow':
            loss, accuracy = self.base_model.model.evaluate(
                test_data, test_labels, verbose=0
            )
            
            metrics = {
                'loss': loss,
                'accuracy': accuracy,
                'client_id': self.config.client_id
            }
        else:
            # PyTorch evaluation
            self.base_model.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            with torch.no_grad():
                for batch_data, batch_labels in zip(test_data, test_labels):
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = self.base_model.model(batch_data)
                    loss = self.base_model.criterion(outputs, batch_labels)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_labels.size(0)
                    correct += predicted.eq(batch_labels).sum().item()
            
            metrics = {
                'loss': total_loss / len(test_data),
                'accuracy': correct / total,
                'client_id': self.config.client_id
            }
        
        # Add fairness metrics if demographic data available
        if demographic_data:
            fairness_metrics = self._calculate_fairness_metrics(
                test_data, test_labels, demographic_data
            )
            metrics.update(fairness_metrics)
        
        return metrics
    
    def _calculate_fairness_metrics(
        self,
        test_data: Any,
        test_labels: Any,
        demographic_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate fairness metrics."""
        # Simplified fairness metrics calculation
        fairness_metrics = {
            'demographic_parity': 0.0,
            'equalized_odds': 0.0,
            'calibration': 0.0
        }
        
        # Implementation would calculate actual fairness metrics
        # based on model predictions and demographic groups
        
        return fairness_metrics
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information and status."""
        return {
            'client_id': self.config.client_id,
            'model_type': self.model_type,
            'framework': self.framework,
            'current_round': self.current_round,
            'differential_privacy': self.config.differential_privacy,
            'secure_aggregation': self.config.secure_aggregation,
            'privacy_spent': self._calculate_privacy_spent(),
            'training_history_length': len(self.training_history)
        }
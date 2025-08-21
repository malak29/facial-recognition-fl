import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from .base_model import BaseModel, ModelConfig
from .cnn_model import create_model
from .client_model import ClientModelConfig
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ServerModelConfig:
    """Configuration for server model."""
    base_config: ModelConfig
    num_clients: int
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox, scaffold
    client_fraction: float = 0.3  # Fraction of clients to select per round
    min_clients: int = 2  # Minimum clients needed for aggregation
    max_rounds: int = 100
    convergence_threshold: float = 0.001
    model_checkpoint_frequency: int = 5
    evaluation_frequency: int = 1


class ServerModel:
    """Server model for federated learning with advanced aggregation strategies."""
    
    def __init__(
        self,
        config: ServerModelConfig,
        model_type: str = "cnn",
        framework: str = "tensorflow"
    ):
        """
        Initialize server model.
        
        Args:
            config: Server model configuration
            model_type: Type of base model
            framework: Framework to use
        """
        self.config = config
        self.model_type = model_type
        self.framework = framework
        
        # Initialize global model
        self.global_model = create_model(
            model_type,
            config.base_config,
            framework
        )
        
        # Initialize model
        self._setup_model()
        
        # Training state
        self.current_round = 0
        self.client_updates_history = []
        self.global_model_history = []
        self.convergence_history = []
        
        # Client management
        self.registered_clients = {}
        self.client_selection_history = []
        
        # Aggregation components
        self.aggregator = self._create_aggregator()
        
        # Thread safety
        self.model_lock = threading.Lock()
        
        logger.info(f"Server initialized with {model_type} model and {config.aggregation_strategy} aggregation")
    
    def _setup_model(self) -> None:
        """Setup and compile the global model."""
        self.global_model.compile_model(
            optimizer='adam',
            learning_rate=0.001,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _create_aggregator(self):
        """Create appropriate aggregation strategy."""
        if self.config.aggregation_strategy == "fedavg":
            return FedAvgAggregator()
        elif self.config.aggregation_strategy == "fedprox":
            return FedProxAggregator(mu=0.01)
        elif self.config.aggregation_strategy == "scaffold":
            return ScaffoldAggregator()
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.config.aggregation_strategy}")
    
    def register_client(
        self,
        client_id: int,
        client_config: ClientModelConfig
    ) -> None:
        """
        Register a client with the server.
        
        Args:
            client_id: Unique client identifier
            client_config: Client configuration
        """
        self.registered_clients[client_id] = {
            'config': client_config,
            'status': 'registered',
            'last_seen': None,
            'participation_count': 0,
            'avg_accuracy': 0.0
        }
        
        logger.info(f"Client {client_id} registered successfully")
    
    def select_clients(
        self,
        round_num: int,
        availability_scores: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """
        Select clients for the current round.
        
        Args:
            round_num: Current round number
            availability_scores: Optional client availability scores
        
        Returns:
            List of selected client IDs
        """
        available_clients = list(self.registered_clients.keys())
        
        if not available_clients:
            raise ValueError("No clients registered")
        
        # Calculate number of clients to select
        num_select = max(
            self.config.min_clients,
            int(len(available_clients) * self.config.client_fraction)
        )
        num_select = min(num_select, len(available_clients))
        
        # Selection strategy
        if availability_scores:
            # Select based on availability scores
            sorted_clients = sorted(
                available_clients,
                key=lambda c: availability_scores.get(c, 0.0),
                reverse=True
            )
            selected_clients = sorted_clients[:num_select]
        else:
            # Random selection
            selected_clients = np.random.choice(
                available_clients,
                size=num_select,
                replace=False
            ).tolist()
        
        # Update participation history
        for client_id in selected_clients:
            self.registered_clients[client_id]['participation_count'] += 1
            self.registered_clients[client_id]['status'] = 'selected'
        
        self.client_selection_history.append({
            'round': round_num,
            'selected_clients': selected_clients,
            'total_available': len(available_clients)
        })
        
        logger.info(f"Round {round_num}: Selected {len(selected_clients)} clients: {selected_clients}")
        return selected_clients
    
    def get_global_weights(self) -> Dict[str, np.ndarray]:
        """
        Get current global model weights.
        
        Returns:
            Global model weights
        """
        with self.model_lock:
            return self.global_model.get_weights()
    
    def aggregate_client_updates(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        round_num: int
    ) -> Dict[str, Any]:
        """
        Aggregate client model updates.
        
        Args:
            client_updates: Dictionary of client updates
            round_num: Current round number
        
        Returns:
            Aggregation results and metrics
        """
        logger.info(f"Round {round_num}: Aggregating updates from {len(client_updates)} clients")
        
        with self.model_lock:
            # Store current weights for convergence check
            previous_weights = self.global_model.get_weights()
            
            # Perform aggregation
            aggregation_result = self.aggregator.aggregate(
                client_updates,
                previous_weights,
                round_num
            )
            
            # Update global model
            self.global_model.set_weights(aggregation_result['aggregated_weights'])
            
            # Calculate convergence metrics
            convergence_metrics = self._calculate_convergence(
                previous_weights,
                aggregation_result['aggregated_weights']
            )
            
            # Store history
            round_result = {
                'round': round_num,
                'num_clients': len(client_updates),
                'client_ids': list(client_updates.keys()),
                'aggregation_metrics': aggregation_result['metrics'],
                'convergence_metrics': convergence_metrics,
                'global_weights_norm': self._calculate_weights_norm(
                    aggregation_result['aggregated_weights']
                )
            }
            
            self.client_updates_history.append(client_updates)
            self.global_model_history.append(round_result)
            self.convergence_history.append(convergence_metrics['weight_change'])
            
            logger.info(
                f"Round {round_num}: Aggregation completed. "
                f"Weight change: {convergence_metrics['weight_change']:.6f}"
            )
            
            return round_result
    
    def _calculate_convergence(
        self,
        previous_weights: Dict[str, np.ndarray],
        current_weights: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate convergence metrics."""
        total_change = 0.0
        total_norm = 0.0
        layer_changes = {}
        
        for layer_name in previous_weights:
            if layer_name in current_weights:
                prev = previous_weights[layer_name]
                curr = current_weights[layer_name]
                
                # Calculate L2 norm of change
                change = np.linalg.norm(curr - prev)
                norm = np.linalg.norm(curr)
                
                layer_changes[layer_name] = change / (norm + 1e-8)  # Relative change
                total_change += change ** 2
                total_norm += norm ** 2
        
        relative_change = np.sqrt(total_change) / (np.sqrt(total_norm) + 1e-8)
        
        return {
            'weight_change': np.sqrt(total_change),
            'relative_change': relative_change,
            'layer_changes': layer_changes,
            'converged': relative_change < self.config.convergence_threshold
        }
    
    def _calculate_weights_norm(self, weights: Dict[str, np.ndarray]) -> float:
        """Calculate L2 norm of all weights."""
        total_norm = 0.0
        
        for layer_weights in weights.values():
            total_norm += np.linalg.norm(layer_weights) ** 2
        
        return np.sqrt(total_norm)
    
    def evaluate_global_model(
        self,
        test_data: Any,
        test_labels: Any,
        demographic_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate global model performance.
        
        Args:
            test_data: Test data
            test_labels: Test labels
            demographic_data: Optional demographic information
        
        Returns:
            Evaluation metrics
        """
        with self.model_lock:
            if self.framework == 'tensorflow':
                loss, accuracy = self.global_model.model.evaluate(
                    test_data, test_labels, verbose=0
                )
                
                metrics = {
                    'loss': float(loss),
                    'accuracy': float(accuracy),
                    'round': self.current_round
                }
            else:
                # PyTorch evaluation
                self.global_model.model.eval()
                total_loss = 0
                correct = 0
                total = 0
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                with torch.no_grad():
                    for batch_data, batch_labels in zip(test_data, test_labels):
                        batch_data = batch_data.to(device)
                        batch_labels = batch_labels.to(device)
                        
                        outputs = self.global_model.model(batch_data)
                        loss = self.global_model.criterion(outputs, batch_labels)
                        
                        total_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += batch_labels.size(0)
                        correct += predicted.eq(batch_labels).sum().item()
                
                metrics = {
                    'loss': total_loss / len(test_data),
                    'accuracy': correct / total,
                    'round': self.current_round
                }
            
            # Add fairness metrics if demographic data available
            if demographic_data:
                fairness_metrics = self._calculate_global_fairness_metrics(
                    test_data, test_labels, demographic_data
                )
                metrics.update(fairness_metrics)
            
            logger.info(
                f"Global model evaluation - Round {self.current_round}: "
                f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
            )
            
            return metrics
    
    def _calculate_global_fairness_metrics(
        self,
        test_data: Any,
        test_labels: Any,
        demographic_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate global fairness metrics."""
        # Implementation would calculate fairness metrics
        # across different demographic groups
        
        fairness_metrics = {
            'demographic_parity': 0.0,
            'equalized_odds': 0.0,
            'calibration': 0.0,
            'bias_score': 0.0
        }
        
        return fairness_metrics
    
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Save server model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / f"global_model_round_{self.current_round}"
        self.global_model.save_model(str(model_path))
        
        # Save server state
        state_data = {
            'current_round': self.current_round,
            'config': asdict(self.config),
            'registered_clients': self.registered_clients,
            'convergence_history': self.convergence_history[-10:],  # Last 10 rounds
            'client_selection_history': self.client_selection_history[-10:]
        }
        
        state_path = checkpoint_path / f"server_state_round_{self.current_round}.json"
        with open(state_path, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved at round {self.current_round}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load server model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        model_files = list(checkpoint_path.glob("global_model_round_*"))
        if model_files:
            latest_model = max(model_files, key=lambda p: int(p.stem.split('_')[-1]))
            self.global_model.load_model(str(latest_model))
        
        # Load server state
        state_files = list(checkpoint_path.glob("server_state_round_*.json"))
        if state_files:
            latest_state = max(state_files, key=lambda p: int(p.stem.split('_')[-1]))
            
            with open(latest_state, 'r') as f:
                state_data = json.load(f)
            
            self.current_round = state_data['current_round']
            self.registered_clients = state_data['registered_clients']
            self.convergence_history = state_data['convergence_history']
            self.client_selection_history = state_data['client_selection_history']
        
        logger.info(f"Checkpoint loaded from round {self.current_round}")
    
    def check_convergence(self) -> bool:
        """
        Check if the global model has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.convergence_history) < 3:
            return False
        
        # Check if last 3 rounds show convergence
        recent_changes = self.convergence_history[-3:]
        converged = all(change < self.config.convergence_threshold for change in recent_changes)
        
        if converged:
            logger.info(f"Model converged at round {self.current_round}")
        
        return converged
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get comprehensive server statistics."""
        stats = {
            'current_round': self.current_round,
            'total_clients': len(self.registered_clients),
            'aggregation_strategy': self.config.aggregation_strategy,
            'convergence_status': {
                'converged': self.check_convergence(),
                'recent_changes': self.convergence_history[-5:] if self.convergence_history else [],
                'convergence_threshold': self.config.convergence_threshold
            },
            'client_participation': {
                'total_selections': sum(
                    client['participation_count'] 
                    for client in self.registered_clients.values()
                ),
                'avg_participation': np.mean([
                    client['participation_count'] 
                    for client in self.registered_clients.values()
                ]) if self.registered_clients else 0
            },
            'model_info': {
                'type': self.model_type,
                'framework': self.framework,
                'parameters': self.global_model.count_parameters()
            }
        }
        
        return stats


class FedAvgAggregator:
    """Federated Averaging (FedAvg) aggregation strategy."""
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        current_weights: Dict[str, np.ndarray],
        round_num: int
    ) -> Dict[str, Any]:
        """
        Perform FedAvg aggregation.
        
        Args:
            client_updates: Client model updates
            current_weights: Current global weights
            round_num: Current round number
        
        Returns:
            Aggregation result
        """
        # Calculate total samples
        total_samples = sum(
            update['num_samples'] for update in client_updates.values()
        )
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Aggregate each layer
        for layer_name in current_weights:
            weighted_sum = np.zeros_like(current_weights[layer_name])
            
            for client_id, update in client_updates.items():
                if 'weight_updates' in update and layer_name in update['weight_updates']:
                    client_weight = update['num_samples'] / total_samples
                    client_update = update['weight_updates'][layer_name]
                    weighted_sum += client_weight * client_update
            
            # Update global weights
            aggregated_weights[layer_name] = current_weights[layer_name] + weighted_sum
        
        # Calculate aggregation metrics
        metrics = {
            'total_samples': total_samples,
            'num_clients': len(client_updates),
            'aggregation_method': 'fedavg',
            'weights_norm': np.linalg.norm([
                np.linalg.norm(w) for w in aggregated_weights.values()
            ])
        }
        
        return {
            'aggregated_weights': aggregated_weights,
            'metrics': metrics
        }


class FedProxAggregator:
    """FedProx aggregation with proximal term."""
    
    def __init__(self, mu: float = 0.01):
        self.mu = mu  # Proximal term coefficient
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        current_weights: Dict[str, np.ndarray],
        round_num: int
    ) -> Dict[str, Any]:
        """Perform FedProx aggregation."""
        # Similar to FedAvg but with proximal regularization
        # This is a simplified version
        
        total_samples = sum(
            update['num_samples'] for update in client_updates.values()
        )
        
        aggregated_weights = {}
        
        for layer_name in current_weights:
            weighted_sum = np.zeros_like(current_weights[layer_name])
            
            for client_id, update in client_updates.items():
                if 'weight_updates' in update and layer_name in update['weight_updates']:
                    client_weight = update['num_samples'] / total_samples
                    client_update = update['weight_updates'][layer_name]
                    
                    # Apply proximal term
                    proximal_update = client_update * (1 - self.mu)
                    weighted_sum += client_weight * proximal_update
            
            aggregated_weights[layer_name] = current_weights[layer_name] + weighted_sum
        
        metrics = {
            'total_samples': total_samples,
            'num_clients': len(client_updates),
            'aggregation_method': 'fedprox',
            'mu': self.mu
        }
        
        return {
            'aggregated_weights': aggregated_weights,
            'metrics': metrics
        }


class ScaffoldAggregator:
    """SCAFFOLD aggregation with control variates."""
    
    def __init__(self):
        self.server_controls = {}
        self.client_controls = {}
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        current_weights: Dict[str, np.ndarray],
        round_num: int
    ) -> Dict[str, Any]:
        """Perform SCAFFOLD aggregation."""
        # Simplified SCAFFOLD implementation
        # Real implementation would maintain control variates
        
        total_samples = sum(
            update['num_samples'] for update in client_updates.values()
        )
        
        aggregated_weights = {}
        
        for layer_name in current_weights:
            weighted_sum = np.zeros_like(current_weights[layer_name])
            
            for client_id, update in client_updates.items():
                if 'weight_updates' in update and layer_name in update['weight_updates']:
                    client_weight = update['num_samples'] / total_samples
                    client_update = update['weight_updates'][layer_name]
                    weighted_sum += client_weight * client_update
            
            aggregated_weights[layer_name] = current_weights[layer_name] + weighted_sum
        
        metrics = {
            'total_samples': total_samples,
            'num_clients': len(client_updates),
            'aggregation_method': 'scaffold'
        }
        
        return {
            'aggregated_weights': aggregated_weights,
            'metrics': metrics
        }
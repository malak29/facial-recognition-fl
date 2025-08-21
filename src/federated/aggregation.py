import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
import scipy.stats as stats
from collections import defaultdict

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for aggregation algorithms."""
    strategy: str = "fedavg"
    fairness_weight: float = 0.1
    robustness_enabled: bool = True
    byzantine_threshold: float = 0.2
    convergence_threshold: float = 1e-4
    adaptive_lr: bool = True
    momentum: float = 0.9


class FederatedAggregator(ABC):
    """Abstract base class for federated learning aggregators."""
    
    def __init__(self, config: AggregationConfig):
        """
        Initialize aggregator.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config
        self.round_history = []
        self.client_reliability_scores = {}
        
    @abstractmethod
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        global_weights: Dict[str, np.ndarray],
        round_num: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Aggregate client updates.
        
        Args:
            client_updates: Client model updates
            global_weights: Current global model weights
            round_num: Current round number
            **kwargs: Additional arguments
        
        Returns:
            Aggregation result
        """
        pass
    
    def update_client_reliability(
        self,
        client_id: int,
        performance_metrics: Dict[str, float]
    ) -> None:
        """Update client reliability scores."""
        if client_id not in self.client_reliability_scores:
            self.client_reliability_scores[client_id] = {
                'accuracy_history': [],
                'loss_history': [],
                'reliability_score': 1.0,
                'participation_count': 0
            }
        
        client_stats = self.client_reliability_scores[client_id]
        client_stats['accuracy_history'].append(performance_metrics.get('accuracy', 0.0))
        client_stats['loss_history'].append(performance_metrics.get('loss', float('inf')))
        client_stats['participation_count'] += 1
        
        # Calculate reliability score based on recent performance
        recent_accuracy = np.mean(client_stats['accuracy_history'][-5:])
        recent_loss = np.mean(client_stats['loss_history'][-5:])
        
        # Simple reliability scoring
        client_stats['reliability_score'] = min(1.0, recent_accuracy * (1.0 / (1.0 + recent_loss)))
    
    def detect_byzantine_clients(
        self,
        client_updates: Dict[int, Dict[str, Any]]
    ) -> List[int]:
        """
        Detect potentially Byzantine (malicious) clients.
        
        Args:
            client_updates: Client updates to analyze
        
        Returns:
            List of suspicious client IDs
        """
        if not self.config.robustness_enabled:
            return []
        
        byzantine_clients = []
        
        # Calculate update magnitudes for each client
        update_magnitudes = {}
        for client_id, update in client_updates.items():
            if 'weight_updates' in update:
                total_magnitude = 0.0
                for layer_updates in update['weight_updates'].values():
                    total_magnitude += np.linalg.norm(layer_updates)
                update_magnitudes[client_id] = total_magnitude
        
        if not update_magnitudes:
            return []
        
        # Use statistical outlier detection
        magnitudes = list(update_magnitudes.values())
        q75, q25 = np.percentile(magnitudes, [75, 25])
        iqr = q75 - q25
        
        # Define outlier thresholds
        upper_threshold = q75 + 1.5 * iqr
        lower_threshold = q25 - 1.5 * iqr
        
        for client_id, magnitude in update_magnitudes.items():
            if magnitude > upper_threshold or magnitude < lower_threshold:
                byzantine_clients.append(client_id)
        
        # Limit number of clients marked as Byzantine
        max_byzantine = int(len(client_updates) * self.config.byzantine_threshold)
        byzantine_clients = byzantine_clients[:max_byzantine]
        
        if byzantine_clients:
            logger.warning(f"Detected {len(byzantine_clients)} potential Byzantine clients: {byzantine_clients}")
        
        return byzantine_clients
    
    def calculate_client_weights(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        byzantine_clients: List[int]
    ) -> Dict[int, float]:
        """Calculate aggregation weights for clients."""
        weights = {}
        total_samples = 0
        
        # Filter out Byzantine clients
        valid_clients = {
            cid: update for cid, update in client_updates.items()
            if cid not in byzantine_clients
        }
        
        # Calculate sample-based weights
        for client_id, update in valid_clients.items():
            num_samples = update.get('num_samples', 1)
            reliability = self.client_reliability_scores.get(
                client_id, {'reliability_score': 1.0}
            )['reliability_score']
            
            # Combine sample size and reliability
            weight = num_samples * reliability
            weights[client_id] = weight
            total_samples += weight
        
        # Normalize weights
        if total_samples > 0:
            for client_id in weights:
                weights[client_id] /= total_samples
        
        return weights


class FedAvgAggregator(FederatedAggregator):
    """Federated Averaging aggregator with robustness enhancements."""
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        global_weights: Dict[str, np.ndarray],
        round_num: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform robust FedAvg aggregation."""
        
        # Detect Byzantine clients
        byzantine_clients = self.detect_byzantine_clients(client_updates)
        
        # Calculate client weights
        client_weights = self.calculate_client_weights(client_updates, byzantine_clients)
        
        if not client_weights:
            logger.warning("No valid clients for aggregation")
            return {
                'aggregated_weights': global_weights,
                'metrics': {'error': 'No valid clients'}
            }
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Aggregate each layer
        for layer_name in global_weights:
            layer_updates = []
            layer_weights = []
            
            for client_id, weight in client_weights.items():
                update = client_updates[client_id]
                if 'weight_updates' in update and layer_name in update['weight_updates']:
                    layer_updates.append(update['weight_updates'][layer_name])
                    layer_weights.append(weight)
            
            if layer_updates:
                # Weighted average of updates
                weighted_update = np.zeros_like(global_weights[layer_name])
                for update, weight in zip(layer_updates, layer_weights):
                    weighted_update += weight * update
                
                # Update global weights
                aggregated_weights[layer_name] = global_weights[layer_name] + weighted_update
            else:
                # No updates for this layer
                aggregated_weights[layer_name] = global_weights[layer_name]
        
        # Calculate metrics
        metrics = {
            'aggregation_method': 'fedavg',
            'num_valid_clients': len(client_weights),
            'num_byzantine_clients': len(byzantine_clients),
            'byzantine_clients': byzantine_clients,
            'client_weights': client_weights,
            'total_samples': sum(
                client_updates[cid].get('num_samples', 0) 
                for cid in client_weights
            )
        }
        
        return {
            'aggregated_weights': aggregated_weights,
            'metrics': metrics
        }


class FedProxAggregator(FederatedAggregator):
    """FedProx aggregator with proximal term."""
    
    def __init__(self, config: AggregationConfig, mu: float = 0.01):
        super().__init__(config)
        self.mu = mu  # Proximal term coefficient
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        global_weights: Dict[str, np.ndarray],
        round_num: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform FedProx aggregation with proximal regularization."""
        
        byzantine_clients = self.detect_byzantine_clients(client_updates)
        client_weights = self.calculate_client_weights(client_updates, byzantine_clients)
        
        if not client_weights:
            return {
                'aggregated_weights': global_weights,
                'metrics': {'error': 'No valid clients'}
            }
        
        aggregated_weights = {}
        
        for layer_name in global_weights:
            layer_updates = []
            layer_weights = []
            
            for client_id, weight in client_weights.items():
                update = client_updates[client_id]
                if 'weight_updates' in update and layer_name in update['weight_updates']:
                    # Apply proximal term
                    client_update = update['weight_updates'][layer_name]
                    proximal_update = client_update * (1 - self.mu)
                    
                    layer_updates.append(proximal_update)
                    layer_weights.append(weight)
            
            if layer_updates:
                weighted_update = np.zeros_like(global_weights[layer_name])
                for update, weight in zip(layer_updates, layer_weights):
                    weighted_update += weight * update
                
                aggregated_weights[layer_name] = global_weights[layer_name] + weighted_update
            else:
                aggregated_weights[layer_name] = global_weights[layer_name]
        
        metrics = {
            'aggregation_method': 'fedprox',
            'mu': self.mu,
            'num_valid_clients': len(client_weights),
            'num_byzantine_clients': len(byzantine_clients),
            'client_weights': client_weights
        }
        
        return {
            'aggregated_weights': aggregated_weights,
            'metrics': metrics
        }


class FairFedAggregator(FederatedAggregator):
    """Fairness-aware federated aggregator for bias mitigation."""
    
    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self.demographic_history = []
        self.fairness_scores = {}
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        global_weights: Dict[str, np.ndarray],
        round_num: int,
        demographic_data: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform fairness-aware aggregation."""
        
        byzantine_clients = self.detect_byzantine_clients(client_updates)
        base_weights = self.calculate_client_weights(client_updates, byzantine_clients)
        
        # Adjust weights for fairness if demographic data available
        if demographic_data and self.config.fairness_weight > 0:
            fairness_weights = self._calculate_fairness_weights(
                client_updates, demographic_data, base_weights
            )
            
            # Combine base weights with fairness weights
            final_weights = {}
            for client_id in base_weights:
                base_w = base_weights[client_id]
                fair_w = fairness_weights.get(client_id, 1.0)
                final_weights[client_id] = (
                    (1 - self.config.fairness_weight) * base_w +
                    self.config.fairness_weight * fair_w
                )
            
            # Renormalize
            total_weight = sum(final_weights.values())
            if total_weight > 0:
                for client_id in final_weights:
                    final_weights[client_id] /= total_weight
        else:
            final_weights = base_weights
        
        if not final_weights:
            return {
                'aggregated_weights': global_weights,
                'metrics': {'error': 'No valid clients'}
            }
        
        # Aggregate with fairness-adjusted weights
        aggregated_weights = {}
        
        for layer_name in global_weights:
            layer_updates = []
            layer_weights = []
            
            for client_id, weight in final_weights.items():
                update = client_updates[client_id]
                if 'weight_updates' in update and layer_name in update['weight_updates']:
                    layer_updates.append(update['weight_updates'][layer_name])
                    layer_weights.append(weight)
            
            if layer_updates:
                weighted_update = np.zeros_like(global_weights[layer_name])
                for update, weight in zip(layer_updates, layer_weights):
                    weighted_update += weight * update
                
                aggregated_weights[layer_name] = global_weights[layer_name] + weighted_update
            else:
                aggregated_weights[layer_name] = global_weights[layer_name]
        
        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(
            client_updates, demographic_data, final_weights
        )
        
        metrics = {
            'aggregation_method': 'fairfed',
            'fairness_weight': self.config.fairness_weight,
            'num_valid_clients': len(final_weights),
            'num_byzantine_clients': len(byzantine_clients),
            'client_weights': final_weights,
            'fairness_metrics': fairness_metrics
        }
        
        return {
            'aggregated_weights': aggregated_weights,
            'metrics': metrics
        }
    
    def _calculate_fairness_weights(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        demographic_data: Dict[int, Dict[str, Any]],
        base_weights: Dict[int, float]
    ) -> Dict[int, float]:
        """Calculate fairness-adjusted weights for clients."""
        fairness_weights = {}
        
        # Analyze demographic distribution across clients
        demographic_stats = self._analyze_demographic_distribution(demographic_data)
        
        for client_id in base_weights:
            if client_id in demographic_data:
                client_demo = demographic_data[client_id]
                
                # Calculate fairness score based on demographic representation
                fairness_score = self._calculate_client_fairness_score(
                    client_demo, demographic_stats
                )
                
                # Higher weight for clients that improve fairness
                fairness_weights[client_id] = fairness_score
            else:
                fairness_weights[client_id] = 1.0
        
        # Normalize fairness weights
        total_fairness = sum(fairness_weights.values())
        if total_fairness > 0:
            for client_id in fairness_weights:
                fairness_weights[client_id] /= total_fairness
        
        return fairness_weights
    
    def _analyze_demographic_distribution(
        self,
        demographic_data: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze overall demographic distribution."""
        all_demographics = []
        for client_demo in demographic_data.values():
            all_demographics.extend(client_demo.get('demographics', []))
        
        demo_stats = defaultdict(lambda: defaultdict(int))
        
        for demo in all_demographics:
            for attr, value in demo.items():
                demo_stats[attr][value] += 1
        
        # Convert to proportions
        demo_proportions = {}
        for attr, values in demo_stats.items():
            total = sum(values.values())
            demo_proportions[attr] = {
                value: count / total
                for value, count in values.items()
            }
        
        return demo_proportions
    
    def _calculate_client_fairness_score(
        self,
        client_demo: Dict[str, Any],
        global_demo_stats: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate fairness score for a client."""
        client_demographics = client_demo.get('demographics', [])
        
        if not client_demographics:
            return 1.0
        
        fairness_scores = []
        
        for demo in client_demographics:
            for attr, value in demo.items():
                if attr in global_demo_stats and value in global_demo_stats[attr]:
                    # Lower global proportion means higher fairness weight
                    global_prop = global_demo_stats[attr][value]
                    fairness_scores.append(1.0 / (global_prop + 0.01))  # Avoid division by zero
        
        # Average fairness score
        return np.mean(fairness_scores) if fairness_scores else 1.0
    
    def _calculate_fairness_metrics(
        self,
        client_updates: Dict[int, Dict[str, Any]],
        demographic_data: Optional[Dict[int, Dict[str, Any]]],
        client_weights: Dict[int, float]
    ) -> Dict[str, float]:
        """Calculate fairness metrics for the aggregation round."""
        if not demographic_data:
            return {}
        
        # Calculate demographic parity in client selection
        selected_demographics = []
        for client_id in client_weights:
            if client_id in demographic_data:
                client_demo = demographic_data[client_id].get('demographics', [])
                selected_demographics.extend(client_demo)
        
        if not selected_demographics:
            return {}
        
        # Calculate demographic parity
        demo_counts = defaultdict(lambda: defaultdict(int))
        for demo in selected_demographics:
            for attr, value in demo.items():
                demo_counts[attr][value] += 1
        
        fairness_metrics = {}
        
        for attr, values in demo_counts.items():
            if len(values) > 1:  # Multiple groups exist
                counts = list(values.values())
                max_count = max(counts)
                min_count = min(counts)
                
                # Demographic parity ratio
                fairness_metrics[f'{attr}_parity_ratio'] = min_count / max_count if max_count > 0 else 1.0
                
                # Entropy-based diversity
                total = sum(counts)
                proportions = [c / total for c in counts]
                entropy = -sum(p * np.log2(p + 1e-8) for p in proportions)
                max_entropy = np.log2(len(proportions))
                fairness_metrics[f'{attr}_diversity'] = entropy / max_entropy if max_entropy > 0 else 1.0
        
        return fairness_metrics


def create_aggregator(
    strategy: str,
    config: AggregationConfig,
    **kwargs
) -> FederatedAggregator:
    """
    Factory function to create aggregator.
    
    Args:
        strategy: Aggregation strategy name
        config: Aggregation configuration
        **kwargs: Additional strategy-specific arguments
    
    Returns:
        Configured aggregator instance
    """
    if strategy == "fedavg":
        return FedAvgAggregator(config)
    elif strategy == "fedprox":
        mu = kwargs.get('mu', 0.01)
        return FedProxAggregator(config, mu)
    elif strategy == "fairfed":
        return FairFedAggregator(config)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import torch
import tensorflow as tf
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from scipy import optimize
from collections import deque

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Privacy budget tracking."""
    epsilon: float
    delta: float
    total_spent: float = 0.0
    spent_history: List[float] = None
    
    def __post_init__(self):
        if self.spent_history is None:
            self.spent_history = []
    
    def spend(self, amount: float) -> bool:
        """
        Spend privacy budget.
        
        Args:
            amount: Amount to spend
        
        Returns:
            True if budget allows, False otherwise
        """
        if self.total_spent + amount <= self.epsilon:
            self.total_spent += amount
            self.spent_history.append(amount)
            return True
        return False
    
    def remaining(self) -> float:
        """Get remaining privacy budget."""
        return max(0.0, self.epsilon - self.total_spent)
    
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.total_spent >= self.epsilon


@dataclass
class PrivacyParameters:
    """Privacy parameters for differential privacy."""
    epsilon: float
    delta: float
    sensitivity: float
    noise_multiplier: float
    clipping_threshold: float
    epochs: int = 1
    batch_size: int = 32
    
    def validate(self) -> bool:
        """Validate privacy parameters."""
        return (
            self.epsilon > 0 and
            self.delta >= 0 and 
            self.delta < 1.0 and
            self.sensitivity > 0 and
            self.noise_multiplier > 0 and
            self.clipping_threshold > 0
        )


class NoiseeMechanism(ABC):
    """Abstract base class for differential privacy noise mechanisms."""
    
    @abstractmethod
    def add_noise(self, data: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """
        Add noise to data for differential privacy.
        
        Args:
            data: Input data
            sensitivity: Sensitivity of the query
            epsilon: Privacy parameter
        
        Returns:
            Noisy data
        """
        pass
    
    @abstractmethod
    def calibrate_noise(self, sensitivity: float, epsilon: float, delta: Optional[float] = None) -> float:
        """
        Calibrate noise parameter for given privacy requirements.
        
        Args:
            sensitivity: Sensitivity of the query
            epsilon: Privacy parameter
            delta: Optional delta parameter for approximate DP
        
        Returns:
            Noise parameter
        """
        pass


class GaussianMechanism(NoiseeMechanism):
    """Gaussian noise mechanism for approximate differential privacy."""
    
    def add_noise(self, data: np.ndarray, sensitivity: float, epsilon: float, delta: float = 1e-5) -> np.ndarray:
        """Add Gaussian noise for (ε, δ)-differential privacy."""
        sigma = self.calibrate_noise(sensitivity, epsilon, delta)
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise
    
    def calibrate_noise(self, sensitivity: float, epsilon: float, delta: Optional[float] = None) -> float:
        """Calibrate Gaussian noise standard deviation."""
        if delta is None:
            delta = 1e-5
        
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        # Gaussian mechanism: σ ≥ sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return sigma


class LaplaceMechanism(NoiseeMechanism):
    """Laplace noise mechanism for pure differential privacy."""
    
    def add_noise(self, data: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """Add Laplace noise for ε-differential privacy."""
        scale = self.calibrate_noise(sensitivity, epsilon)
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise
    
    def calibrate_noise(self, sensitivity: float, epsilon: float, delta: Optional[float] = None) -> float:
        """Calibrate Laplace noise scale parameter."""
        # Laplace mechanism: scale = sensitivity / ε
        return sensitivity / epsilon


class ExponentialMechanism(NoiseeMechanism):
    """Exponential mechanism for non-numeric outputs."""
    
    def __init__(self, utility_function: Callable[[Any], float]):
        """
        Initialize exponential mechanism.
        
        Args:
            utility_function: Function to compute utility scores
        """
        self.utility_function = utility_function
    
    def add_noise(self, data: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """Not applicable for exponential mechanism."""
        raise NotImplementedError("Use select_output method instead")
    
    def calibrate_noise(self, sensitivity: float, epsilon: float, delta: Optional[float] = None) -> float:
        """Not applicable for exponential mechanism."""
        return epsilon / (2 * sensitivity)
    
    def select_output(
        self, 
        candidates: List[Any], 
        sensitivity: float, 
        epsilon: float
    ) -> Any:
        """
        Select output using exponential mechanism.
        
        Args:
            candidates: List of candidate outputs
            sensitivity: Sensitivity of utility function
            epsilon: Privacy parameter
        
        Returns:
            Selected output
        """
        # Compute utility scores
        utilities = [self.utility_function(candidate) for candidate in candidates]
        
        # Compute probabilities using exponential mechanism
        max_utility = max(utilities)
        probabilities = []
        
        for utility in utilities:
            # Normalize to prevent overflow
            normalized_utility = utility - max_utility
            probability = np.exp(epsilon * normalized_utility / (2 * sensitivity))
            probabilities.append(probability)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Sample from the distribution
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_idx]


class PrivacyAccountant:
    """Privacy budget accounting using various composition theorems."""
    
    def __init__(self, total_epsilon: float, total_delta: float):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget (epsilon)
            total_delta: Total privacy budget (delta)
        """
        self.total_budget = PrivacyBudget(total_epsilon, total_delta)
        self.spent_entries = deque()
        self.composition_method = "advanced"  # basic, advanced, rdp
    
    def spend_budget(
        self, 
        epsilon: float, 
        delta: float = 0.0, 
        num_queries: int = 1,
        composition_method: Optional[str] = None
    ) -> bool:
        """
        Spend privacy budget with composition tracking.
        
        Args:
            epsilon: Epsilon to spend
            delta: Delta to spend
            num_queries: Number of queries
            composition_method: Composition method override
        
        Returns:
            True if budget allows spending
        """
        method = composition_method or self.composition_method
        
        if method == "basic":
            total_epsilon_cost = epsilon * num_queries
            total_delta_cost = delta * num_queries
        elif method == "advanced":
            # Advanced composition (Dwork et al.)
            total_epsilon_cost, total_delta_cost = self._advanced_composition(
                epsilon, delta, num_queries
            )
        else:
            # Default to basic composition
            total_epsilon_cost = epsilon * num_queries
            total_delta_cost = delta * num_queries
        
        # Check if we can afford this
        if (self.total_budget.total_spent + total_epsilon_cost <= self.total_budget.epsilon and
            total_delta_cost <= self.total_budget.delta):
            
            # Record the expenditure
            self.total_budget.spend(total_epsilon_cost)
            self.spent_entries.append({
                'epsilon': epsilon,
                'delta': delta,
                'num_queries': num_queries,
                'total_epsilon_cost': total_epsilon_cost,
                'total_delta_cost': total_delta_cost,
                'method': method
            })
            
            logger.info(f"Spent privacy: ε={total_epsilon_cost:.4f}, δ={total_delta_cost:.4f}")
            return True
        
        logger.warning(f"Insufficient privacy budget. Requested: ε={total_epsilon_cost:.4f}, Available: ε={self.total_budget.remaining():.4f}")
        return False
    
    def _advanced_composition(self, epsilon: float, delta: float, k: int) -> Tuple[float, float]:
        """
        Advanced composition theorem.
        
        Args:
            epsilon: Per-query epsilon
            delta: Per-query delta
            k: Number of queries
        
        Returns:
            Composed (epsilon, delta)
        """
        if k == 1:
            return epsilon, delta
        
        # Advanced composition: for k (ε, δ)-DP algorithms
        # Result is (ε', δ')-DP where:
        # ε' = √(2k ln(1/δ')) * ε + k * ε * (e^ε - 1)
        # δ' = k * δ + δ_tilde
        
        delta_tilde = 1e-6  # Additional delta for advanced composition
        
        if epsilon <= 1.0:
            # For small epsilon, use the tight composition
            epsilon_composed = np.sqrt(2 * k * np.log(1 / delta_tilde)) * epsilon + k * epsilon * (np.exp(epsilon) - 1)
        else:
            # For large epsilon, fall back to basic composition
            epsilon_composed = k * epsilon
        
        delta_composed = k * delta + delta_tilde
        
        return epsilon_composed, delta_composed
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get current privacy expenditure."""
        return {
            'epsilon_spent': self.total_budget.total_spent,
            'epsilon_remaining': self.total_budget.remaining(),
            'delta_budget': self.total_budget.delta,
            'num_queries': len(self.spent_entries)
        }
    
    def reset_budget(self, epsilon: Optional[float] = None, delta: Optional[float] = None):
        """Reset privacy budget."""
        if epsilon is not None:
            self.total_budget.epsilon = epsilon
        if delta is not None:
            self.total_budget.delta = delta
        
        self.total_budget.total_spent = 0.0
        self.total_budget.spent_history.clear()
        self.spent_entries.clear()
        
        logger.info("Privacy budget reset")


class DPOptimizer:
    """Differentially private optimizer wrapper."""
    
    def __init__(
        self,
        optimizer: Any,
        privacy_params: PrivacyParameters,
        accountant: Optional[PrivacyAccountant] = None
    ):
        """
        Initialize DP optimizer.
        
        Args:
            optimizer: Base optimizer (PyTorch or TensorFlow)
            privacy_params: Privacy parameters
            accountant: Privacy accountant
        """
        self.base_optimizer = optimizer
        self.privacy_params = privacy_params
        self.accountant = accountant
        self.noise_mechanism = GaussianMechanism()
        
        if not privacy_params.validate():
            raise ValueError("Invalid privacy parameters")
    
    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            gradients: List of gradient arrays
        
        Returns:
            Clipped gradients
        """
        clipped_gradients = []
        
        for grad in gradients:
            # Calculate L2 norm
            grad_norm = np.linalg.norm(grad)
            
            # Clip if necessary
            if grad_norm > self.privacy_params.clipping_threshold:
                clipping_factor = self.privacy_params.clipping_threshold / grad_norm
                clipped_grad = grad * clipping_factor
            else:
                clipped_grad = grad.copy()
            
            clipped_gradients.append(clipped_grad)
        
        return clipped_gradients
    
    def add_noise_to_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add noise to gradients for differential privacy.
        
        Args:
            gradients: List of gradient arrays
        
        Returns:
            Noisy gradients
        """
        noisy_gradients = []
        
        for grad in gradients:
            # Add Gaussian noise
            noisy_grad = self.noise_mechanism.add_noise(
                grad,
                sensitivity=self.privacy_params.sensitivity,
                epsilon=self.privacy_params.epsilon,
                delta=self.privacy_params.delta
            )
            noisy_gradients.append(noisy_grad)
        
        return noisy_gradients
    
    def step(self, loss: Any, model_parameters: List[Any]) -> bool:
        """
        Perform one optimization step with differential privacy.
        
        Args:
            loss: Loss value
            model_parameters: Model parameters
        
        Returns:
            True if step was successful
        """
        # Check privacy budget
        if self.accountant and not self.accountant.spend_budget(
            self.privacy_params.epsilon / self.privacy_params.epochs,
            self.privacy_params.delta / self.privacy_params.epochs
        ):
            logger.warning("Insufficient privacy budget for optimization step")
            return False
        
        # This is framework-specific implementation
        # Would need to be implemented for specific frameworks
        return True


class DifferentialPrivacyManager:
    """High-level differential privacy manager for federated learning."""
    
    def __init__(
        self,
        total_epsilon: float = 1.0,
        total_delta: float = 1e-5,
        noise_mechanism: str = "gaussian"
    ):
        """
        Initialize DP manager.
        
        Args:
            total_epsilon: Total privacy budget epsilon
            total_delta: Total privacy budget delta
            noise_mechanism: Type of noise mechanism
        """
        self.accountant = PrivacyAccountant(total_epsilon, total_delta)
        
        if noise_mechanism == "gaussian":
            self.mechanism = GaussianMechanism()
        elif noise_mechanism == "laplace":
            self.mechanism = LaplaceMechanism()
        else:
            raise ValueError(f"Unknown noise mechanism: {noise_mechanism}")
        
        self.privacy_history = []
        
        logger.info(f"DP Manager initialized: ε={total_epsilon}, δ={total_delta}")
    
    def privatize_data(
        self,
        data: np.ndarray,
        sensitivity: float,
        epsilon_fraction: float = 0.1
    ) -> np.ndarray:
        """
        Add noise to data for privacy.
        
        Args:
            data: Input data
            sensitivity: Data sensitivity
            epsilon_fraction: Fraction of total epsilon to use
        
        Returns:
            Privatized data
        """
        epsilon_to_spend = self.accountant.total_budget.epsilon * epsilon_fraction
        
        if not self.accountant.spend_budget(epsilon_to_spend):
            raise ValueError("Insufficient privacy budget")
        
        if isinstance(self.mechanism, GaussianMechanism):
            noisy_data = self.mechanism.add_noise(
                data, sensitivity, epsilon_to_spend, self.accountant.total_budget.delta
            )
        else:
            noisy_data = self.mechanism.add_noise(data, sensitivity, epsilon_to_spend)
        
        self.privacy_history.append({
            'operation': 'data_privatization',
            'epsilon_spent': epsilon_to_spend,
            'sensitivity': sensitivity,
            'data_shape': data.shape
        })
        
        return noisy_data
    
    def privatize_aggregation(
        self,
        aggregated_updates: Dict[str, np.ndarray],
        num_clients: int,
        epsilon_fraction: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Add noise to aggregated model updates.
        
        Args:
            aggregated_updates: Aggregated model updates
            num_clients: Number of participating clients
            epsilon_fraction: Fraction of epsilon to use
        
        Returns:
            Privatized aggregated updates
        """
        epsilon_to_spend = self.accountant.total_budget.epsilon * epsilon_fraction
        
        if not self.accountant.spend_budget(epsilon_to_spend):
            raise ValueError("Insufficient privacy budget")
        
        # Sensitivity is bounded by clipping in federated setting
        sensitivity = 2.0 / num_clients  # Assuming L2 clipping threshold of 1.0
        
        privatized_updates = {}
        
        for layer_name, update in aggregated_updates.items():
            if isinstance(self.mechanism, GaussianMechanism):
                privatized_update = self.mechanism.add_noise(
                    update, sensitivity, epsilon_to_spend, self.accountant.total_budget.delta
                )
            else:
                privatized_update = self.mechanism.add_noise(update, sensitivity, epsilon_to_spend)
            
            privatized_updates[layer_name] = privatized_update
        
        self.privacy_history.append({
            'operation': 'aggregation_privatization',
            'epsilon_spent': epsilon_to_spend,
            'sensitivity': sensitivity,
            'num_clients': num_clients,
            'num_layers': len(aggregated_updates)
        })
        
        logger.info(f"Privatized aggregation with ε={epsilon_to_spend:.4f}")
        return privatized_updates
    
    def create_dp_optimizer(
        self,
        optimizer: Any,
        clipping_threshold: float = 1.0,
        noise_multiplier: float = 1.0,
        epochs: int = 1,
        batch_size: int = 32
    ) -> DPOptimizer:
        """
        Create differentially private optimizer.
        
        Args:
            optimizer: Base optimizer
            clipping_threshold: Gradient clipping threshold
            noise_multiplier: Noise multiplier
            epochs: Number of training epochs
            batch_size: Training batch size
        
        Returns:
            DP optimizer
        """
        # Calculate per-epoch privacy parameters
        epsilon_per_epoch = self.accountant.total_budget.epsilon / epochs
        delta_per_epoch = self.accountant.total_budget.delta / epochs
        
        privacy_params = PrivacyParameters(
            epsilon=epsilon_per_epoch,
            delta=delta_per_epoch,
            sensitivity=clipping_threshold,
            noise_multiplier=noise_multiplier,
            clipping_threshold=clipping_threshold,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return DPOptimizer(optimizer, privacy_params, self.accountant)
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy expenditure report."""
        privacy_spent = self.accountant.get_privacy_spent()
        
        return {
            'privacy_budget': {
                'total_epsilon': self.accountant.total_budget.epsilon,
                'total_delta': self.accountant.total_budget.delta,
                'epsilon_spent': privacy_spent['epsilon_spent'],
                'epsilon_remaining': privacy_spent['epsilon_remaining'],
                'utilization_percentage': (privacy_spent['epsilon_spent'] / self.accountant.total_budget.epsilon) * 100
            },
            'operations_history': self.privacy_history,
            'composition_details': {
                'num_queries': len(self.accountant.spent_entries),
                'composition_method': self.accountant.composition_method
            },
            'recommendations': self._generate_privacy_recommendations()
        }
    
    def _generate_privacy_recommendations(self) -> List[str]:
        """Generate privacy budget usage recommendations."""
        recommendations = []
        
        utilization = self.accountant.total_budget.total_spent / self.accountant.total_budget.epsilon
        
        if utilization > 0.8:
            recommendations.append("Privacy budget utilization is high (>80%). Consider reducing noise or increasing budget.")
        
        if utilization < 0.2:
            recommendations.append("Privacy budget utilization is low (<20%). You may be over-protecting.")
        
        if len(self.privacy_history) > 10:
            recommendations.append("Many privacy operations performed. Consider batching operations to improve utility.")
        
        if not recommendations:
            recommendations.append("Privacy budget usage appears balanced.")
        
        return recommendations
    
    def reset_privacy_state(self):
        """Reset privacy accounting state."""
        self.accountant.reset_budget()
        self.privacy_history.clear()
        logger.info("Privacy state reset")
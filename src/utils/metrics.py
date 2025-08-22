import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import time
import json
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    loss: float = float('inf')
    calibration_error: float = 0.0
    
    # Detailed metrics
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    num_samples: int = 0
    computation_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'loss': self.loss,
            'calibration_error': self.calibration_error,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix.size > 0 else [],
            'class_metrics': self.class_metrics,
            'num_samples': self.num_samples,
            'computation_time': self.computation_time,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create ModelMetrics from dictionary."""
        # Handle numpy arrays
        confusion_matrix = np.array(data.get('confusion_matrix', []))
        
        return cls(
            accuracy=data.get('accuracy', 0.0),
            precision=data.get('precision', 0.0),
            recall=data.get('recall', 0.0),
            f1_score=data.get('f1_score', 0.0),
            auc_roc=data.get('auc_roc', 0.0),
            auc_pr=data.get('auc_pr', 0.0),
            loss=data.get('loss', float('inf')),
            calibration_error=data.get('calibration_error', 0.0),
            confusion_matrix=confusion_matrix,
            class_metrics=data.get('class_metrics', {}),
            num_samples=data.get('num_samples', 0),
            computation_time=data.get('computation_time', 0.0),
            timestamp=data.get('timestamp', time.time())
        )


@dataclass
class FederatedMetrics:
    """Container for federated learning specific metrics."""
    round_number: int = 0
    participating_clients: int = 0
    total_clients: int = 0
    client_participation_rate: float = 0.0
    
    # Aggregation metrics
    aggregation_time: float = 0.0
    communication_overhead: float = 0.0
    model_size_mb: float = 0.0
    
    # Convergence metrics
    weight_divergence: float = 0.0
    gradient_norm: float = 0.0
    convergence_rate: float = 0.0
    
    # Client diversity metrics
    client_data_sizes: List[int] = field(default_factory=list)
    client_performance_variance: float = 0.0
    statistical_heterogeneity: float = 0.0
    
    # Privacy metrics
    privacy_budget_used: float = 0.0
    privacy_budget_remaining: float = 0.0
    noise_level: float = 0.0
    
    # Fairness metrics
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    fairness_violations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'round_number': self.round_number,
            'participating_clients': self.participating_clients,
            'total_clients': self.total_clients,
            'client_participation_rate': self.client_participation_rate,
            'aggregation_time': self.aggregation_time,
            'communication_overhead': self.communication_overhead,
            'model_size_mb': self.model_size_mb,
            'weight_divergence': self.weight_divergence,
            'gradient_norm': self.gradient_norm,
            'convergence_rate': self.convergence_rate,
            'client_data_sizes': self.client_data_sizes,
            'client_performance_variance': self.client_performance_variance,
            'statistical_heterogeneity': self.statistical_heterogeneity,
            'privacy_budget_used': self.privacy_budget_used,
            'privacy_budget_remaining': self.privacy_budget_remaining,
            'noise_level': self.noise_level,
            'demographic_parity': self.demographic_parity,
            'equalized_odds': self.equalized_odds,
            'fairness_violations': self.fairness_violations
        }


class PerformanceTracker:
    """Track model and federated learning performance over time."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.model_metrics_history = deque(maxlen=max_history)
        self.federated_metrics_history = deque(maxlen=max_history)
        self.custom_metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        
    def add_model_metrics(self, metrics: ModelMetrics) -> None:
        """Add model performance metrics."""
        self.model_metrics_history.append(metrics)
        
    def add_federated_metrics(self, metrics: FederatedMetrics) -> None:
        """Add federated learning metrics."""
        self.federated_metrics_history.append(metrics)
        
    def add_custom_metric(self, name: str, value: Union[float, Dict[str, Any]], timestamp: Optional[float] = None) -> None:
        """Add custom metric."""
        entry = {
            'value': value,
            'timestamp': timestamp or time.time()
        }
        self.custom_metrics_history[name].append(entry)
        
    def get_latest_model_metrics(self) -> Optional[ModelMetrics]:
        """Get latest model metrics."""
        return self.model_metrics_history[-1] if self.model_metrics_history else None
    
    def get_latest_federated_metrics(self) -> Optional[FederatedMetrics]:
        """Get latest federated metrics."""
        return self.federated_metrics_history[-1] if self.federated_metrics_history else None
    
    def get_model_metrics_trend(self, metric_name: str, window: int = 10) -> List[float]:
        """Get trend for specific model metric."""
        if not self.model_metrics_history:
            return []
        
        recent_metrics = list(self.model_metrics_history)[-window:]
        return [getattr(m, metric_name, 0.0) for m in recent_metrics]
    
    def get_federated_metrics_trend(self, metric_name: str, window: int = 10) -> List[float]:
        """Get trend for specific federated metric."""
        if not self.federated_metrics_history:
            return []
        
        recent_metrics = list(self.federated_metrics_history)[-window:]
        return [getattr(m, metric_name, 0.0) for m in recent_metrics]
    
    def calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate comprehensive performance summary."""
        summary = {
            'model_performance': self._summarize_model_performance(),
            'federated_performance': self._summarize_federated_performance(),
            'trends': self._calculate_trends(),
            'custom_metrics': self._summarize_custom_metrics()
        }
        
        return summary
    
    def _summarize_model_performance(self) -> Dict[str, Any]:
        """Summarize model performance metrics."""
        if not self.model_metrics_history:
            return {'message': 'No model metrics available'}
        
        metrics = list(self.model_metrics_history)
        
        # Calculate statistics for key metrics
        accuracies = [m.accuracy for m in metrics]
        f1_scores = [m.f1_score for m in metrics]
        losses = [m.loss for m in metrics if m.loss != float('inf')]
        
        return {
            'total_evaluations': len(metrics),
            'accuracy': {
                'latest': accuracies[-1] if accuracies else 0,
                'mean': np.mean(accuracies) if accuracies else 0,
                'std': np.std(accuracies) if accuracies else 0,
                'min': np.min(accuracies) if accuracies else 0,
                'max': np.max(accuracies) if accuracies else 0
            },
            'f1_score': {
                'latest': f1_scores[-1] if f1_scores else 0,
                'mean': np.mean(f1_scores) if f1_scores else 0,
                'std': np.std(f1_scores) if f1_scores else 0
            },
            'loss': {
                'latest': losses[-1] if losses else 0,
                'mean': np.mean(losses) if losses else 0,
                'std': np.std(losses) if losses else 0
            }
        }
    
    def _summarize_federated_performance(self) -> Dict[str, Any]:
        """Summarize federated learning performance."""
        if not self.federated_metrics_history:
            return {'message': 'No federated metrics available'}
        
        metrics = list(self.federated_metrics_history)
        
        participation_rates = [m.client_participation_rate for m in metrics]
        aggregation_times = [m.aggregation_time for m in metrics]
        convergence_rates = [m.convergence_rate for m in metrics]
        
        return {
            'total_rounds': len(metrics),
            'participation_rate': {
                'latest': participation_rates[-1] if participation_rates else 0,
                'mean': np.mean(participation_rates) if participation_rates else 0,
                'std': np.std(participation_rates) if participation_rates else 0
            },
            'aggregation_time': {
                'latest': aggregation_times[-1] if aggregation_times else 0,
                'mean': np.mean(aggregation_times) if aggregation_times else 0,
                'std': np.std(aggregation_times) if aggregation_times else 0
            },
            'convergence_rate': {
                'latest': convergence_rates[-1] if convergence_rates else 0,
                'mean': np.mean(convergence_rates) if convergence_rates else 0
            }
        }
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends."""
        trends = {}
        
        # Model performance trends
        if len(self.model_metrics_history) >= 2:
            recent_accuracy = self.get_model_metrics_trend('accuracy', 5)
            if len(recent_accuracy) >= 2:
                accuracy_trend = 'improving' if recent_accuracy[-1] > recent_accuracy[0] else 'declining'
                trends['accuracy'] = accuracy_trend
        
        # Federated performance trends
        if len(self.federated_metrics_history) >= 2:
            participation_trend = self.get_federated_metrics_trend('client_participation_rate', 5)
            if len(participation_trend) >= 2:
                participation_direction = 'increasing' if participation_trend[-1] > participation_trend[0] else 'decreasing'
                trends['participation'] = participation_direction
        
        return trends
    
    def _summarize_custom_metrics(self) -> Dict[str, Any]:
        """Summarize custom metrics."""
        summary = {}
        
        for metric_name, history in self.custom_metrics_history.items():
            if history:
                values = [entry['value'] for entry in history if isinstance(entry['value'], (int, float))]
                if values:
                    summary[metric_name] = {
                        'count': len(values),
                        'latest': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        
        return summary
    
    def export_metrics(self, filepath: str) -> None:
        """Export all metrics to file."""
        export_data = {
            'model_metrics': [m.to_dict() for m in self.model_metrics_history],
            'federated_metrics': [m.to_dict() for m in self.federated_metrics_history],
            'custom_metrics': {
                name: list(history) 
                for name, history in self.custom_metrics_history.items()
            },
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def import_metrics(self, filepath: str) -> None:
        """Import metrics from file."""
        with open(filepath, 'r') as f:
            import_data = json.load(f)
        
        # Import model metrics
        for metric_data in import_data.get('model_metrics', []):
            metrics = ModelMetrics.from_dict(metric_data)
            self.model_metrics_history.append(metrics)
        
        # Import federated metrics
        for metric_data in import_data.get('federated_metrics', []):
            metrics = FederatedMetrics(**metric_data)
            self.federated_metrics_history.append(metrics)
        
        # Import custom metrics
        for name, history in import_data.get('custom_metrics', {}).items():
            self.custom_metrics_history[name].extend(history)
        
        logger.info(f"Metrics imported from {filepath}")


class MetricsAggregator:
    """Aggregate metrics across multiple clients/models."""
    
    def __init__(self):
        self.client_metrics = {}  # client_id -> PerformanceTracker
    
    def add_client_tracker(self, client_id: str, tracker: PerformanceTracker) -> None:
        """Add performance tracker for a client."""
        self.client_metrics[client_id] = tracker
    
    def aggregate_model_metrics(self, round_number: int) -> ModelMetrics:
        """Aggregate model metrics across all clients."""
        if not self.client_metrics:
            return ModelMetrics()
        
        # Collect latest metrics from all clients
        client_metrics = []
        for client_id, tracker in self.client_metrics.items():
            latest = tracker.get_latest_model_metrics()
            if latest:
                client_metrics.append(latest)
        
        if not client_metrics:
            return ModelMetrics()
        
        # Calculate weighted averages
        total_samples = sum(m.num_samples for m in client_metrics)
        
        if total_samples == 0:
            weights = [1.0 / len(client_metrics)] * len(client_metrics)
        else:
            weights = [m.num_samples / total_samples for m in client_metrics]
        
        # Aggregate metrics
        aggregated = ModelMetrics(
            accuracy=sum(w * m.accuracy for w, m in zip(weights, client_metrics)),
            precision=sum(w * m.precision for w, m in zip(weights, client_metrics)),
            recall=sum(w * m.recall for w, m in zip(weights, client_metrics)),
            f1_score=sum(w * m.f1_score for w, m in zip(weights, client_metrics)),
            auc_roc=sum(w * m.auc_roc for w, m in zip(weights, client_metrics)),
            auc_pr=sum(w * m.auc_pr for w, m in zip(weights, client_metrics)),
            loss=sum(w * m.loss for w, m in zip(weights, client_metrics) if m.loss != float('inf')),
            calibration_error=sum(w * m.calibration_error for w, m in zip(weights, client_metrics)),
            num_samples=total_samples,
            computation_time=sum(m.computation_time for m in client_metrics),
            timestamp=time.time()
        )
        
        return aggregated
    
    def calculate_client_diversity_metrics(self) -> Dict[str, float]:
        """Calculate diversity metrics across clients."""
        if not self.client_metrics:
            return {}
        
        # Collect latest metrics
        accuracies = []
        f1_scores = []
        data_sizes = []
        
        for tracker in self.client_metrics.values():
            latest = tracker.get_latest_model_metrics()
            if latest:
                accuracies.append(latest.accuracy)
                f1_scores.append(latest.f1_score)
                data_sizes.append(latest.num_samples)
        
        if not accuracies:
            return {}
        
        return {
            'accuracy_variance': float(np.var(accuracies)),
            'accuracy_range': float(np.max(accuracies) - np.min(accuracies)),
            'f1_variance': float(np.var(f1_scores)),
            'data_size_variance': float(np.var(data_sizes)),
            'coefficient_of_variation': float(np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0
        }
    
    def get_client_rankings(self, metric: str = 'accuracy') -> List[Tuple[str, float]]:
        """Get client rankings by specified metric."""
        rankings = []
        
        for client_id, tracker in self.client_metrics.items():
            latest = tracker.get_latest_model_metrics()
            if latest:
                value = getattr(latest, metric, 0.0)
                rankings.append((client_id, value))
        
        # Sort by metric value (descending for accuracy, ascending for loss)
        reverse_sort = metric != 'loss'
        rankings.sort(key=lambda x: x[1], reverse=reverse_sort)
        
        return rankings


def calculate_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    compute_time: float = 0.0
) -> ModelMetrics:
    """
    Calculate comprehensive model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        labels: Class labels (optional)
        compute_time: Computation time for metrics
    
    Returns:
        ModelMetrics object with calculated metrics
    """
    start_time = time.time()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC metrics (if probabilities available)
    auc_roc = 0.0
    auc_pr = 0.0
    if y_prob is not None and len(np.unique(y_true)) == 2:  # Binary classification
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            auc_pr = auc(recall_curve, precision_curve)
        except Exception as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
    
    # Calibration error
    calibration_error = 0.0
    if y_prob is not None:
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        except Exception as e:
            logger.warning(f"Could not calculate calibration error: {e}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    class_metrics = {}
    if labels:
        try:
            class_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
            for label in labels:
                if label in class_report:
                    class_metrics[label] = class_report[label]
        except Exception as e:
            logger.warning(f"Could not calculate class metrics: {e}")
    
    # Create metrics object
    metrics = ModelMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        calibration_error=calibration_error,
        confusion_matrix=conf_matrix,
        class_metrics=class_metrics,
        num_samples=len(y_true),
        computation_time=time.time() - start_time + compute_time,
        timestamp=time.time()
    )
    
    return metrics


def calculate_federated_metrics(
    round_number: int,
    participating_clients: int,
    total_clients: int,
    client_data_sizes: List[int],
    client_metrics: List[ModelMetrics],
    aggregation_time: float,
    model_size_bytes: int,
    **kwargs
) -> FederatedMetrics:
    """
    Calculate federated learning specific metrics.
    
    Args:
        round_number: Current round number
        participating_clients: Number of participating clients
        total_clients: Total number of available clients
        client_data_sizes: List of data sizes for each participating client
        client_metrics: List of model metrics from participating clients
        aggregation_time: Time taken for aggregation
        model_size_bytes: Size of model in bytes
        **kwargs: Additional metrics
    
    Returns:
        FederatedMetrics object
    """
    # Basic federated metrics
    participation_rate = participating_clients / max(total_clients, 1)
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # Client diversity metrics
    client_performance_variance = 0.0
    if client_metrics and len(client_metrics) > 1:
        accuracies = [m.accuracy for m in client_metrics]
        client_performance_variance = np.var(accuracies)
    
    # Statistical heterogeneity
    statistical_heterogeneity = 0.0
    if client_data_sizes and len(client_data_sizes) > 1:
        # Use coefficient of variation as a measure of heterogeneity
        mean_size = np.mean(client_data_sizes)
        std_size = np.std(client_data_sizes)
        statistical_heterogeneity = std_size / mean_size if mean_size > 0 else 0
    
    # Extract additional metrics from kwargs
    privacy_budget_used = kwargs.get('privacy_budget_used', 0.0)
    privacy_budget_remaining = kwargs.get('privacy_budget_remaining', 0.0)
    noise_level = kwargs.get('noise_level', 0.0)
    weight_divergence = kwargs.get('weight_divergence', 0.0)
    gradient_norm = kwargs.get('gradient_norm', 0.0)
    convergence_rate = kwargs.get('convergence_rate', 0.0)
    demographic_parity = kwargs.get('demographic_parity', 0.0)
    equalized_odds = kwargs.get('equalized_odds', 0.0)
    fairness_violations = kwargs.get('fairness_violations', 0)
    
    metrics = FederatedMetrics(
        round_number=round_number,
        participating_clients=participating_clients,
        total_clients=total_clients,
        client_participation_rate=participation_rate,
        aggregation_time=aggregation_time,
        model_size_mb=model_size_mb,
        client_data_sizes=client_data_sizes,
        client_performance_variance=client_performance_variance,
        statistical_heterogeneity=statistical_heterogeneity,
        privacy_budget_used=privacy_budget_used,
        privacy_budget_remaining=privacy_budget_remaining,
        noise_level=noise_level,
        weight_divergence=weight_divergence,
        gradient_norm=gradient_norm,
        convergence_rate=convergence_rate,
        demographic_parity=demographic_parity,
        equalized_odds=equalized_odds,
        fairness_violations=fairness_violations
    )
    
    return metrics


def compare_metrics(metrics1: ModelMetrics, metrics2: ModelMetrics) -> Dict[str, float]:
    """
    Compare two sets of model metrics.
    
    Args:
        metrics1: First set of metrics
        metrics2: Second set of metrics
    
    Returns:
        Dictionary of metric differences
    """
    comparison = {
        'accuracy_diff': metrics2.accuracy - metrics1.accuracy,
        'precision_diff': metrics2.precision - metrics1.precision,
        'recall_diff': metrics2.recall - metrics1.recall,
        'f1_diff': metrics2.f1_score - metrics1.f1_score,
        'auc_roc_diff': metrics2.auc_roc - metrics1.auc_roc,
        'loss_diff': metrics2.loss - metrics1.loss,
        'relative_improvement': {
            'accuracy': (metrics2.accuracy - metrics1.accuracy) / max(metrics1.accuracy, 1e-8),
            'f1_score': (metrics2.f1_score - metrics1.f1_score) / max(metrics1.f1_score, 1e-8)
        }
    }
    
    return comparison


def calculate_statistical_significance(
    metrics_a: List[ModelMetrics],
    metrics_b: List[ModelMetrics],
    metric_name: str = 'accuracy'
) -> Dict[str, float]:
    """
    Calculate statistical significance between two groups of metrics.
    
    Args:
        metrics_a: First group of metrics
        metrics_b: Second group of metrics
        metric_name: Name of metric to compare
    
    Returns:
        Statistical test results
    """
    from scipy import stats as scipy_stats
    
    values_a = [getattr(m, metric_name) for m in metrics_a]
    values_b = [getattr(m, metric_name) for m in metrics_b]
    
    if len(values_a) < 2 or len(values_b) < 2:
        return {'error': 'Insufficient data for statistical test'}
    
    # Perform t-test
    t_stat, p_value = scipy_stats.ttest_ind(values_a, values_b)
    
    # Calculate effect size (Cohen's d)
    mean_a, mean_b = np.mean(values_a), np.mean(values_b)
    std_a, std_b = np.std(values_a, ddof=1), np.std(values_b, ddof=1)
    pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                        (len(values_a) + len(values_b) - 2))
    
    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    }
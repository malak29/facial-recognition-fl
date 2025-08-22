import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .metrics import PerformanceTracker, ModelMetrics, FederatedMetrics
from ..bias_mitigation.fairness_metrics import FairnessResult
from config.settings import settings

logger = logging.getLogger(__name__)

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualizer for training progress and model performance."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize training visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_training_progress(
        self,
        performance_tracker: PerformanceTracker,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot training progress over time.
        
        Args:
            performance_tracker: Performance tracker with history
            metrics: List of metrics to plot
            save_path: Path to save the plot
        
        Returns:
            Path to saved plot
        """
        if not performance_tracker.model_metrics_history:
            logger.warning("No training history available for plotting")
            return None
        
        metrics = metrics or ['accuracy', 'loss', 'f1_score', 'precision', 'recall']
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics <= 2:
            axes = axes.flatten()[:n_metrics]
        else:
            axes = axes.flatten()
        
        # Extract data
        model_history = list(performance_tracker.model_metrics_history)
        rounds = range(1, len(model_history) + 1)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            values = [getattr(m, metric) for m in model_history]
            
            axes[i].plot(rounds, values, marker='o', linewidth=2, markersize=6)
            axes[i].set_title(f'{metric.title()} Over Training Rounds', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Training Round')
            axes[i].set_ylabel(metric.title())
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line
            if len(values) > 1:
                z = np.polyfit(rounds, values, 1)
                p = np.poly1d(z)
                axes[i].plot(rounds, p(rounds), "--", alpha=0.8, color='red')
        
        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")
        
        return save_path
    
    def plot_metric_comparison(
        self,
        trackers: Dict[str, PerformanceTracker],
        metric: str = 'accuracy',
        save_path: Optional[str] = None
    ) -> str:
        """
        Compare specific metric across multiple experiments.
        
        Args:
            trackers: Dictionary of experiment name -> tracker
            metric: Metric to compare
            save_path: Path to save plot
        
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (exp_name, tracker) in enumerate(trackers.items()):
            if not tracker.model_metrics_history:
                continue
            
            history = list(tracker.model_metrics_history)
            rounds = range(1, len(history) + 1)
            values = [getattr(m, metric) for m in history]
            
            ax.plot(rounds, values, marker='o', label=exp_name, 
                   linewidth=2, markersize=6, color=self.colors[i % len(self.colors)])
        
        ax.set_title(f'{metric.title()} Comparison Across Experiments', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Round')
        ax.set_ylabel(metric.title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metric comparison plot saved to {save_path}")
        
        return save_path
    
    def plot_performance_distribution(
        self,
        performance_tracker: PerformanceTracker,
        metric: str = 'accuracy',
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot distribution of performance metric.
        
        Args:
            performance_tracker: Performance tracker
            metric: Metric to analyze
            save_path: Path to save plot
        
        Returns:
            Path to saved plot
        """
        if not performance_tracker.model_metrics_history:
            return None
        
        history = list(performance_tracker.model_metrics_history)
        values = [getattr(m, metric) for m in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(f'Distribution of {metric.title()}', fontsize=14, fontweight='bold')
        ax1.set_xlabel(metric.title())
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(values, vert=True)
        ax2.set_title(f'{metric.title()} Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel(metric.title())
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance distribution plot saved to {save_path}")
        
        return save_path


class FairnessVisualizer:
    """Visualizer for fairness and bias metrics."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_fairness_metrics(
        self,
        fairness_results: Dict[str, FairnessResult],
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot fairness metrics overview.
        
        Args:
            fairness_results: Dictionary of fairness results
            save_path: Path to save plot
        
        Returns:
            Path to saved plot
        """
        if not fairness_results:
            logger.warning("No fairness results available for plotting")
            return None
        
        # Extract data
        metrics = []
        scores = []
        parity_achieved = []
        
        for metric_name, result in fairness_results.items():
            if hasattr(result, 'overall_score') and metric_name != 'overall_fairness':
                metrics.append(metric_name.replace('_', ' ').title())
                scores.append(result.overall_score)
                parity_achieved.append(result.parity_achieved)
        
        if not metrics:
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Color bars based on parity achievement
        colors = ['green' if achieved else 'red' for achieved in parity_achieved]
        
        bars = ax.bar(metrics, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add threshold line
        threshold = fairness_results[list(fairness_results.keys())[0]].threshold_used
        ax.axhline(y=threshold, color='blue', linestyle='--', linewidth=2,
                  label=f'Fairness Threshold ({threshold})')
        
        # Customize plot
        ax.set_title('Fairness Metrics Overview', fontsize=16, fontweight='bold')
        ax.set_ylabel('Fairness Score')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fairness metrics plot saved to {save_path}")
        
        return save_path
    
    def plot_demographic_performance(
        self,
        group_metrics: Dict[str, Dict[str, float]],
        metric_name: str = 'accuracy',
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot performance across demographic groups.
        
        Args:
            group_metrics: Nested dict of attribute -> group -> metric
            metric_name: Name of metric to plot
            save_path: Path to save plot
        
        Returns:
            Path to saved plot
        """
        if not group_metrics:
            return None
        
        n_attributes = len(group_metrics)
        fig, axes = plt.subplots(1, n_attributes, figsize=(5 * n_attributes, 6))
        
        if n_attributes == 1:
            axes = [axes]
        
        for i, (attribute, groups) in enumerate(group_metrics.items()):
            group_names = list(groups.keys())
            values = [groups[group].get(metric_name, 0) for group in group_names]
            
            bars = axes[i].bar(group_names, values, alpha=0.7, 
                             color=plt.cm.viridis(np.linspace(0, 1, len(group_names))))
            
            axes[i].set_title(f'{metric_name.title()} by {attribute.title()}', 
                            fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric_name.title())
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Demographic performance plot saved to {save_path}")
        
        return save_path
    
    def plot_bias_heatmap(
        self,
        bias_matrix: np.ndarray,
        group_names: List[str],
        metric_name: str = 'Bias Score',
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot bias as a heatmap.
        
        Args:
            bias_matrix: Matrix of bias scores
            group_names: Names of demographic groups
            metric_name: Name of metric being visualized
            save_path: Path to save plot
        
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(bias_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(group_names)))
        ax.set_yticks(np.arange(len(group_names)))
        ax.set_xticklabels(group_names)
        ax.set_yticklabels(group_names)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name, rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(group_names)):
            for j in range(len(group_names)):
                text = ax.text(j, i, f'{bias_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(f'{metric_name} Heatmap Across Demographic Groups', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bias heatmap saved to {save_path}")
        
        return save_path


class FederatedVisualizer:
    """Visualizer for federated learning specific metrics."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_federated_progress(
        self,
        federated_history: List[FederatedMetrics],
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot federated learning progress.
        
        Args:
            federated_history: List of federated metrics
            save_path: Path to save plot
        
        Returns:
            Path to saved plot
        """
        if not federated_history:
            return None
        
        rounds = [m.round_number for m in federated_history]
        participation_rates = [m.client_participation_rate for m in federated_history]
        aggregation_times = [m.aggregation_time for m in federated_history]
        convergence_rates = [m.convergence_rate for m in federated_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Client participation rate
        axes[0, 0].plot(rounds, participation_rates, marker='o', linewidth=2)
        axes[0, 0].set_title('Client Participation Rate', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Participation Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Aggregation time
        axes[0, 1].plot(rounds, aggregation_times, marker='s', color='orange', linewidth=2)
        axes[0, 1].set_title('Aggregation Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Convergence rate
        axes[1, 0].plot(rounds, convergence_rates, marker='^', color='green', linewidth=2)
        axes[1, 0].set_title('Convergence Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Convergence Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Client data distribution
        all_data_sizes = []
        for m in federated_history:
            all_data_sizes.extend(m.client_data_sizes)
        
        if all_data_sizes:
            axes[1, 1].hist(all_data_sizes, bins=20, alpha=0.7, color='skyblue')
            axes[1, 1].set_title('Client Data Size Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Data Size')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Federated progress plot saved to {save_path}")
        
        return save_path
    
    def plot_client_diversity(
        self,
        client_metrics: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot client diversity metrics.
        
        Args:
            client_metrics: Dictionary of client_id -> metric values
            save_path: Path to save plot
        
        Returns:
            Path to saved plot
        """
        if not client_metrics:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Client performance comparison
        client_ids = list(client_metrics.keys())
        latest_performance = [metrics[-1] if metrics else 0 for metrics in client_metrics.values()]
        
        bars = ax1.bar(range(len(client_ids)), latest_performance, alpha=0.7)
        ax1.set_title('Latest Client Performance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Client')
        ax1.set_ylabel('Performance')
        ax1.set_xticks(range(len(client_ids)))
        ax1.set_xticklabels([f'Client {i}' for i in range(len(client_ids))], rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Performance evolution for each client
        for i, (client_id, metrics) in enumerate(client_metrics.items()):
            if metrics:
                ax2.plot(range(len(metrics)), metrics, marker='o', 
                        label=f'Client {i}', linewidth=2)
        
        ax2.set_title('Client Performance Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Client diversity plot saved to {save_path}")
        
        return save_path
    
    def plot_communication_overhead(
        self,
        communication_stats: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot communication overhead metrics.
        
        Args:
            communication_stats: List of communication statistics
            save_path: Path to save plot
        
        Returns:
            Path to saved plot
        """
        if not communication_stats:
            return None
        
        rounds = list(range(1, len(communication_stats) + 1))
        messages_sent = [stats.get('messages_sent', 0) for stats in communication_stats]
        send_failures = [stats.get('send_failures', 0) for stats in communication_stats]
        avg_response_time = [stats.get('avg_response_time', 0) for stats in communication_stats]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Messages sent per round
        axes[0, 0].bar(rounds, messages_sent, alpha=0.7, color='blue')
        axes[0, 0].set_title('Messages Sent Per Round', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Messages Sent')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Send failures
        axes[0, 1].bar(rounds, send_failures, alpha=0.7, color='red')
        axes[0, 1].set_title('Communication Failures Per Round', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Failures')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Response time trend
        axes[1, 0].plot(rounds, avg_response_time, marker='o', color='green', linewidth=2)
        axes[1, 0].set_title('Average Response Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Response Time (ms)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate
        total_attempts = [sent + failed for sent, failed in zip(messages_sent, send_failures)]
        success_rates = [sent / max(total, 1) for sent, total in zip(messages_sent, total_attempts)]
        
        axes[1, 1].plot(rounds, success_rates, marker='s', color='orange', linewidth=2)
        axes[1, 1].set_title('Communication Success Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Communication overhead plot saved to {save_path}")
        
        return save_path


# Convenience functions for creating plots
def create_performance_plots(
    performance_tracker: PerformanceTracker,
    save_dir: str = "plots",
    plot_types: List[str] = None
) -> Dict[str, str]:
    """
    Create comprehensive performance plots.
    
    Args:
        performance_tracker: Performance tracker with data
        save_dir: Directory to save plots
        plot_types: Types of plots to create
    
    Returns:
        Dictionary mapping plot types to file paths
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    plot_types = plot_types or ['training_progress', 'distribution']
    visualizer = TrainingVisualizer()
    
    plots = {}
    
    if 'training_progress' in plot_types:
        path = Path(save_dir) / "training_progress.png"
        plots['training_progress'] = visualizer.plot_training_progress(
            performance_tracker, save_path=str(path)
        )
    
    if 'distribution' in plot_types:
        path = Path(save_dir) / "performance_distribution.png"
        plots['distribution'] = visualizer.plot_performance_distribution(
            performance_tracker, save_path=str(path)
        )
    
    return plots


def create_fairness_plots(
    fairness_results: Dict[str, FairnessResult],
    save_dir: str = "plots"
) -> str:
    """
    Create fairness visualization plots.
    
    Args:
        fairness_results: Fairness analysis results
        save_dir: Directory to save plots
    
    Returns:
        Path to saved plot
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    visualizer = FairnessVisualizer()
    path = Path(save_dir) / "fairness_metrics.png"
    
    return visualizer.plot_fairness_metrics(fairness_results, str(path))


def create_federated_plots(
    federated_history: List[FederatedMetrics],
    save_dir: str = "plots"
) -> str:
    """
    Create federated learning visualization plots.
    
    Args:
        federated_history: Federated learning metrics history
        save_dir: Directory to save plots
    
    Returns:
        Path to saved plot
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    visualizer = FederatedVisualizer()
    path = Path(save_dir) / "federated_progress.png"
    
    return visualizer.plot_federated_progress(federated_history, str(path))


def create_interactive_dashboard(
    performance_tracker: PerformanceTracker,
    federated_history: List[FederatedMetrics] = None,
    save_path: str = "dashboard.html"
) -> str:
    """
    Create interactive Plotly dashboard.
    
    Args:
        performance_tracker: Performance tracker
        federated_history: Optional federated metrics
        save_path: Path to save HTML dashboard
    
    Returns:
        Path to saved dashboard
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Training Progress', 'Performance Distribution', 
                       'Federated Progress', 'Client Participation'],
        specs=[[{"secondary_y": True}, {"type": "histogram"}],
               [{"secondary_y": True}, {"type": "bar"}]]
    )
    
    # Training progress
    if performance_tracker.model_metrics_history:
        history = list(performance_tracker.model_metrics_history)
        rounds = list(range(1, len(history) + 1))
        accuracies = [m.accuracy for m in history]
        losses = [m.loss for m in history if m.loss != float('inf')]
        
        fig.add_trace(
            go.Scatter(x=rounds, y=accuracies, name='Accuracy', line=dict(color='blue')),
            row=1, col=1
        )
        
        if losses:
            fig.add_trace(
                go.Scatter(x=rounds[:len(losses)], y=losses, name='Loss', 
                          line=dict(color='red'), yaxis='y2'),
                row=1, col=1, secondary_y=True
            )
    
    # Performance distribution
    if performance_tracker.model_metrics_history:
        accuracies = [m.accuracy for m in list(performance_tracker.model_metrics_history)]
        fig.add_trace(
            go.Histogram(x=accuracies, name='Accuracy Distribution', nbinsx=20),
            row=1, col=2
        )
    
    # Federated progress
    if federated_history:
        rounds = [m.round_number for m in federated_history]
        participation = [m.client_participation_rate for m in federated_history]
        
        fig.add_trace(
            go.Scatter(x=rounds, y=participation, name='Client Participation',
                      line=dict(color='green')),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text="Federated Learning Dashboard",
        showlegend=True,
        height=800
    )
    
    # Save dashboard
    fig.write_html(save_path)
    logger.info(f"Interactive dashboard saved to {save_path}")
    
    return save_path
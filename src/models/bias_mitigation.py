import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix
from loguru import logger
import warnings
from collections import defaultdict


@dataclass
class FairnessMetrics:
    """Container for fairness metrics"""
    demographic_parity: float
    equal_opportunity: float
    equalized_odds: float
    disparate_impact: float
    statistical_parity: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'demographic_parity': self.demographic_parity,
            'equal_opportunity': self.equal_opportunity,
            'equalized_odds': self.equalized_odds,
            'disparate_impact': self.disparate_impact,
            'statistical_parity': self.statistical_parity
        }
    
    def is_fair(
        self,
        dp_threshold: float = 0.1,
        eo_threshold: float = 0.1
    ) -> bool:
        """Check if metrics meet fairness thresholds"""
        return (
            self.demographic_parity < dp_threshold and
            self.equal_opportunity < eo_threshold
        )


class FairBatchSampler(torch.utils.data.Sampler):
    """
    Fair batch sampler that ensures balanced representation of sensitive attributes
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        sensitive_attribute: str = 'ethnicity',
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sensitive_attribute = sensitive_attribute
        self.drop_last = drop_last
        
        # Group indices by sensitive attribute
        self.groups = defaultdict(list)
        for idx in range(len(dataset)):
            item = dataset[idx]
            group = item.get(sensitive_attribute, 'unknown')
            self.groups[group].append(idx)
        
        # Calculate samples per group
        self.num_groups = len(self.groups)
        self.samples_per_group = batch_size // self.num_groups
        
    def __iter__(self):
        # Shuffle indices within each group
        group_indices = {}
        for group, indices in self.groups.items():
            shuffled = torch.randperm(len(indices)).tolist()
            group_indices[group] = [indices[i] for i in shuffled]
        
        # Create balanced batches
        batches = []
        min_group_size = min(len(indices) for indices in group_indices.values())
        num_batches = min_group_size // self.samples_per_group
        
        for batch_idx in range(num_batches):
            batch = []
            for group in group_indices:
                start_idx = batch_idx * self.samples_per_group
                end_idx = start_idx + self.samples_per_group
                batch.extend(group_indices[group][start_idx:end_idx])
            
            # Shuffle within batch
            batch = [batch[i] for i in torch.randperm(len(batch)).tolist()]
            batches.extend(batch)
        
        return iter(batches)
    
    def __len__(self):
        min_group_size = min(len(indices) for indices in self.groups.values())
        num_batches = min_group_size // self.samples_per_group
        return num_batches * self.batch_size


class AdversarialDebiasing(nn.Module):
    """
    Adversarial debiasing module to remove bias from embeddings
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_sensitive_classes: int,
        hidden_dim: int = 256,
        adversarial_weight: float = 1.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_sensitive_classes = num_sensitive_classes
        self.adversarial_weight = adversarial_weight
        
        # Adversarial discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_sensitive_classes)
        )
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer(adversarial_weight)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        sensitive_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            embeddings: Feature embeddings [B, D]
            sensitive_labels: Sensitive attribute labels [B]
        
        Returns:
            Dictionary with predictions and adversarial loss
        """
        # Apply gradient reversal
        reversed_embeddings = self.gradient_reversal(embeddings)
        
        # Discriminator predictions
        sensitive_predictions = self.discriminator(reversed_embeddings)
        
        outputs = {
            'sensitive_predictions': sensitive_predictions
        }
        
        # Calculate adversarial loss if labels provided
        if sensitive_labels is not None:
            adv_loss = F.cross_entropy(sensitive_predictions, sensitive_labels)
            outputs['adversarial_loss'] = adv_loss
        
        return outputs


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for adversarial training"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal function"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class BiasMitigator:
    """
    Comprehensive bias mitigation system for facial recognition
    """
    
    def __init__(
        self,
        model: nn.Module,
        sensitive_attributes: List[str],
        mitigation_strategies: List[str] = ['reweighting', 'adversarial'],
        device: str = 'cuda'
    ):
        self.model = model
        self.sensitive_attributes = sensitive_attributes
        self.mitigation_strategies = mitigation_strategies
        self.device = device
        
        # Initialize adversarial debiasing if needed
        if 'adversarial' in mitigation_strategies:
            self.adversarial_debiasing = AdversarialDebiasing(
                embedding_dim=512,  # Should match model output
                num_sensitive_classes=10  # Adjust based on your data
            ).to(device)
        
        # Sample weights for reweighting
        self.sample_weights = {}
    
    def calculate_fairness_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> FairnessMetrics:
        """
        Calculate comprehensive fairness metrics
        
        Args:
            predictions: Model predictions
            labels: True labels
            sensitive_attr: Sensitive attribute values
        
        Returns:
            FairnessMetrics object
        """
        unique_groups = np.unique(sensitive_attr)
        
        # Calculate metrics for each group
        group_metrics = {}
        for group in unique_groups:
            mask = sensitive_attr == group
            group_preds = predictions[mask]
            group_labels = labels[mask]
            
            # True positive rate
            tp = np.sum((group_preds == 1) & (group_labels == 1))
            fn = np.sum((group_preds == 0) & (group_labels == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False positive rate
            fp = np.sum((group_preds == 1) & (group_labels == 0))
            tn = np.sum((group_preds == 0) & (group_labels == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Positive prediction rate
            ppr = np.mean(group_preds)
            
            group_metrics[group] = {
                'tpr': tpr,
                'fpr': fpr,
                'ppr': ppr,
                'size': len(group_preds)
            }
        
        # Calculate fairness metrics
        pprs = [m['ppr'] for m in group_metrics.values()]
        tprs = [m['tpr'] for m in group_metrics.values()]
        fprs = [m['fpr'] for m in group_metrics.values()]
        
        # Demographic parity: max difference in positive prediction rates
        demographic_parity = max(pprs) - min(pprs) if pprs else 0
        
        # Equal opportunity: max difference in true positive rates
        equal_opportunity = max(tprs) - min(tprs) if tprs else 0
        
        # Equalized odds: max of TPR and FPR differences
        equalized_odds = max(
            max(tprs) - min(tprs) if tprs else 0,
            max(fprs) - min(fprs) if fprs else 0
        )
        
        # Disparate impact: ratio of min to max positive prediction rates
        disparate_impact = min(pprs) / max(pprs) if max(pprs) > 0 else 1
        
        # Statistical parity: average deviation from overall positive rate
        overall_ppr = np.mean(predictions)
        statistical_parity = np.mean([abs(ppr - overall_ppr) for ppr in pprs])
        
        metrics = FairnessMetrics(
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            equalized_odds=equalized_odds,
            disparate_impact=disparate_impact,
            statistical_parity=statistical_parity
        )
        
        logger.info(f"Fairness metrics calculated: {metrics.to_dict()}")
        
        return metrics
    
    def compute_sample_weights(
        self,
        labels: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> np.ndarray:
        """
        Compute sample weights for reweighting strategy
        
        Args:
            labels: True labels
            sensitive_attr: Sensitive attribute values
        
        Returns:
            Sample weights array
        """
        weights = np.ones(len(labels))
        
        # Calculate group statistics
        unique_groups = np.unique(sensitive_attr)
        group_stats = {}
        
        for group in unique_groups:
            mask = sensitive_attr == group
            group_size = np.sum(mask)
            positive_rate = np.mean(labels[mask])
            
            group_stats[group] = {
                'size': group_size,
                'positive_rate': positive_rate
            }
        
        # Calculate target rates
        overall_positive_rate = np.mean(labels)
        total_size = len(labels)
        
        # Compute weights to balance positive rates across groups
        for group in unique_groups:
            mask = sensitive_attr == group
            stats = group_stats[group]
            
            # Weight to balance group representation
            size_weight = total_size / (len(unique_groups) * stats['size'])
            
            # Weight to balance positive rates
            if stats['positive_rate'] > 0:
                rate_weight = overall_positive_rate / stats['positive_rate']
            else:
                rate_weight = 1.0
            
            # Combined weight
            weights[mask] = size_weight * rate_weight
        
        # Normalize weights
        weights = weights / np.mean(weights)
        
        return weights
    
    def mitigate_bias(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10
    ) -> Dict[str, List[float]]:
        """
        Apply bias mitigation during training
        
        Args:
            dataloader: Training dataloader
            optimizer: Model optimizer
            num_epochs: Number of training epochs
        
        Returns:
            Training history
        """
        history = {
            'loss': [],
            'accuracy': [],
            'fairness_score': []
        }
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_acc = 0
            epoch_fairness = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                sensitive_attr = batch['sensitive_attributes'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, labels)
                
                # Calculate main task loss
                main_loss = F.cross_entropy(outputs['logits'], labels)
                
                # Apply mitigation strategies
                total_loss = main_loss
                
                if 'adversarial' in self.mitigation_strategies:
                    # Adversarial debiasing
                    adv_outputs = self.adversarial_debiasing(
                        outputs['embeddings'],
                        sensitive_attr
                    )
                    total_loss += adv_outputs['adversarial_loss']
                
                if 'reweighting' in self.mitigation_strategies:
                    # Apply sample weights
                    weights = self.compute_sample_weights(
                        labels.cpu().numpy(),
                        sensitive_attr.cpu().numpy()
                    )
                    weights = torch.tensor(weights, device=self.device)
                    total_loss = (total_loss * weights).mean()
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    preds = torch.argmax(outputs['logits'], dim=1)
                    accuracy = (preds == labels).float().mean()
                    
                    # Calculate fairness metrics
                    fairness_metrics = self.calculate_fairness_metrics(
                        preds.cpu().numpy(),
                        labels.cpu().numpy(),
                        sensitive_attr.cpu().numpy()
                    )
                    
                    epoch_loss += total_loss.item()
                    epoch_acc += accuracy.item()
                    epoch_fairness.append(fairness_metrics.demographic_parity)
                
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"Loss={total_loss.item():.4f}, "
                        f"Acc={accuracy.item():.4f}, "
                        f"DP={fairness_metrics.demographic_parity:.4f}"
                    )
            
            # Record epoch metrics
            history['loss'].append(epoch_loss / len(dataloader))
            history['accuracy'].append(epoch_acc / len(dataloader))
            history['fairness_score'].append(np.mean(epoch_fairness))
        
        return history
    
    def evaluate_fairness(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Comprehensive fairness evaluation
        
        Args:
            dataloader: Evaluation dataloader
        
        Returns:
            Evaluation results
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_sensitive = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                labels = batch['labels']
                sensitive_attr = batch['sensitive_attributes']
                
                outputs = self.model(images)
                preds = torch.argmax(outputs['logits'], dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_sensitive.extend(sensitive_attr.numpy())
                all_embeddings.append(outputs['embeddings'].cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_sensitive = np.array(all_sensitive)
        all_embeddings = np.vstack(all_embeddings)
        
        # Calculate fairness metrics
        fairness_metrics = self.calculate_fairness_metrics(
            all_predictions,
            all_labels,
            all_sensitive
        )
        
        # Calculate per-group accuracy
        unique_groups = np.unique(all_sensitive)
        group_accuracies = {}
        
        for group in unique_groups:
            mask = all_sensitive == group
            group_acc = np.mean(all_predictions[mask] == all_labels[mask])
            group_accuracies[f'group_{group}_accuracy'] = group_acc
        
        # Overall accuracy
        overall_accuracy = np.mean(all_predictions == all_labels)
        
        results = {
            'overall_accuracy': overall_accuracy,
            'fairness_metrics': fairness_metrics.to_dict(),
            'group_accuracies': group_accuracies,
            'embedding_statistics': {
                'mean_norm': np.mean(np.linalg.norm(all_embeddings, axis=1)),
                'std_norm': np.std(np.linalg.norm(all_embeddings, axis=1))
            }
        }
        
        return results
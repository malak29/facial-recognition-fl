import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings

from .fairness_metrics import FairnessMetricsCalculator, FairnessResult
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class BiasDetectionResult:
    """Container for bias detection results."""
    detection_type: str
    bias_detected: bool
    severity_level: str  # low, medium, high, critical
    confidence_score: float
    affected_groups: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]
    statistical_evidence: Dict[str, Any]


class BiasDetectionSuite:
    """Comprehensive bias detection suite."""
    
    def __init__(
        self,
        sensitive_attributes: List[str],
        detection_threshold: float = 0.05,
        severity_thresholds: Dict[str, float] = None
    ):
        """
        Initialize bias detection suite.
        
        Args:
            sensitive_attributes: List of sensitive demographic attributes
            detection_threshold: Statistical significance threshold
            severity_thresholds: Thresholds for bias severity levels
        """
        self.sensitive_attributes = sensitive_attributes
        self.detection_threshold = detection_threshold
        self.severity_thresholds = severity_thresholds or {
            'low': 0.05,
            'medium': 0.15,
            'high': 0.30,
            'critical': 0.50
        }
        
        # Initialize component detectors
        self.statistical_detector = StatisticalBiasDetector(detection_threshold)
        self.model_detector = ModelBiasDetector(detection_threshold)
        self.dataset_analyzer = DatasetBiasAnalyzer(sensitive_attributes)
        
        logger.info(f"BiasDetectionSuite initialized for attributes: {sensitive_attributes}")
    
    def run_comprehensive_detection(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sensitive_data: Dict[str, np.ndarray],
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        sensitive_train: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, BiasDetectionResult]:
        """
        Run comprehensive bias detection across all components.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            sensitive_data: Test sensitive attributes
            X_train: Optional training features
            y_train: Optional training labels
            sensitive_train: Optional training sensitive attributes
        
        Returns:
            Dictionary of bias detection results
        """
        results = {}
        
        # Get model predictions
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        
        # 1. Statistical bias detection
        logger.info("Running statistical bias detection...")
        stat_results = self.statistical_detector.detect_statistical_bias(
            y_test, y_pred, y_prob, sensitive_data
        )
        results.update(stat_results)
        
        # 2. Model performance bias detection
        logger.info("Running model bias detection...")
        model_results = self.model_detector.detect_model_bias(
            model, X_test, y_test, sensitive_data, y_pred, y_prob
        )
        results.update(model_results)
        
        # 3. Dataset bias analysis (if training data available)
        if X_train is not None and y_train is not None and sensitive_train is not None:
            logger.info("Running dataset bias analysis...")
            dataset_results = self.dataset_analyzer.analyze_dataset_bias(
                X_train, y_train, sensitive_train, X_test, y_test, sensitive_data
            )
            results.update(dataset_results)
        
        # 4. Intersectional bias analysis
        logger.info("Running intersectional bias analysis...")
        intersectional_results = self._detect_intersectional_bias(
            y_test, y_pred, y_prob, sensitive_data
        )
        results.update(intersectional_results)
        
        # Generate overall assessment
        results['overall_assessment'] = self._generate_overall_assessment(results)
        
        logger.info(f"Bias detection completed. Found {len([r for r in results.values() if r.bias_detected])} bias issues.")
        
        return results
    
    def _detect_intersectional_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        sensitive_data: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Detect bias across intersections of multiple sensitive attributes."""
        results = {}
        
        if len(self.sensitive_attributes) < 2:
            return results
        
        # Create intersectional groups
        intersections = []
        for i, attr1 in enumerate(self.sensitive_attributes):
            for attr2 in self.sensitive_attributes[i+1:]:
                if attr1 in sensitive_data and attr2 in sensitive_data:
                    intersections.append((attr1, attr2))
        
        for attr1, attr2 in intersections:
            values1 = sensitive_data[attr1]
            values2 = sensitive_data[attr2]
            
            # Create combined groups
            combined_groups = []
            for v1, v2 in zip(values1, values2):
                combined_groups.append(f"{attr1}_{v1}_{attr2}_{v2}")
            
            combined_groups = np.array(combined_groups)
            unique_groups = np.unique(combined_groups)
            
            if len(unique_groups) < 2:
                continue
            
            # Calculate metrics for each intersectional group
            group_metrics = {}
            group_sizes = {}
            
            for group in unique_groups:
                group_mask = combined_groups == group
                if np.sum(group_mask) > 5:  # Minimum group size
                    group_y_true = y_true[group_mask]
                    group_y_pred = y_pred[group_mask]
                    
                    accuracy = accuracy_score(group_y_true, group_y_pred)
                    precision = precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
                    recall = recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
                    
                    group_metrics[group] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'size': np.sum(group_mask)
                    }
                    group_sizes[group] = np.sum(group_mask)
            
            if len(group_metrics) < 2:
                continue
            
            # Detect bias across intersectional groups
            accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
            accuracy_variance = np.var(accuracies)
            min_accuracy = min(accuracies)
            max_accuracy = max(accuracies)
            
            # Statistical test
            try:
                # Test for significant differences in accuracy
                group_predictions = []
                group_labels = []
                
                for group in group_metrics:
                    group_mask = combined_groups == group
                    group_predictions.append(y_pred[group_mask])
                    group_labels.extend([group] * np.sum(group_mask))
                
                if len(group_predictions) > 1:
                    f_stat, p_value = stats.f_oneway(*[
                        y_true[combined_groups == group] == y_pred[combined_groups == group]
                        for group in group_metrics
                    ])
                else:
                    p_value = 1.0
                    
            except:
                p_value = 1.0
            
            # Determine bias severity
            accuracy_gap = max_accuracy - min_accuracy
            bias_detected = p_value < self.detection_threshold or accuracy_gap > 0.1
            
            if accuracy_gap < self.severity_thresholds['low']:
                severity = 'low'
            elif accuracy_gap < self.severity_thresholds['medium']:
                severity = 'medium'
            elif accuracy_gap < self.severity_thresholds['high']:
                severity = 'high'
            else:
                severity = 'critical'
            
            # Identify most affected groups
            sorted_groups = sorted(group_metrics.items(), key=lambda x: x[1]['accuracy'])
            most_affected = [group for group, _ in sorted_groups[:2]]
            
            recommendations = []
            if bias_detected:
                recommendations.extend([
                    f"Investigate intersectional bias between {attr1} and {attr2}",
                    f"Consider targeted data collection for underperforming groups: {most_affected}",
                    "Apply intersectional fairness constraints during training"
                ])
            
            intersection_name = f"intersectional_{attr1}_{attr2}"
            results[intersection_name] = BiasDetectionResult(
                detection_type="intersectional_bias",
                bias_detected=bias_detected,
                severity_level=severity,
                confidence_score=1 - p_value,
                affected_groups=most_affected,
                metrics={
                    'accuracy_variance': accuracy_variance,
                    'accuracy_gap': accuracy_gap,
                    'min_accuracy': min_accuracy,
                    'max_accuracy': max_accuracy
                },
                recommendations=recommendations,
                statistical_evidence={'p_value': p_value, 'group_metrics': group_metrics}
            )
        
        return results
    
    def _generate_overall_assessment(
        self,
        results: Dict[str, BiasDetectionResult]
    ) -> BiasDetectionResult:
        """Generate overall bias assessment."""
        # Count bias detections by severity
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        bias_count = 0
        total_confidence = 0
        all_affected_groups = set()
        all_recommendations = []
        
        for result in results.values():
            if isinstance(result, BiasDetectionResult) and result.bias_detected:
                bias_count += 1
                severity_counts[result.severity_level] += 1
                total_confidence += result.confidence_score
                all_affected_groups.update(result.affected_groups)
                all_recommendations.extend(result.recommendations)
        
        # Determine overall severity
        if severity_counts['critical'] > 0:
            overall_severity = 'critical'
        elif severity_counts['high'] > 0:
            overall_severity = 'high'
        elif severity_counts['medium'] > 0:
            overall_severity = 'medium'
        else:
            overall_severity = 'low'
        
        # Calculate overall metrics
        avg_confidence = total_confidence / max(bias_count, 1)
        bias_detected = bias_count > 0
        
        # Generate prioritized recommendations
        priority_recommendations = list(set(all_recommendations))
        if not priority_recommendations:
            priority_recommendations = ["No significant bias detected. Continue monitoring."]
        
        return BiasDetectionResult(
            detection_type="overall_assessment",
            bias_detected=bias_detected,
            severity_level=overall_severity,
            confidence_score=avg_confidence,
            affected_groups=list(all_affected_groups),
            metrics={
                'total_bias_issues': bias_count,
                'severity_breakdown': severity_counts,
                'bias_types_detected': len(results) - 1  # Excluding this overall assessment
            },
            recommendations=priority_recommendations,
            statistical_evidence={'detailed_results': len(results)}
        )


class StatisticalBiasDetector:
    """Statistical bias detection using hypothesis testing."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def detect_statistical_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        sensitive_data: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Detect statistical bias using various statistical tests."""
        results = {}
        
        for attribute in sensitive_data:
            attribute_values = sensitive_data[attribute]
            
            # Test 1: Chi-square test for independence
            chi2_result = self._chi_square_test(y_pred, attribute_values, attribute)
            results[f"chi2_{attribute}"] = chi2_result
            
            # Test 2: Permutation test for accuracy differences
            perm_result = self._permutation_test(y_true, y_pred, attribute_values, attribute)
            results[f"permutation_{attribute}"] = perm_result
            
            # Test 3: Kolmogorov-Smirnov test for probability distributions
            if y_prob is not None:
                ks_result = self._ks_test(y_prob, attribute_values, attribute)
                results[f"ks_{attribute}"] = ks_result
        
        return results
    
    def _chi_square_test(
        self,
        y_pred: np.ndarray,
        sensitive_values: np.ndarray,
        attribute: str
    ) -> BiasDetectionResult:
        """Chi-square test for independence between predictions and sensitive attributes."""
        try:
            contingency_table = pd.crosstab(sensitive_values, y_pred)
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            bias_detected = p_value < self.significance_level
            confidence = 1 - p_value
            
            # Calculate effect size (Cramer's V)
            n = np.sum(contingency_table.values)
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            if cramers_v < 0.1:
                severity = 'low'
            elif cramers_v < 0.3:
                severity = 'medium'
            elif cramers_v < 0.5:
                severity = 'high'
            else:
                severity = 'critical'
            
            recommendations = []
            if bias_detected:
                recommendations.extend([
                    f"Significant association found between {attribute} and predictions",
                    "Consider demographic parity constraints during training",
                    "Investigate data collection bias"
                ])
            
            return BiasDetectionResult(
                detection_type="chi_square_independence",
                bias_detected=bias_detected,
                severity_level=severity,
                confidence_score=confidence,
                affected_groups=list(contingency_table.index),
                metrics={'chi2_statistic': chi2, 'cramers_v': cramers_v},
                recommendations=recommendations,
                statistical_evidence={'p_value': p_value, 'dof': dof}
            )
            
        except Exception as e:
            logger.error(f"Chi-square test failed for {attribute}: {e}")
            return BiasDetectionResult(
                detection_type="chi_square_independence",
                bias_detected=False,
                severity_level='low',
                confidence_score=0.0,
                affected_groups=[],
                metrics={},
                recommendations=["Chi-square test could not be performed"],
                statistical_evidence={'error': str(e)}
            )
    
    def _permutation_test(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_values: np.ndarray,
        attribute: str,
        n_permutations: int = 1000
    ) -> BiasDetectionResult:
        """Permutation test for accuracy differences between groups."""
        unique_groups = np.unique(sensitive_values)
        
        if len(unique_groups) < 2:
            return BiasDetectionResult(
                detection_type="permutation_test",
                bias_detected=False,
                severity_level='low',
                confidence_score=1.0,
                affected_groups=[],
                metrics={},
                recommendations=["Only one group present"],
                statistical_evidence={}
            )
        
        # Calculate observed accuracy difference
        group_accuracies = []
        for group in unique_groups:
            group_mask = sensitive_values == group
            if np.sum(group_mask) > 0:
                group_acc = accuracy_score(y_true[group_mask], y_pred[group_mask])
                group_accuracies.append(group_acc)
        
        observed_diff = max(group_accuracies) - min(group_accuracies)
        
        # Perform permutation test
        permutation_diffs = []
        combined_data = np.column_stack([y_true, y_pred])
        
        for _ in range(n_permutations):
            # Shuffle the sensitive attribute labels
            permuted_sensitive = np.random.permutation(sensitive_values)
            
            # Calculate accuracy difference for permuted data
            perm_accuracies = []
            for group in unique_groups:
                group_mask = permuted_sensitive == group
                if np.sum(group_mask) > 0:
                    perm_acc = accuracy_score(
                        combined_data[group_mask, 0],  # y_true
                        combined_data[group_mask, 1]   # y_pred
                    )
                    perm_accuracies.append(perm_acc)
            
            if len(perm_accuracies) > 1:
                perm_diff = max(perm_accuracies) - min(perm_accuracies)
                permutation_diffs.append(perm_diff)
        
        # Calculate p-value
        p_value = np.mean(np.array(permutation_diffs) >= observed_diff)
        bias_detected = p_value < self.significance_level
        confidence = 1 - p_value
        
        # Determine severity based on observed difference
        if observed_diff < 0.05:
            severity = 'low'
        elif observed_diff < 0.15:
            severity = 'medium'
        elif observed_diff < 0.30:
            severity = 'high'
        else:
            severity = 'critical'
        
        # Identify most affected group
        group_performance = {}
        for i, group in enumerate(unique_groups):
            if i < len(group_accuracies):
                group_performance[str(group)] = group_accuracies[i]
        
        worst_group = min(group_performance.keys(), key=lambda k: group_performance[k])
        
        recommendations = []
        if bias_detected:
            recommendations.extend([
                f"Significant accuracy difference detected across {attribute} groups",
                f"Focus improvement efforts on {worst_group} group",
                "Consider group-specific model training or post-processing"
            ])
        
        return BiasDetectionResult(
            detection_type="permutation_test",
            bias_detected=bias_detected,
            severity_level=severity,
            confidence_score=confidence,
            affected_groups=[worst_group] if bias_detected else [],
            metrics={
                'observed_difference': observed_diff,
                'group_accuracies': group_performance
            },
            recommendations=recommendations,
            statistical_evidence={
                'p_value': p_value,
                'n_permutations': n_permutations,
                'permutation_diffs_mean': np.mean(permutation_diffs)
            }
        )
    
    def _ks_test(
        self,
        y_prob: np.ndarray,
        sensitive_values: np.ndarray,
        attribute: str
    ) -> BiasDetectionResult:
        """Kolmogorov-Smirnov test for probability distribution differences."""
        unique_groups = np.unique(sensitive_values)
        
        if len(unique_groups) != 2:
            return BiasDetectionResult(
                detection_type="ks_test",
                bias_detected=False,
                severity_level='low',
                confidence_score=1.0,
                affected_groups=[],
                metrics={},
                recommendations=["KS test requires exactly 2 groups"],
                statistical_evidence={}
            )
        
        # Get probability distributions for each group
        group1_probs = y_prob[sensitive_values == unique_groups[0]]
        group2_probs = y_prob[sensitive_values == unique_groups[1]]
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(group1_probs, group2_probs)
        
        bias_detected = p_value < self.significance_level
        confidence = 1 - p_value
        
        # Severity based on KS statistic
        if ks_statistic < 0.1:
            severity = 'low'
        elif ks_statistic < 0.3:
            severity = 'medium'
        elif ks_statistic < 0.5:
            severity = 'high'
        else:
            severity = 'critical'
        
        recommendations = []
        if bias_detected:
            recommendations.extend([
                f"Significant difference in prediction confidence between {attribute} groups",
                "Consider calibration techniques to equalize confidence distributions",
                "Investigate threshold optimization for fair classification"
            ])
        
        return BiasDetectionResult(
            detection_type="ks_test",
            bias_detected=bias_detected,
            severity_level=severity,
            confidence_score=confidence,
            affected_groups=list(map(str, unique_groups)),
            metrics={'ks_statistic': ks_statistic},
            recommendations=recommendations,
            statistical_evidence={'p_value': p_value}
        )


class ModelBiasDetector:
    """Model-specific bias detection."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def detect_model_bias(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sensitive_data: Dict[str, np.ndarray],
        y_pred: Optional[np.ndarray] = None,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, BiasDetectionResult]:
        """Detect bias in model behavior."""
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        if y_prob is None and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        
        results = {}
        
        # Feature importance bias
        if hasattr(model, 'feature_importances_'):
            importance_results = self._analyze_feature_importance_bias(model, sensitive_data)
            results.update(importance_results)
        
        # Prediction confidence bias
        if y_prob is not None:
            confidence_results = self._analyze_confidence_bias(y_prob, sensitive_data)
            results.update(confidence_results)
        
        # Error pattern analysis
        error_results = self._analyze_error_patterns(y_test, y_pred, sensitive_data)
        results.update(error_results)
        
        return results
    
    def _analyze_feature_importance_bias(
        self,
        model: Any,
        sensitive_data: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Analyze if model relies too heavily on sensitive features."""
        results = {}
        
        if not hasattr(model, 'feature_importances_'):
            return results
        
        feature_importance = model.feature_importances_
        
        # This is a simplified analysis - would need feature mapping in practice
        # For now, we'll analyze general importance distribution
        
        mean_importance = np.mean(feature_importance)
        std_importance = np.std(feature_importance)
        max_importance = np.max(feature_importance)
        
        # Check for over-reliance on few features (potential proxy discrimination)
        over_reliance_threshold = mean_importance + 3 * std_importance
        high_importance_features = np.sum(feature_importance > over_reliance_threshold)
        
        bias_detected = high_importance_features < 0.1 * len(feature_importance)  # Less than 10% of features dominate
        
        recommendations = []
        if bias_detected:
            recommendations.extend([
                "Model shows over-reliance on few features",
                "Investigate if high-importance features correlate with sensitive attributes",
                "Consider feature regularization or selection techniques",
                "Apply fairness constraints during training"
            ])
        
        results["feature_importance_bias"] = BiasDetectionResult(
            detection_type="feature_importance",
            bias_detected=bias_detected,
            severity_level='medium' if bias_detected else 'low',
            confidence_score=0.7 if bias_detected else 0.3,
            affected_groups=[],
            metrics={
                'max_importance': max_importance,
                'mean_importance': mean_importance,
                'importance_concentration': high_importance_features / len(feature_importance)
            },
            recommendations=recommendations,
            statistical_evidence={'feature_importances': feature_importance.tolist()}
        )
        
        return results
    
    def _analyze_confidence_bias(
        self,
        y_prob: np.ndarray,
        sensitive_data: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Analyze confidence distribution bias across groups."""
        results = {}
        
        for attribute in sensitive_data:
            attribute_values = sensitive_data[attribute]
            unique_groups = np.unique(attribute_values)
            
            if len(unique_groups) < 2:
                continue
            
            # Calculate confidence statistics for each group
            group_confidences = {}
            confidence_stats = {}
            
            for group in unique_groups:
                group_mask = attribute_values == group
                group_probs = y_prob[group_mask]
                
                if len(group_probs) > 0:
                    # Convert probabilities to confidence (distance from 0.5)
                    confidence = np.abs(group_probs - 0.5) * 2
                    
                    group_confidences[str(group)] = confidence
                    confidence_stats[str(group)] = {
                        'mean_confidence': np.mean(confidence),
                        'std_confidence': np.std(confidence),
                        'median_confidence': np.median(confidence)
                    }
            
            if len(confidence_stats) < 2:
                continue
            
            # Test for significant differences in confidence
            confidence_values = list(group_confidences.values())
            try:
                if len(confidence_values) == 2:
                    statistic, p_value = stats.mannwhitneyu(
                        confidence_values[0], confidence_values[1], alternative='two-sided'
                    )
                else:
                    statistic, p_value = stats.kruskal(*confidence_values)
            except:
                p_value = 1.0
            
            bias_detected = p_value < self.significance_level
            
            # Calculate confidence gap
            mean_confidences = [stats['mean_confidence'] for stats in confidence_stats.values()]
            confidence_gap = max(mean_confidences) - min(mean_confidences)
            
            # Determine severity
            if confidence_gap < 0.1:
                severity = 'low'
            elif confidence_gap < 0.2:
                severity = 'medium'
            elif confidence_gap < 0.4:
                severity = 'high'
            else:
                severity = 'critical'
            
            # Identify groups with low confidence
            low_confidence_groups = [
                group for group, stats in confidence_stats.items()
                if stats['mean_confidence'] < 0.6
            ]
            
            recommendations = []
            if bias_detected:
                recommendations.extend([
                    f"Significant confidence differences detected for {attribute}",
                    "Consider confidence calibration techniques",
                    f"Investigate why certain groups ({low_confidence_groups}) have lower confidence"
                ])
            
            results[f"confidence_bias_{attribute}"] = BiasDetectionResult(
                detection_type="confidence_bias",
                bias_detected=bias_detected,
                severity_level=severity,
                confidence_score=1 - p_value,
                affected_groups=low_confidence_groups,
                metrics={
                    'confidence_gap': confidence_gap,
                    'group_confidence_stats': confidence_stats
                },
                recommendations=recommendations,
                statistical_evidence={'p_value': p_value}
            )
        
        return results
    
    def _analyze_error_patterns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_data: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Analyze error patterns across demographic groups."""
        results = {}
        
        for attribute in sensitive_data:
            attribute_values = sensitive_data[attribute]
            unique_groups = np.unique(attribute_values)
            
            if len(unique_groups) < 2:
                continue
            
            # Calculate error rates for each group
            group_errors = {}
            error_types = {}  # False positives, false negatives
            
            for group in unique_groups:
                group_mask = attribute_values == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                if len(group_y_true) > 0:
                    # Overall error rate
                    error_rate = 1 - accuracy_score(group_y_true, group_y_pred)
                    group_errors[str(group)] = error_rate
                    
                    # Error type analysis
                    fp_rate = np.sum((group_y_true == 0) & (group_y_pred == 1)) / max(np.sum(group_y_true == 0), 1)
                    fn_rate = np.sum((group_y_true == 1) & (group_y_pred == 0)) / max(np.sum(group_y_true == 1), 1)
                    
                    error_types[str(group)] = {
                        'false_positive_rate': fp_rate,
                        'false_negative_rate': fn_rate,
                        'total_error_rate': error_rate
                    }
            
            if len(group_errors) < 2:
                continue
            
            # Test for significant differences in error rates
            error_rates = list(group_errors.values())
            max_error = max(error_rates)
            min_error = min(error_rates)
            error_gap = max_error - min_error
            
            # Statistical test using bootstrap
            n_bootstrap = 1000
            bootstrap_gaps = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_errors = []
                for group in unique_groups:
                    group_mask = attribute_values == group
                    group_indices = np.where(group_mask)[0]
                    
                    if len(group_indices) > 0:
                        bootstrap_indices = np.random.choice(group_indices, size=len(group_indices), replace=True)
                        boot_y_true = y_true[bootstrap_indices]
                        boot_y_pred = y_pred[bootstrap_indices]
                        boot_error = 1 - accuracy_score(boot_y_true, boot_y_pred)
                        bootstrap_errors.append(boot_error)
                
                if len(bootstrap_errors) > 1:
                    boot_gap = max(bootstrap_errors) - min(bootstrap_errors)
                    bootstrap_gaps.append(boot_gap)
            
            # Calculate p-value
            if bootstrap_gaps:
                p_value = np.mean(np.array(bootstrap_gaps) >= error_gap)
            else:
                p_value = 1.0
            
            bias_detected = p_value < self.significance_level or error_gap > 0.1
            
            # Determine severity
            if error_gap < 0.05:
                severity = 'low'
            elif error_gap < 0.15:
                severity = 'medium'
            elif error_gap < 0.30:
                severity = 'high'
            else:
                severity = 'critical'
            
            # Identify most affected group
            worst_group = max(group_errors.keys(), key=lambda k: group_errors[k])
            
            recommendations = []
            if bias_detected:
                recommendations.extend([
                    f"Significant error rate differences detected for {attribute}",
                    f"Focus on improving performance for {worst_group} group",
                    "Consider group-specific training or data augmentation",
                    "Investigate root causes of differential error patterns"
                ])
            
            results[f"error_pattern_{attribute}"] = BiasDetectionResult(
                detection_type="error_pattern",
                bias_detected=bias_detected,
                severity_level=severity,
                confidence_score=1 - p_value,
                affected_groups=[worst_group] if bias_detected else [],
                metrics={
                    'error_gap': error_gap,
                    'group_error_rates': group_errors,
                    'error_type_breakdown': error_types
                },
                recommendations=recommendations,
                statistical_evidence={
                    'p_value': p_value,
                    'bootstrap_confidence_interval': np.percentile(bootstrap_gaps, [2.5, 97.5]) if bootstrap_gaps else []
                }
            )
        
        return results


class DatasetBiasAnalyzer:
    """Analyze bias in training and test datasets."""
    
    def __init__(self, sensitive_attributes: List[str]):
        self.sensitive_attributes = sensitive_attributes
    
    def analyze_dataset_bias(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_train: Dict[str, np.ndarray],
        X_test: np.ndarray,
        y_test: np.ndarray,
        sensitive_test: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Analyze bias in dataset composition and distribution."""
        results = {}
        
        # 1. Representation bias
        repr_results = self._analyze_representation_bias(sensitive_train, sensitive_test)
        results.update(repr_results)
        
        # 2. Label distribution bias
        label_results = self._analyze_label_distribution_bias(y_train, sensitive_train, y_test, sensitive_test)
        results.update(label_results)
        
        # 3. Feature distribution bias
        if X_train is not None and X_test is not None:
            feature_results = self._analyze_feature_distribution_bias(X_train, X_test, sensitive_train, sensitive_test)
            results.update(feature_results)
        
        return results
    
    def _analyze_representation_bias(
        self,
        sensitive_train: Dict[str, np.ndarray],
        sensitive_test: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Analyze demographic representation across train/test splits."""
        results = {}
        
        for attribute in self.sensitive_attributes:
            if attribute not in sensitive_train or attribute not in sensitive_test:
                continue
            
            train_values = sensitive_train[attribute]
            test_values = sensitive_test[attribute]
            
            # Calculate group proportions
            train_groups, train_counts = np.unique(train_values, return_counts=True)
            test_groups, test_counts = np.unique(test_values, return_counts=True)
            
            train_props = {str(group): count / len(train_values) for group, count in zip(train_groups, train_counts)}
            test_props = {str(group): count / len(test_values) for group, count in zip(test_groups, test_counts)}
            
            # Find representation gaps
            all_groups = set(train_props.keys()) | set(test_props.keys())
            representation_gaps = {}
            
            for group in all_groups:
                train_prop = train_props.get(group, 0.0)
                test_prop = test_props.get(group, 0.0)
                representation_gaps[group] = abs(train_prop - test_prop)
            
            max_gap = max(representation_gaps.values()) if representation_gaps else 0.0
            
            # Check for minimum representation
            min_train_prop = min(train_props.values()) if train_props else 0.0
            min_test_prop = min(test_props.values()) if test_props else 0.0
            
            # Bias detection criteria
            bias_detected = (
                max_gap > 0.1 or  # Large representation difference
                min_train_prop < 0.05 or  # Underrepresentation in training
                min_test_prop < 0.05  # Underrepresentation in test
            )
            
            if max_gap < 0.05:
                severity = 'low'
            elif max_gap < 0.15:
                severity = 'medium'
            elif max_gap < 0.30:
                severity = 'high'
            else:
                severity = 'critical'
            
            # Identify underrepresented groups
            underrepresented = [
                group for group in all_groups
                if train_props.get(group, 0) < 0.1 or test_props.get(group, 0) < 0.1
            ]
            
            recommendations = []
            if bias_detected:
                recommendations.extend([
                    f"Representation imbalance detected for {attribute}",
                    f"Underrepresented groups: {underrepresented}",
                    "Consider stratified sampling or data collection",
                    "Apply balanced sampling techniques"
                ])
            
            results[f"representation_bias_{attribute}"] = BiasDetectionResult(
                detection_type="representation_bias",
                bias_detected=bias_detected,
                severity_level=severity,
                confidence_score=0.8 if bias_detected else 0.2,
                affected_groups=underrepresented,
                metrics={
                    'max_representation_gap': max_gap,
                    'train_proportions': train_props,
                    'test_proportions': test_props,
                    'representation_gaps': representation_gaps
                },
                recommendations=recommendations,
                statistical_evidence={}
            )
        
        return results
    
    def _analyze_label_distribution_bias(
        self,
        y_train: np.ndarray,
        sensitive_train: Dict[str, np.ndarray],
        y_test: np.ndarray,
        sensitive_test: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Analyze label distribution bias across demographic groups."""
        results = {}
        
        for attribute in self.sensitive_attributes:
            if attribute not in sensitive_train or attribute not in sensitive_test:
                continue
            
            train_sensitive = sensitive_train[attribute]
            test_sensitive = sensitive_test[attribute]
            
            # Calculate positive rate for each group in train and test
            train_pos_rates = {}
            test_pos_rates = {}
            
            unique_groups = np.unique(np.concatenate([train_sensitive, test_sensitive]))
            
            for group in unique_groups:
                # Training set
                train_group_mask = train_sensitive == group
                if np.sum(train_group_mask) > 0:
                    train_pos_rate = np.mean(y_train[train_group_mask])
                    train_pos_rates[str(group)] = train_pos_rate
                
                # Test set
                test_group_mask = test_sensitive == group
                if np.sum(test_group_mask) > 0:
                    test_pos_rate = np.mean(y_test[test_group_mask])
                    test_pos_rates[str(group)] = test_pos_rate
            
            # Calculate distribution inconsistency
            distribution_gaps = {}
            for group in unique_groups:
                group_str = str(group)
                if group_str in train_pos_rates and group_str in test_pos_rates:
                    gap = abs(train_pos_rates[group_str] - test_pos_rates[group_str])
                    distribution_gaps[group_str] = gap
            
            max_gap = max(distribution_gaps.values()) if distribution_gaps else 0.0
            
            # Check for extreme label imbalance within groups
            train_rates = list(train_pos_rates.values())
            test_rates = list(test_pos_rates.values())
            
            train_imbalance = max(train_rates) - min(train_rates) if len(train_rates) > 1 else 0.0
            test_imbalance = max(test_rates) - min(test_rates) if len(test_rates) > 1 else 0.0
            
            bias_detected = (
                max_gap > 0.1 or  # Large train/test inconsistency
                train_imbalance > 0.3 or  # Large imbalance in training
                test_imbalance > 0.3  # Large imbalance in test
            )
            
            if max(max_gap, train_imbalance, test_imbalance) < 0.1:
                severity = 'low'
            elif max(max_gap, train_imbalance, test_imbalance) < 0.2:
                severity = 'medium'
            elif max(max_gap, train_imbalance, test_imbalance) < 0.4:
                severity = 'high'
            else:
                severity = 'critical'
            
            recommendations = []
            if bias_detected:
                recommendations.extend([
                    f"Label distribution bias detected for {attribute}",
                    "Consider balanced sampling strategies",
                    "Apply class weighting or resampling techniques",
                    "Investigate data collection methodology"
                ])
            
            results[f"label_distribution_bias_{attribute}"] = BiasDetectionResult(
                detection_type="label_distribution_bias",
                bias_detected=bias_detected,
                severity_level=severity,
                confidence_score=0.7 if bias_detected else 0.3,
                affected_groups=list(unique_groups.astype(str)),
                metrics={
                    'train_positive_rates': train_pos_rates,
                    'test_positive_rates': test_pos_rates,
                    'distribution_gaps': distribution_gaps,
                    'train_imbalance': train_imbalance,
                    'test_imbalance': test_imbalance
                },
                recommendations=recommendations,
                statistical_evidence={}
            )
        
        return results
    
    def _analyze_feature_distribution_bias(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        sensitive_train: Dict[str, np.ndarray],
        sensitive_test: Dict[str, np.ndarray]
    ) -> Dict[str, BiasDetectionResult]:
        """Analyze feature distribution differences across demographic groups."""
        results = {}
        
        # This is a simplified analysis - in practice would need more sophisticated feature analysis
        
        # Overall distribution shift
        try:
            # Calculate distribution differences using simple statistics
            train_mean = np.mean(X_train, axis=0)
            test_mean = np.mean(X_test, axis=0)
            train_std = np.std(X_train, axis=0)
            test_std = np.std(X_test, axis=0)
            
            # Standardized mean difference
            mean_diff = np.abs(train_mean - test_mean) / (train_std + 1e-8)
            max_mean_diff = np.max(mean_diff)
            avg_mean_diff = np.mean(mean_diff)
            
            # Variance ratio
            var_ratio = np.maximum(train_std, test_std) / (np.minimum(train_std, test_std) + 1e-8)
            max_var_ratio = np.max(var_ratio)
            
            bias_detected = max_mean_diff > 2.0 or max_var_ratio > 3.0  # Conservative thresholds
            
            if max(max_mean_diff, max_var_ratio) < 1.5:
                severity = 'low'
            elif max(max_mean_diff, max_var_ratio) < 2.5:
                severity = 'medium'
            elif max(max_mean_diff, max_var_ratio) < 4.0:
                severity = 'high'
            else:
                severity = 'critical'
            
            recommendations = []
            if bias_detected:
                recommendations.extend([
                    "Feature distribution shift detected between train and test sets",
                    "Consider domain adaptation techniques",
                    "Apply feature normalization or standardization",
                    "Investigate data collection differences"
                ])
            
            results["feature_distribution_bias"] = BiasDetectionResult(
                detection_type="feature_distribution_bias",
                bias_detected=bias_detected,
                severity_level=severity,
                confidence_score=0.6 if bias_detected else 0.4,
                affected_groups=[],
                metrics={
                    'max_mean_difference': max_mean_diff,
                    'avg_mean_difference': avg_mean_diff,
                    'max_variance_ratio': max_var_ratio
                },
                recommendations=recommendations,
                statistical_evidence={}
            )
            
        except Exception as e:
            logger.error(f"Feature distribution analysis failed: {e}")
        
        return results
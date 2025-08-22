import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FairnessResult:
    """Container for fairness metric results."""
    metric_name: str
    overall_score: float
    group_scores: Dict[str, float]
    parity_achieved: bool
    threshold_used: float
    interpretation: str


class FairnessMetricsCalculator:
    """Comprehensive fairness metrics calculator."""
    
    def __init__(
        self,
        sensitive_attributes: List[str],
        fairness_threshold: float = 0.8,
        statistical_significance: float = 0.05
    ):
        """
        Initialize fairness metrics calculator.
        
        Args:
            sensitive_attributes: List of sensitive demographic attributes
            fairness_threshold: Minimum fairness ratio for parity
            statistical_significance: Significance level for statistical tests
        """
        self.sensitive_attributes = sensitive_attributes
        self.fairness_threshold = fairness_threshold
        self.statistical_significance = statistical_significance
        
        # Initialize metric calculators
        self.demographic_parity = DemographicParity(fairness_threshold)
        self.equalized_odds = EqualizedOdds(fairness_threshold)
        self.calibration_metrics = CalibrationMetrics(fairness_threshold)
        
        logger.info(f"FairnessMetricsCalculator initialized for attributes: {sensitive_attributes}")
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        sensitive_data: Dict[str, np.ndarray]
    ) -> Dict[str, FairnessResult]:
        """
        Calculate all fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            sensitive_data: Dictionary mapping attribute names to values
        
        Returns:
            Dictionary of fairness results
        """
        results = {}
        
        for attribute in self.sensitive_attributes:
            if attribute not in sensitive_data:
                logger.warning(f"Sensitive attribute {attribute} not found in data")
                continue
            
            attr_values = sensitive_data[attribute]
            
            # Demographic Parity
            dp_result = self.demographic_parity.calculate(
                y_true, y_pred, attr_values, attribute
            )
            results[f"demographic_parity_{attribute}"] = dp_result
            
            # Equalized Odds
            eo_result = self.equalized_odds.calculate(
                y_true, y_pred, attr_values, attribute
            )
            results[f"equalized_odds_{attribute}"] = eo_result
            
            # Calibration metrics (if probabilities available)
            if y_prob is not None:
                cal_result = self.calibration_metrics.calculate(
                    y_true, y_prob, attr_values, attribute
                )
                results[f"calibration_{attribute}"] = cal_result
            
            # Individual fairness (counterfactual)
            if_result = self._calculate_individual_fairness(
                y_true, y_pred, attr_values, attribute
            )
            results[f"individual_fairness_{attribute}"] = if_result
        
        # Overall fairness summary
        results["overall_fairness"] = self._calculate_overall_fairness(results)
        
        return results
    
    def _calculate_individual_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_values: np.ndarray,
        attribute: str
    ) -> FairnessResult:
        """Calculate individual fairness metrics."""
        unique_groups = np.unique(sensitive_values)
        
        if len(unique_groups) < 2:
            return FairnessResult(
                metric_name="individual_fairness",
                overall_score=1.0,
                group_scores={},
                parity_achieved=True,
                threshold_used=self.fairness_threshold,
                interpretation="Only one group present"
            )
        
        # Calculate prediction consistency for similar individuals
        group_predictions = {}
        group_accuracies = {}
        
        for group in unique_groups:
            group_mask = sensitive_values == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(group_y_pred) > 0:
                accuracy = np.mean(group_y_true == group_y_pred)
                group_accuracies[str(group)] = accuracy
                group_predictions[str(group)] = np.mean(group_y_pred)
        
        # Calculate fairness score as minimum pairwise ratio
        accuracy_values = list(group_accuracies.values())
        if len(accuracy_values) > 1:
            min_accuracy = min(accuracy_values)
            max_accuracy = max(accuracy_values)
            fairness_score = min_accuracy / max_accuracy if max_accuracy > 0 else 0.0
        else:
            fairness_score = 1.0
        
        parity_achieved = fairness_score >= self.fairness_threshold
        
        interpretation = (
            f"Individual fairness score: {fairness_score:.3f}. "
            f"{'Achieves' if parity_achieved else 'Fails to achieve'} fairness threshold."
        )
        
        return FairnessResult(
            metric_name="individual_fairness",
            overall_score=fairness_score,
            group_scores=group_accuracies,
            parity_achieved=parity_achieved,
            threshold_used=self.fairness_threshold,
            interpretation=interpretation
        )
    
    def _calculate_overall_fairness(
        self,
        individual_results: Dict[str, FairnessResult]
    ) -> FairnessResult:
        """Calculate overall fairness summary."""
        fairness_scores = []
        parity_count = 0
        total_metrics = 0
        
        for metric_name, result in individual_results.items():
            if isinstance(result, FairnessResult):
                fairness_scores.append(result.overall_score)
                if result.parity_achieved:
                    parity_count += 1
                total_metrics += 1
        
        overall_score = np.mean(fairness_scores) if fairness_scores else 0.0
        parity_percentage = (parity_count / total_metrics) if total_metrics > 0 else 0.0
        
        interpretation = (
            f"Overall fairness score: {overall_score:.3f}. "
            f"{parity_percentage:.1%} of metrics achieve fairness parity. "
            f"{'Good' if overall_score >= self.fairness_threshold else 'Poor'} overall fairness."
        )
        
        return FairnessResult(
            metric_name="overall_fairness",
            overall_score=overall_score,
            group_scores={"parity_percentage": parity_percentage},
            parity_achieved=overall_score >= self.fairness_threshold,
            threshold_used=self.fairness_threshold,
            interpretation=interpretation
        )
    
    def generate_fairness_report(
        self,
        results: Dict[str, FairnessResult],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive fairness report."""
        report = ["FAIRNESS ASSESSMENT REPORT", "=" * 50, ""]
        
        # Overall summary
        if "overall_fairness" in results:
            overall = results["overall_fairness"]
            report.extend([
                "OVERALL SUMMARY:",
                f"Overall Fairness Score: {overall.overall_score:.3f}",
                f"Threshold Used: {overall.threshold_used}",
                f"Parity Status: {'ACHIEVED' if overall.parity_achieved else 'NOT ACHIEVED'}",
                f"Interpretation: {overall.interpretation}",
                ""
            ])
        
        # Detailed metrics
        report.append("DETAILED METRICS:")
        report.append("-" * 30)
        
        for metric_name, result in results.items():
            if metric_name != "overall_fairness":
                report.extend([
                    f"{metric_name.upper()}:",
                    f"  Score: {result.overall_score:.3f}",
                    f"  Parity: {'✓' if result.parity_achieved else '✗'}",
                    f"  Group Scores: {result.group_scores}",
                    f"  Interpretation: {result.interpretation}",
                    ""
                ])
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Fairness report saved to {save_path}")
        
        return report_text
    
    def create_fairness_visualizations(
        self,
        results: Dict[str, FairnessResult],
        save_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Create fairness visualization plots."""
        plots_created = {}
        
        # Fairness scores bar chart
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        metric_names = []
        scores = []
        colors = []
        
        for metric_name, result in results.items():
            if metric_name != "overall_fairness":
                metric_names.append(metric_name.replace("_", " ").title())
                scores.append(result.overall_score)
                colors.append('green' if result.parity_achieved else 'red')
        
        bars = ax.bar(metric_names, scores, color=colors, alpha=0.7)
        ax.axhline(y=self.fairness_threshold, color='blue', linestyle='--', 
                  label=f'Fairness Threshold ({self.fairness_threshold})')
        ax.set_ylabel('Fairness Score')
        ax.set_title('Fairness Metrics Overview')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        if save_dir:
            plot_path = f"{save_dir}/fairness_overview.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots_created['overview'] = plot_path
        
        plt.close()
        
        # Group-wise comparison plots
        for metric_name, result in results.items():
            if metric_name != "overall_fairness" and result.group_scores:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                groups = list(result.group_scores.keys())
                values = list(result.group_scores.values())
                
                bars = ax.bar(groups, values, alpha=0.7)
                ax.set_ylabel('Score')
                ax.set_title(f'{metric_name.replace("_", " ").title()} by Group')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                if save_dir:
                    plot_path = f"{save_dir}/{metric_name}_by_group.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plots_created[metric_name] = plot_path
                
                plt.close()
        
        return plots_created


class DemographicParity:
    """Demographic parity (statistical parity) calculator."""
    
    def __init__(self, fairness_threshold: float = 0.8):
        self.fairness_threshold = fairness_threshold
    
    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_values: np.ndarray,
        attribute_name: str
    ) -> FairnessResult:
        """Calculate demographic parity."""
        unique_groups = np.unique(sensitive_values)
        
        if len(unique_groups) < 2:
            return FairnessResult(
                metric_name="demographic_parity",
                overall_score=1.0,
                group_scores={},
                parity_achieved=True,
                threshold_used=self.fairness_threshold,
                interpretation="Only one group present"
            )
        
        # Calculate positive prediction rates for each group
        group_rates = {}
        for group in unique_groups:
            group_mask = sensitive_values == group
            group_predictions = y_pred[group_mask]
            
            if len(group_predictions) > 0:
                positive_rate = np.mean(group_predictions == 1)
                group_rates[str(group)] = positive_rate
        
        # Calculate parity score as minimum ratio between groups
        rates = list(group_rates.values())
        if len(rates) > 1:
            min_rate = min(rates)
            max_rate = max(rates)
            parity_score = min_rate / max_rate if max_rate > 0 else 0.0
        else:
            parity_score = 1.0
        
        parity_achieved = parity_score >= self.fairness_threshold
        
        # Statistical significance test
        p_value = self._statistical_test(y_pred, sensitive_values)
        
        interpretation = (
            f"Demographic parity score: {parity_score:.3f}. "
            f"{'Achieves' if parity_achieved else 'Fails to achieve'} parity threshold. "
            f"Statistical test p-value: {p_value:.4f}"
        )
        
        return FairnessResult(
            metric_name="demographic_parity",
            overall_score=parity_score,
            group_scores=group_rates,
            parity_achieved=parity_achieved,
            threshold_used=self.fairness_threshold,
            interpretation=interpretation
        )
    
    def _statistical_test(
        self,
        y_pred: np.ndarray,
        sensitive_values: np.ndarray
    ) -> float:
        """Chi-square test for independence."""
        try:
            contingency_table = pd.crosstab(sensitive_values, y_pred)
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            return p_value
        except:
            return 1.0  # No significant difference


class EqualizedOdds:
    """Equalized odds (equal opportunity) calculator."""
    
    def __init__(self, fairness_threshold: float = 0.8):
        self.fairness_threshold = fairness_threshold
    
    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_values: np.ndarray,
        attribute_name: str
    ) -> FairnessResult:
        """Calculate equalized odds."""
        unique_groups = np.unique(sensitive_values)
        
        if len(unique_groups) < 2:
            return FairnessResult(
                metric_name="equalized_odds",
                overall_score=1.0,
                group_scores={},
                parity_achieved=True,
                threshold_used=self.fairness_threshold,
                interpretation="Only one group present"
            )
        
        # Calculate TPR and FPR for each group
        group_metrics = {}
        tpr_scores = []
        fpr_scores = []
        
        for group in unique_groups:
            group_mask = sensitive_values == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(group_y_true) > 0:
                # Calculate confusion matrix components
                tn, fp, fn, tp = confusion_matrix(
                    group_y_true, group_y_pred, labels=[0, 1]
                ).ravel()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                
                group_metrics[str(group)] = {'tpr': tpr, 'fpr': fpr}
                tpr_scores.append(tpr)
                fpr_scores.append(fpr)
        
        # Calculate equalized odds score
        if len(tpr_scores) > 1 and len(fpr_scores) > 1:
            tpr_parity = min(tpr_scores) / max(tpr_scores) if max(tpr_scores) > 0 else 0.0
            fpr_parity = min(fpr_scores) / max(fpr_scores) if max(fpr_scores) > 0 else 1.0
            
            # Equalized odds requires both TPR and FPR parity
            equalized_odds_score = min(tpr_parity, 1 - abs(max(fpr_scores) - min(fpr_scores)))
        else:
            equalized_odds_score = 1.0
        
        parity_achieved = equalized_odds_score >= self.fairness_threshold
        
        interpretation = (
            f"Equalized odds score: {equalized_odds_score:.3f}. "
            f"{'Achieves' if parity_achieved else 'Fails to achieve'} equalized odds. "
            f"TPR parity: {tpr_parity:.3f}, FPR difference: {abs(max(fpr_scores) - min(fpr_scores)):.3f}"
        )
        
        return FairnessResult(
            metric_name="equalized_odds",
            overall_score=equalized_odds_score,
            group_scores=group_metrics,
            parity_achieved=parity_achieved,
            threshold_used=self.fairness_threshold,
            interpretation=interpretation
        )


class CalibrationMetrics:
    """Calibration and predictive parity calculator."""
    
    def __init__(self, fairness_threshold: float = 0.8):
        self.fairness_threshold = fairness_threshold
    
    def calculate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        sensitive_values: np.ndarray,
        attribute_name: str
    ) -> FairnessResult:
        """Calculate calibration metrics."""
        unique_groups = np.unique(sensitive_values)
        
        if len(unique_groups) < 2:
            return FairnessResult(
                metric_name="calibration",
                overall_score=1.0,
                group_scores={},
                parity_achieved=True,
                threshold_used=self.fairness_threshold,
                interpretation="Only one group present"
            )
        
        # Calculate calibration for each group
        group_calibration = {}
        calibration_errors = []
        
        for group in unique_groups:
            group_mask = sensitive_values == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            if len(group_y_true) > 10:  # Need sufficient samples for calibration
                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        group_y_true, group_y_prob, n_bins=5
                    )
                    
                    # Calculate Expected Calibration Error (ECE)
                    bin_boundaries = np.linspace(0, 1, 6)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    ece = 0
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        in_bin = (group_y_prob > bin_lower) & (group_y_prob <= bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = group_y_true[in_bin].mean()
                            avg_confidence_in_bin = group_y_prob[in_bin].mean()
                            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    group_calibration[str(group)] = 1 - ece  # Convert to calibration score
                    calibration_errors.append(ece)
                    
                except:
                    group_calibration[str(group)] = 0.5
                    calibration_errors.append(0.5)
            else:
                group_calibration[str(group)] = 0.5
                calibration_errors.append(0.5)
        
        # Calculate overall calibration parity
        if len(calibration_errors) > 1:
            calibration_scores = list(group_calibration.values())
            min_calibration = min(calibration_scores)
            max_calibration = max(calibration_scores)
            calibration_parity = min_calibration / max_calibration if max_calibration > 0 else 0.0
        else:
            calibration_parity = 1.0
        
        parity_achieved = calibration_parity >= self.fairness_threshold
        
        interpretation = (
            f"Calibration parity score: {calibration_parity:.3f}. "
            f"{'Good' if parity_achieved else 'Poor'} calibration across groups. "
            f"Mean calibration error: {np.mean(calibration_errors):.3f}"
        )
        
        return FairnessResult(
            metric_name="calibration",
            overall_score=calibration_parity,
            group_scores=group_calibration,
            parity_achieved=parity_achieved,
            threshold_used=self.fairness_threshold,
            interpretation=interpretation
        )


class BiasDetector:
    """High-level bias detection coordinator."""
    
    def __init__(
        self,
        sensitive_attributes: List[str],
        fairness_threshold: float = 0.8,
        enable_statistical_tests: bool = True
    ):
        self.metrics_calculator = FairnessMetricsCalculator(
            sensitive_attributes, fairness_threshold
        )
        self.enable_statistical_tests = enable_statistical_tests
        
    def detect_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        sensitive_data: Dict[str, np.ndarray],
        generate_report: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive bias detection.
        
        Returns:
            Comprehensive bias detection results
        """
        # Calculate fairness metrics
        fairness_results = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, y_prob, sensitive_data
        )
        
        # Generate report and visualizations
        report = None
        visualizations = {}
        
        if generate_report:
            report = self.metrics_calculator.generate_fairness_report(
                fairness_results,
                save_path=f"{save_dir}/fairness_report.txt" if save_dir else None
            )
            
            visualizations = self.metrics_calculator.create_fairness_visualizations(
                fairness_results, save_dir
            )
        
        # Overall bias assessment
        overall_result = fairness_results.get("overall_fairness")
        bias_detected = not overall_result.parity_achieved if overall_result else True
        
        return {
            'bias_detected': bias_detected,
            'overall_fairness_score': overall_result.overall_score if overall_result else 0.0,
            'fairness_results': fairness_results,
            'report': report,
            'visualizations': visualizations,
            'recommendations': self._generate_recommendations(fairness_results)
        }
    
    def _generate_recommendations(
        self,
        fairness_results: Dict[str, FairnessResult]
    ) -> List[str]:
        """Generate bias mitigation recommendations."""
        recommendations = []
        
        for metric_name, result in fairness_results.items():
            if not result.parity_achieved and metric_name != "overall_fairness":
                if "demographic_parity" in metric_name:
                    recommendations.append(
                        f"Consider data balancing or re-weighting for {metric_name.split('_')[-1]} "
                        "to improve demographic parity"
                    )
                elif "equalized_odds" in metric_name:
                    recommendations.append(
                        f"Apply post-processing threshold adjustment for {metric_name.split('_')[-1]} "
                        "to achieve equalized odds"
                    )
                elif "calibration" in metric_name:
                    recommendations.append(
                        f"Consider calibration techniques for {metric_name.split('_')[-1]} "
                        "to improve predictive parity"
                    )
        
        if not recommendations:
            recommendations.append("Model shows good fairness across all measured metrics.")
        
        return recommendations
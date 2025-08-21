import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from config.settings import settings

logger = logging.getLogger(__name__)


class FairAugmentation:
    """Bias-aware data augmentation for facial recognition."""
    
    def __init__(
        self,
        augment_probability: float = 0.8,
        preserve_demographics: bool = True,
        intensity: str = "medium"  # light, medium, heavy
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            augment_probability: Probability of applying augmentation
            preserve_demographics: Whether to preserve demographic features
            intensity: Augmentation intensity level
        """
        self.augment_probability = augment_probability
        self.preserve_demographics = preserve_demographics
        self.intensity = intensity
        
        # Define augmentation pipelines by intensity
        self.pipelines = {
            "light": self._create_light_pipeline(),
            "medium": self._create_medium_pipeline(),
            "heavy": self._create_heavy_pipeline()
        }
        
        # Demographic-aware augmentation settings
        self.demographic_safe_augs = self._create_demographic_safe_pipeline()
        
        logger.info(f"Initialized FairAugmentation with intensity: {intensity}")
    
    def _create_light_pipeline(self) -> A.Compose:
        """Create light augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _create_medium_pipeline(self) -> A.Compose:
        """Create medium augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.MotionBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _create_heavy_pipeline(self) -> A.Compose:
        """Create heavy augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0)),
                A.MotionBlur(blur_limit=7),
                A.GaussianBlur(blur_limit=7),
                A.MedianBlur(blur_limit=5),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.3),
                A.ElasticTransform(p=0.3),
            ], p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _create_demographic_safe_pipeline(self) -> A.Compose:
        """Create augmentations that preserve demographic features."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
            A.RandomGamma(gamma_limit=(90, 110), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def augment_image(
        self, 
        image: np.ndarray, 
        demographic_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Apply augmentation to a single image.
        
        Args:
            image: Input image
            demographic_info: Optional demographic information
        
        Returns:
            Augmented image
        """
        if random.random() > self.augment_probability:
            # Convert to tensor without augmentation
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            return transform(image=image)["image"]
        
        # Choose appropriate pipeline
        if self.preserve_demographics and demographic_info:
            pipeline = self.demographic_safe_augs
        else:
            pipeline = self.pipelines[self.intensity]
        
        try:
            augmented = pipeline(image=image)
            return augmented["image"]
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            # Fallback to basic transform
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            return transform(image=image)["image"]
    
    def create_balanced_augmentation_strategy(
        self,
        demographic_distribution: Dict[str, Dict[str, int]]
    ) -> Dict[str, Callable]:
        """
        Create demographic-specific augmentation strategies.
        
        Args:
            demographic_distribution: Distribution of demographics in dataset
        
        Returns:
            Dictionary mapping demographic groups to augmentation functions
        """
        strategies = {}
        
        # Find underrepresented groups
        total_samples = sum(
            sum(group_counts.values()) 
            for group_counts in demographic_distribution.values()
        )
        
        for attr, groups in demographic_distribution.items():
            for group, count in groups.items():
                representation_ratio = count / total_samples
                
                # Increase augmentation for underrepresented groups
                if representation_ratio < 0.1:  # Less than 10%
                    intensity = "heavy"
                    prob = 0.9
                elif representation_ratio < 0.2:  # Less than 20%
                    intensity = "medium"
                    prob = 0.8
                else:
                    intensity = "light"
                    prob = 0.6
                
                group_key = f"{attr}_{group}"
                strategies[group_key] = self._create_group_specific_augmentation(
                    intensity, prob
                )
        
        return strategies
    
    def _create_group_specific_augmentation(
        self, 
        intensity: str, 
        probability: float
    ) -> Callable:
        """Create group-specific augmentation function."""
        pipeline = self.pipelines[intensity]
        
        def augment_fn(image: np.ndarray) -> np.ndarray:
            if random.random() < probability:
                return pipeline(image=image)["image"]
            else:
                transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
                return transform(image=image)["image"]
        
        return augment_fn
    
    def generate_synthetic_samples(
        self,
        images: List[np.ndarray],
        demographics: List[Dict[str, Any]],
        target_count: int,
        demographic_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Generate synthetic samples through heavy augmentation.
        
        Args:
            images: Source images
            demographics: Source demographic data
            target_count: Number of synthetic samples to generate
            demographic_filter: Filter for specific demographic groups
        
        Returns:
            Synthetic images and their demographic data
        """
        # Filter source data if specified
        if demographic_filter:
            filtered_indices = []
            for i, demo in enumerate(demographics):
                if all(demo.get(k) == v for k, v in demographic_filter.items()):
                    filtered_indices.append(i)
            
            if not filtered_indices:
                logger.warning(f"No samples found for filter: {demographic_filter}")
                return [], []
            
            source_images = [images[i] for i in filtered_indices]
            source_demographics = [demographics[i] for i in filtered_indices]
        else:
            source_images = images
            source_demographics = demographics
        
        synthetic_images = []
        synthetic_demographics = []
        
        # Generate synthetic samples
        heavy_pipeline = self.pipelines["heavy"]
        
        for _ in range(target_count):
            # Randomly select source sample
            idx = random.randint(0, len(source_images) - 1)
            source_image = source_images[idx]
            source_demo = source_demographics[idx].copy()
            
            # Apply heavy augmentation
            try:
                augmented = heavy_pipeline(image=source_image)
                synthetic_images.append(augmented["image"])
                
                # Mark as synthetic
                source_demo["synthetic"] = True
                synthetic_demographics.append(source_demo)
                
            except Exception as e:
                logger.error(f"Failed to generate synthetic sample: {e}")
                continue
        
        logger.info(f"Generated {len(synthetic_images)} synthetic samples")
        return synthetic_images, synthetic_demographics
    
    def validate_augmentation_fairness(
        self,
        original_distribution: Dict[str, Any],
        augmented_distribution: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Validate that augmentation maintains fairness.
        
        Args:
            original_distribution: Original demographic distribution
            augmented_distribution: Post-augmentation distribution
        
        Returns:
            Fairness metrics
        """
        fairness_scores = {}
        
        for attr in original_distribution.get('demographic_breakdown', {}):
            original_groups = original_distribution['demographic_breakdown'][attr]
            augmented_groups = augmented_distribution['demographic_breakdown'][attr]
            
            # Calculate representation shift
            shifts = []
            for group in original_groups:
                if group in augmented_groups:
                    orig_prop = original_groups[group].get('proportion', 0)
                    aug_prop = augmented_groups[group].get('proportion', 0)
                    shift = abs(aug_prop - orig_prop)
                    shifts.append(shift)
            
            # Average representation shift (lower is better)
            fairness_scores[f"{attr}_representation_stability"] = np.mean(shifts)
        
        # Overall fairness score
        fairness_scores["overall_fairness"] = np.mean(list(fairness_scores.values()))
        
        return fairness_scores
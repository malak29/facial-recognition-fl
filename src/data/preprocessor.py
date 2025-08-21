import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from config.settings import settings

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for facial recognition."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = None,
        normalize: bool = True,
        augment: bool = False,
        preserve_aspect_ratio: bool = True
    ):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image dimensions (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
            augment: Whether to apply data augmentation
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
        """
        self.target_size = target_size or (settings.image_height, settings.image_width)
        self.normalize = normalize
        self.augment = augment
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Initialize scalers
        self.pixel_scaler = StandardScaler() if not normalize else None
        self.label_encoder = LabelEncoder()
        
        # Face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using OpenCV cascade classifier.
        
        Args:
            image: Input image array
        
        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces.tolist()
    
    def crop_face(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Crop face from image with padding.
        
        Args:
            image: Input image
            bbox: Face bounding box (x, y, w, h)
            padding: Padding factor around face
        
        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        # Calculate crop coordinates
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)
        
        return image[y1:y2, x1:x2]
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
        
        Returns:
            Resized image
        """
        if self.preserve_aspect_ratio:
            return self._resize_with_aspect_ratio(image)
        else:
            return cv2.resize(image, (self.target_size[1], self.target_size[0]))
    
    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """Resize image while preserving aspect ratio."""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas and center image
        canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image
        
        Returns:
            Normalized image
        """
        if self.normalize:
            return image.astype(np.float32) / 255.0
        else:
            # Use fitted scaler for standardization
            if self.pixel_scaler is not None:
                original_shape = image.shape
                flattened = image.reshape(-1, 1)
                scaled = self.pixel_scaler.transform(flattened)
                return scaled.reshape(original_shape).astype(np.float32)
            return image.astype(np.float32)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Complete image preprocessing pipeline.
        
        Args:
            image: Raw input image
        
        Returns:
            Preprocessed image ready for model
        """
        try:
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                logger.warning("No faces detected, using full image")
                processed = image
            else:
                # Use largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                processed = self.crop_face(image, largest_face)
            
            # Resize image
            processed = self.resize_image(processed)
            
            # Normalize pixels
            processed = self.normalize_image(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Fallback to basic resize and normalize
            processed = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            return self.normalize_image(processed)
    
    def fit_scalers(self, images: List[np.ndarray], labels: List[str]) -> None:
        """
        Fit scalers on training data.
        
        Args:
            images: List of training images
            labels: List of training labels
        """
        logger.info("Fitting preprocessor scalers...")
        
        # Fit label encoder
        self.label_encoder.fit(labels)
        
        # Fit pixel scaler if not using simple normalization
        if self.pixel_scaler is not None:
            # Sample images for fitting to avoid memory issues
            sample_size = min(1000, len(images))
            sample_indices = np.random.choice(len(images), sample_size, replace=False)
            
            pixel_values = []
            for idx in sample_indices:
                img = self.resize_image(images[idx])
                pixel_values.extend(img.flatten())
            
            pixel_array = np.array(pixel_values).reshape(-1, 1)
            self.pixel_scaler.fit(pixel_array)
        
        logger.info("Preprocessor fitting completed")
    
    def compute_class_weights(self, labels: List[str]) -> Dict[str, float]:
        """
        Compute class weights for balanced training.
        
        Args:
            labels: List of training labels
        
        Returns:
            Dictionary mapping class names to weights
        """
        encoded_labels = self.label_encoder.transform(labels)
        unique_classes = np.unique(encoded_labels)
        
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=encoded_labels
        )
        
        # Map to class names
        class_names = self.label_encoder.inverse_transform(unique_classes)
        return dict(zip(class_names, class_weights))


class BiasAwarePreprocessor(ImagePreprocessor):
    """Extended preprocessor with bias mitigation features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demographic_stats = {}
        self.bias_mitigation_enabled = True
    
    def analyze_demographic_distribution(
        self, 
        images: List[np.ndarray], 
        demographics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze demographic distribution in dataset.
        
        Args:
            images: List of images
            demographics: List of demographic metadata
        
        Returns:
            Distribution statistics
        """
        stats = {
            'total_samples': len(images),
            'demographic_breakdown': {},
            'intersectional_analysis': {}
        }
        
        # Analyze each demographic attribute
        for demo_data in demographics:
            for attr, value in demo_data.items():
                if attr not in stats['demographic_breakdown']:
                    stats['demographic_breakdown'][attr] = {}
                
                if value not in stats['demographic_breakdown'][attr]:
                    stats['demographic_breakdown'][attr][value] = 0
                
                stats['demographic_breakdown'][attr][value] += 1
        
        # Calculate proportions
        for attr in stats['demographic_breakdown']:
            total = sum(stats['demographic_breakdown'][attr].values())
            for value in stats['demographic_breakdown'][attr]:
                count = stats['demographic_breakdown'][attr][value]
                stats['demographic_breakdown'][attr][value] = {
                    'count': count,
                    'proportion': count / total
                }
        
        self.demographic_stats = stats
        return stats
    
    def apply_bias_mitigation(
        self, 
        images: List[np.ndarray], 
        demographics: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Apply bias mitigation strategies during preprocessing.
        
        Args:
            images: List of images
            demographics: List of demographic data
        
        Returns:
            Bias-mitigated images and demographics
        """
        if not self.bias_mitigation_enabled:
            return images, demographics
        
        # Implement demographic parity through sampling
        return self._balance_demographic_representation(images, demographics)
    
    def _balance_demographic_representation(
        self, 
        images: List[np.ndarray], 
        demographics: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Balance demographic representation in dataset."""
        # Implementation for demographic balancing
        # This is a simplified version - real implementation would be more sophisticated
        
        balanced_images = []
        balanced_demographics = []
        
        # Group by demographics
        demographic_groups = {}
        for i, demo in enumerate(demographics):
            # Create a key from demographic attributes
            key = tuple(sorted(demo.items()))
            if key not in demographic_groups:
                demographic_groups[key] = []
            demographic_groups[key].append(i)
        
        # Find minimum group size for balanced sampling
        min_size = min(len(indices) for indices in demographic_groups.values())
        target_size = min(min_size, len(images) // len(demographic_groups))
        
        # Sample equally from each group
        for indices in demographic_groups.values():
            sampled_indices = np.random.choice(indices, target_size, replace=False)
            for idx in sampled_indices:
                balanced_images.append(images[idx])
                balanced_demographics.append(demographics[idx])
        
        logger.info(
            f"Balanced dataset: {len(images)} -> {len(balanced_images)} samples "
            f"across {len(demographic_groups)} demographic groups"
        )
        
        return balanced_images, balanced_demographics
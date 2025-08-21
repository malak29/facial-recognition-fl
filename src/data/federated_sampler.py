import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from collections import defaultdict, Counter
import random
from abc import ABC, abstractmethod

from config.settings import settings

logger = logging.getLogger(__name__)


class FederatedSampler(ABC):
    """Base class for federated learning data sampling strategies."""
    
    def __init__(self, num_clients: int, seed: Optional[int] = None):
        """
        Initialize federated sampler.
        
        Args:
            num_clients: Number of federated clients
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    @abstractmethod
    def sample(
        self,
        data: List[Any],
        labels: List[str],
        demographics: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[int, Dict[str, List[Any]]]:
        """
        Sample data for federated clients.
        
        Args:
            data: List of data samples
            labels: List of labels
            demographics: Optional demographic information
        
        Returns:
            Dictionary mapping client_id to client data
        """
        pass
    
    def _create_client_data_dict(self, indices: List[int]) -> Dict[str, List[int]]:
        """Create standardized client data dictionary."""
        return {
            'indices': indices,
            'size': len(indices)
        }


class IIDSampler(FederatedSampler):
    """Independent and Identically Distributed sampling."""
    
    def sample(
        self,
        data: List[Any],
        labels: List[str],
        demographics: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[int, Dict[str, List[Any]]]:
        """
        Randomly distribute data across clients (IID).
        
        Args:
            data: List of data samples
            labels: List of labels
            demographics: Optional demographic information
        
        Returns:
            Dictionary mapping client_id to client data
        """
        total_samples = len(data)
        samples_per_client = total_samples // self.num_clients
        
        # Shuffle indices
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        
        client_data = {}
        
        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            
            if client_id == self.num_clients - 1:
                # Last client gets remaining samples
                client_indices = indices[start_idx:]
            else:
                end_idx = start_idx + samples_per_client
                client_indices = indices[start_idx:end_idx]
            
            client_data[client_id] = self._create_client_data_dict(client_indices)
        
        logger.info(f"IID sampling completed for {self.num_clients} clients")
        return client_data


class NonIIDSampler(FederatedSampler):
    """Non-IID sampling with label distribution skew."""
    
    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize Non-IID sampler.
        
        Args:
            num_clients: Number of federated clients
            alpha: Dirichlet distribution parameter (lower = more skewed)
            seed: Random seed for reproducibility
        """
        super().__init__(num_clients, seed)
        self.alpha = alpha
    
    def sample(
        self,
        data: List[Any],
        labels: List[str],
        demographics: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[int, Dict[str, List[Any]]]:
        """
        Distribute data with label skew using Dirichlet distribution.
        
        Args:
            data: List of data samples
            labels: List of labels
            demographics: Optional demographic information
        
        Returns:
            Dictionary mapping client_id to client data
        """
        # Group samples by label
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)
        
        unique_labels = list(label_to_indices.keys())
        num_classes = len(unique_labels)
        
        # Generate Dirichlet distribution for each class
        client_data = {i: [] for i in range(self.num_clients)}
        
        for label in unique_labels:
            label_indices = label_to_indices[label]
            np.random.shuffle(label_indices)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Distribute samples based on proportions
            cumsum_props = np.cumsum(proportions)
            total_label_samples = len(label_indices)
            
            start_idx = 0
            for client_id in range(self.num_clients):
                if client_id == self.num_clients - 1:
                    end_idx = total_label_samples
                else:
                    end_idx = int(cumsum_props[client_id] * total_label_samples)
                
                client_samples = label_indices[start_idx:end_idx]
                client_data[client_id].extend(client_samples)
                start_idx = end_idx
        
        # Convert to standard format
        formatted_client_data = {}
        for client_id, indices in client_data.items():
            formatted_client_data[client_id] = self._create_client_data_dict(indices)
        
        logger.info(
            f"Non-IID sampling completed for {self.num_clients} clients "
            f"with alpha={self.alpha}"
        )
        return formatted_client_data


class DemographicAwareSampler(FederatedSampler):
    """Demographic-aware sampling for bias mitigation."""
    
    def __init__(
        self,
        num_clients: int,
        strategy: str = "balanced",  # balanced, hospital, geographic
        seed: Optional[int] = None
    ):
        """
        Initialize demographic-aware sampler.
        
        Args:
            num_clients: Number of federated clients
            strategy: Sampling strategy
            seed: Random seed for reproducibility
        """
        super().__init__(num_clients, seed)
        self.strategy = strategy
    
    def sample(
        self,
        data: List[Any],
        labels: List[str],
        demographics: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[int, Dict[str, List[Any]]]:
        """
        Sample data considering demographic fairness.
        
        Args:
            data: List of data samples
            labels: List of labels
            demographics: Demographic information (required)
        
        Returns:
            Dictionary mapping client_id to client data
        """
        if demographics is None:
            raise ValueError("Demographics required for demographic-aware sampling")
        
        if self.strategy == "balanced":
            return self._balanced_demographic_sampling(data, labels, demographics)
        elif self.strategy == "hospital":
            return self._hospital_based_sampling(data, labels, demographics)
        elif self.strategy == "geographic":
            return self._geographic_sampling(data, labels, demographics)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _balanced_demographic_sampling(
        self,
        data: List[Any],
        labels: List[str],
        demographics: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, List[Any]]]:
        """Ensure balanced demographic representation across clients."""
        
        # Group samples by demographic attributes
        demographic_groups = defaultdict(list)
        for idx, demo in enumerate(demographics):
            # Create composite key from demographic attributes
            demo_key = tuple(sorted(
                (k, v) for k, v in demo.items() 
                if k in ['age_group', 'gender', 'ethnicity']
            ))
            demographic_groups[demo_key].append(idx)
        
        client_data = {i: [] for i in range(self.num_clients)}
        
        # Distribute each demographic group across all clients
        for demo_key, indices in demographic_groups.items():
            np.random.shuffle(indices)
            
            # Distribute evenly across clients
            samples_per_client = len(indices) // self.num_clients
            remainder = len(indices) % self.num_clients
            
            start_idx = 0
            for client_id in range(self.num_clients):
                # Add extra sample to first 'remainder' clients
                extra = 1 if client_id < remainder else 0
                end_idx = start_idx + samples_per_client + extra
                
                client_samples = indices[start_idx:end_idx]
                client_data[client_id].extend(client_samples)
                start_idx = end_idx
        
        # Convert to standard format
        formatted_client_data = {}
        for client_id, indices in client_data.items():
            formatted_client_data[client_id] = self._create_client_data_dict(indices)
        
        logger.info("Balanced demographic sampling completed")
        return formatted_client_data
    
    def _hospital_based_sampling(
        self,
        data: List[Any],
        labels: List[str],
        demographics: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, List[Any]]]:
        """Simulate hospital-based federated learning."""
        
        # Group by hospital/institution
        hospital_groups = defaultdict(list)
        for idx, demo in enumerate(demographics):
            hospital_id = demo.get('hospital_id', f'hospital_{idx % 10}')
            hospital_groups[hospital_id].append(idx)
        
        hospitals = list(hospital_groups.keys())
        
        # Assign hospitals to clients
        hospitals_per_client = len(hospitals) // self.num_clients
        if hospitals_per_client == 0:
            hospitals_per_client = 1
        
        client_data = {}
        hospital_idx = 0
        
        for client_id in range(self.num_clients):
            client_indices = []
            
            # Assign hospitals to this client
            for _ in range(hospitals_per_client):
                if hospital_idx < len(hospitals):
                    hospital = hospitals[hospital_idx]
                    client_indices.extend(hospital_groups[hospital])
                    hospital_idx += 1
            
            # Assign remaining hospitals to last client
            if client_id == self.num_clients - 1:
                while hospital_idx < len(hospitals):
                    hospital = hospitals[hospital_idx]
                    client_indices.extend(hospital_groups[hospital])
                    hospital_idx += 1
            
            client_data[client_id] = self._create_client_data_dict(client_indices)
        
        logger.info("Hospital-based sampling completed")
        return client_data
    
    def _geographic_sampling(
        self,
        data: List[Any],
        labels: List[str],
        demographics: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, List[Any]]]:
        """Sample based on geographic regions."""
        
        # Group by geographic region
        region_groups = defaultdict(list)
        for idx, demo in enumerate(demographics):
            region = demo.get('region', f'region_{idx % 5}')
            region_groups[region].append(idx)
        
        regions = list(region_groups.keys())
        client_data = {}
        
        # Assign regions to clients
        regions_per_client = max(1, len(regions) // self.num_clients)
        
        region_idx = 0
        for client_id in range(self.num_clients):
            client_indices = []
            
            # Assign regions to this client
            for _ in range(regions_per_client):
                if region_idx < len(regions):
                    region = regions[region_idx]
                    client_indices.extend(region_groups[region])
                    region_idx += 1
            
            # Assign remaining regions to last client
            if client_id == self.num_clients - 1:
                while region_idx < len(regions):
                    region = regions[region_idx]
                    client_indices.extend(region_groups[region])
                    region_idx += 1
            
            client_data[client_id] = self._create_client_data_dict(client_indices)
        
        logger.info("Geographic sampling completed")
        return client_data
    
    def analyze_client_demographics(
        self,
        client_data: Dict[int, Dict[str, List[Any]]],
        demographics: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze demographic distribution across clients.
        
        Args:
            client_data: Client data assignments
            demographics: Full demographic information
        
        Returns:
            Demographic analysis for each client
        """
        client_demographics = {}
        
        for client_id, client_info in client_data.items():
            indices = client_info['indices']
            client_demos = [demographics[i] for i in indices]
            
            # Analyze distribution
            analysis = {
                'total_samples': len(indices),
                'demographic_breakdown': {}
            }
            
            # Count each demographic attribute
            for demo in client_demos:
                for attr, value in demo.items():
                    if attr not in analysis['demographic_breakdown']:
                        analysis['demographic_breakdown'][attr] = Counter()
                    analysis['demographic_breakdown'][attr][value] += 1
            
            # Convert to proportions
            for attr in analysis['demographic_breakdown']:
                total = sum(analysis['demographic_breakdown'][attr].values())
                for value in analysis['demographic_breakdown'][attr]:
                    count = analysis['demographic_breakdown'][attr][value]
                    analysis['demographic_breakdown'][attr][value] = {
                        'count': count,
                        'proportion': count / total
                    }
            
            client_demographics[client_id] = analysis
        
        return client_demographics


def create_sampler(
    sampler_type: str,
    num_clients: int,
    **kwargs
) -> FederatedSampler:
    """
    Factory function to create appropriate sampler.
    
    Args:
        sampler_type: Type of sampler ('iid', 'non_iid', 'demographic_aware')
        num_clients: Number of federated clients
        **kwargs: Additional sampler-specific arguments
    
    Returns:
        Configured sampler instance
    """
    if sampler_type == "iid":
        return IIDSampler(num_clients, **kwargs)
    elif sampler_type == "non_iid":
        return NonIIDSampler(num_clients, **kwargs)
    elif sampler_type == "demographic_aware":
        return DemographicAwareSampler(num_clients, **kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
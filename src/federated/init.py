from .aggregation import (
    FederatedAggregator,
    FedAvgAggregator,
    FedProxAggregator,
    FairFedAggregator,
    create_aggregator
)
from .client import FederatedClient
from .server import FederatedServer
from .communication import CommunicationManager

__version__ = "1.0.0"

__all__ = [
    'FederatedAggregator',
    'FedAvgAggregator', 
    'FedProxAggregator',
    'FairFedAggregator',
    'create_aggregator',
    'FederatedClient',
    'FederatedServer',
    'CommunicationManager'
]
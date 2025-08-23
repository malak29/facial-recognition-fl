import asyncio
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.federated.communication import Message, CommunicationProtocol
from src.models.base_model import BaseModel, ModelConfig
from src.utils.metrics import ModelMetrics, FederatedMetrics


@dataclass
class MockTrainingData:
    """Mock training data for testing."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray = None
    y_val: np.ndarray = None
    demographics: Dict[str, np.ndarray] = None


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, config: ModelConfig, framework: str = "tensorflow"):
        super().__init__(config, framework)
        self.weights = self._generate_random_weights()
        self.training_history = []
        self.evaluation_results = []
        
    def build_model(self):
        """Build mock model."""
        # Return a simple mock object
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=self._mock_predict)
        mock_model.fit = Mock(side_effect=self._mock_fit)
        mock_model.evaluate = Mock(side_effect=self._mock_evaluate)
        return mock_model
    
    def _generate_random_weights(self) -> Dict[str, np.ndarray]:
        """Generate random weights for mock model."""
        return {
            'conv1_kernel': np.random.randn(3, 3, 3, 32) * 0.1,
            'conv1_bias': np.random.randn(32) * 0.1,
            'conv2_kernel': np.random.randn(3, 3, 32, 64) * 0.1,
            'conv2_bias': np.random.randn(64) * 0.1,
            'dense_kernel': np.random.randn(1024, self.config.num_classes) * 0.1,
            'dense_bias': np.random.randn(self.config.num_classes) * 0.1
        }
    
    def _mock_predict(self, X):
        """Mock prediction method."""
        batch_size = len(X) if hasattr(X, '__len__') else 1
        return np.random.randint(0, self.config.num_classes, batch_size)
    
    def _mock_fit(self, X, y, **kwargs):
        """Mock fitting method."""
        # Simulate training by adding to history
        epochs = kwargs.get('epochs', 1)
        
        mock_history = Mock()
        mock_history.history = {
            'loss': [0.8 - i * 0.1 for i in range(epochs)],
            'accuracy': [0.6 + i * 0.05 for i in range(epochs)],
            'val_loss': [0.9 - i * 0.08 for i in range(epochs)],
            'val_accuracy': [0.55 + i * 0.06 for i in range(epochs)]
        }
        
        self.training_history.append(mock_history.history)
        return mock_history
    
    def _mock_evaluate(self, X, y):
        """Mock evaluation method."""
        # Return random but reasonable metrics
        loss = 0.3 + np.random.random() * 0.4
        accuracy = 0.6 + np.random.random() * 0.3
        
        self.evaluation_results.append({'loss': loss, 'accuracy': accuracy})
        return loss, accuracy
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get mock model weights."""
        return self.weights.copy()
    
    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set mock model weights."""
        self.weights = weights.copy()
    
    def update_weights(self, weight_updates: Dict[str, np.ndarray], learning_rate: float = 0.01):
        """Apply weight updates to mock model."""
        for layer_name, update in weight_updates.items():
            if layer_name in self.weights:
                self.weights[layer_name] += learning_rate * update


class MockFederatedClient:
    """Mock federated client for testing."""
    
    def __init__(self, client_id: str, data_samples: int = 100, auto_participate: bool = True):
        self.client_id = client_id
        self.data_samples = data_samples
        self.auto_participate = auto_participate
        self.is_active = True
        self.current_round = 0
        self.training_data = None
        self.model = None
        self.performance_history = []
        
        # Communication simulation
        self.message_queue = []
        self.sent_messages = []
        
    def set_training_data(self, training_data: MockTrainingData):
        """Set training data for mock client."""
        self.training_data = training_data
        self.data_samples = len(training_data.X_train)
    
    def set_model(self, model: MockModel):
        """Set model for mock client."""
        self.model = model
    
    async def register_with_server(self, server_endpoint: str) -> bool:
        """Mock client registration."""
        registration_msg = {
            'client_id': self.client_id,
            'capabilities': {
                'data_samples': self.data_samples,
                'privacy_enabled': False,
                'fairness_enabled': True
            },
            'auto_participate': self.auto_participate
        }
        
        self.sent_messages.append(('registration', registration_msg))
        return True
    
    async def handle_training_request(self, round_number: int, global_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Mock training request handling."""
        if not self.training_data or not self.model:
            return None
        
        # Update model with global weights
        self.model.set_weights(global_weights)
        
        # Simulate local training
        training_time = 2.0 + np.random.normal(0, 0.5)
        
        # Generate weight updates
        weight_updates = {}
        for layer_name, weights in global_weights.items():
            # Small random updates
            update = np.random.randn(*weights.shape) * 0.01
            weight_updates[layer_name] = update
        
        # Simulate training metrics
        loss = 0.5 + np.random.normal(0, 0.1)
        accuracy = 0.7 + np.random.normal(0, 0.1)
        
        training_result = {
            'client_id': self.client_id,
            'round_number': round_number,
            'weight_updates': weight_updates,
            'num_samples': self.data_samples,
            'metrics': {
                'loss': loss,
                'accuracy': max(0.1, min(0.99, accuracy)),
                'training_time': training_time
            },
            'timestamp': time.time()
        }
        
        self.performance_history.append(training_result)
        self.current_round = round_number
        
        return training_result
    
    async def evaluate_model(self) -> Dict[str, float]:
        """Mock model evaluation."""
        if not self.model or not self.training_data:
            return {'error': 'No model or data available'}
        
        # Simulate evaluation
        loss, accuracy = self.model._mock_evaluate(
            self.training_data.X_val or self.training_data.X_train,
            self.training_data.y_val or self.training_data.y_train
        )
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'client_id': self.client_id,
            'timestamp': time.time()
        }
    
    def disconnect(self):
        """Mock client disconnection."""
        self.is_active = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get mock client status."""
        return {
            'client_id': self.client_id,
            'is_active': self.is_active,
            'current_round': self.current_round,
            'data_samples': self.data_samples,
            'model_initialized': self.model is not None,
            'training_data_available': self.training_data is not None,
            'performance_history_length': len(self.performance_history)
        }


class MockFederatedServer:
    """Mock federated server for testing."""
    
    def __init__(self, server_id: str = "mock_server", max_clients: int = 10, min_clients: int = 2):
        self.server_id = server_id
        self.max_clients = max_clients
        self.min_clients = min_clients
        
        self.registered_clients = {}
        self.active_clients = set()
        self.current_round = 0
        self.is_training = False
        
        self.global_model = None
        self.aggregation_results = []
        self.round_history = []
        
        # Communication simulation
        self.message_queue = []
        self.broadcast_messages = []
    
    def set_global_model(self, model: MockModel):
        """Set global model for mock server."""
        self.global_model = model
    
    async def register_client(self, client: MockFederatedClient) -> bool:
        """Register a mock client."""
        if len(self.registered_clients) >= self.max_clients:
            return False
        
        self.registered_clients[client.client_id] = {
            'client': client,
            'registration_time': time.time(),
            'data_samples': client.data_samples,
            'last_seen': time.time()
        }
        self.active_clients.add(client.client_id)
        
        return True
    
    def select_clients_for_round(self, fraction: float = 0.5) -> List[str]:
        """Select clients for training round."""
        if not self.active_clients:
            return []
        
        num_select = max(self.min_clients, int(len(self.active_clients) * fraction))
        num_select = min(num_select, len(self.active_clients))
        
        selected = random.sample(list(self.active_clients), num_select)
        return selected
    
    async def execute_training_round(self, client_fraction: float = 0.5) -> Dict[str, Any]:
        """Execute mock training round."""
        if not self.global_model:
            return {'error': 'No global model available'}
        
        self.current_round += 1
        selected_clients = self.select_clients_for_round(client_fraction)
        
        if len(selected_clients) < self.min_clients:
            return {'error': 'Insufficient clients for training'}
        
        # Get global weights
        global_weights = self.global_model.get_weights()
        
        # Collect client updates
        client_updates = {}
        successful_clients = []
        
        for client_id in selected_clients:
            if client_id in self.registered_clients:
                client = self.registered_clients[client_id]['client']
                
                # Simulate some clients failing
                if random.random() < 0.1:  # 10% failure rate
                    continue
                
                try:
                    update = await client.handle_training_request(self.current_round, global_weights)
                    if update:
                        client_updates[client_id] = update
                        successful_clients.append(client_id)
                except Exception as e:
                    # Client failed
                    continue
        
        if len(successful_clients) < self.min_clients:
            return {'error': 'Too many client failures'}
        
        # Aggregate updates
        aggregated_weights = self._aggregate_updates(client_updates, global_weights)
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        # Record round results
        round_result = {
            'round_number': self.current_round,
            'selected_clients': selected_clients,
            'successful_clients': successful_clients,
            'num_participants': len(successful_clients),
            'total_samples': sum(update['num_samples'] for update in client_updates.values()),
            'aggregation_metrics': {
                'avg_loss': np.mean([update['metrics']['loss'] for update in client_updates.values()]),
                'avg_accuracy': np.mean([update['metrics']['accuracy'] for update in client_updates.values()])
            },
            'timestamp': time.time()
        }
        
        self.round_history.append(round_result)
        return round_result
    
    def _aggregate_updates(self, client_updates: Dict[str, Dict], global_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simple FedAvg aggregation for mock server."""
        if not client_updates:
            return global_weights
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in client_updates.values())
        
        # Initialize aggregated updates
        aggregated_updates = {}
        for layer_name in global_weights:
            aggregated_updates[layer_name] = np.zeros_like(global_weights[layer_name])
        
        # Weighted aggregation
        for client_id, update in client_updates.items():
            weight = update['num_samples'] / total_samples
            for layer_name, layer_update in update['weight_updates'].items():
                aggregated_updates[layer_name] += weight * layer_update
        
        # Apply updates to global weights
        new_weights = {}
        for layer_name in global_weights:
            new_weights[layer_name] = global_weights[layer_name] + aggregated_updates[layer_name]
        
        return new_weights
    
    async def evaluate_global_model(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate global model on test data."""
        if not self.global_model:
            return {'error': 'No global model available'}
        
        X_test, y_test = test_data
        loss, accuracy = self.global_model._mock_evaluate(X_test, y_test)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'round': self.current_round,
            'timestamp': time.time()
        }
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get mock server statistics."""
        return {
            'server_id': self.server_id,
            'current_round': self.current_round,
            'is_training': self.is_training,
            'registered_clients': len(self.registered_clients),
            'active_clients': len(self.active_clients),
            'total_rounds_completed': len(self.round_history),
            'global_model_available': self.global_model is not None
        }


class MockCommunicationProtocol(CommunicationProtocol):
    """Mock communication protocol for testing."""
    
    def __init__(self, node_id: str, simulate_failures: bool = False, failure_rate: float = 0.1):
        self.node_id = node_id
        self.simulate_failures = simulate_failures
        self.failure_rate = failure_rate
        
        self.sent_messages = []
        self.received_messages = []
        self.connections = {}
        self.is_listening = False
        
        # Global message bus for inter-node communication
        if not hasattr(MockCommunicationProtocol, '_global_message_bus'):
            MockCommunicationProtocol._global_message_bus = {}
    
    async def send_message(self, message: Message, endpoint: str) -> bool:
        """Mock message sending."""
        # Simulate network failures
        if self.simulate_failures and random.random() < self.failure_rate:
            return False
        
        # Simulate network delay
        await asyncio.sleep(random.uniform(0.01, 0.1))
        
        # Store sent message
        self.sent_messages.append((message, endpoint))
        
        # Deliver to global message bus
        target_node = endpoint.split('/')[-1] if '/' in endpoint else endpoint
        if target_node not in MockCommunicationProtocol._global_message_bus:
            MockCommunicationProtocol._global_message_bus[target_node] = []
        
        MockCommunicationProtocol._global_message_bus[target_node].append(message)
        
        return True
    
    async def receive_message(self) -> Optional[Message]:
        """Mock message receiving."""
        # Check global message bus for messages
        if self.node_id in MockCommunicationProtocol._global_message_bus:
            messages = MockCommunicationProtocol._global_message_bus[self.node_id]
            if messages:
                message = messages.pop(0)
                self.received_messages.append(message)
                return message
        
        return None
    
    async def start_listening(self, host: str, port: int) -> None:
        """Mock start listening."""
        self.is_listening = True
        # Initialize message bus entry
        MockCommunicationProtocol._global_message_bus[self.node_id] = []
    
    async def stop_listening(self) -> None:
        """Mock stop listening."""
        self.is_listening = False
        # Clean up message bus
        if self.node_id in MockCommunicationProtocol._global_message_bus:
            del MockCommunicationProtocol._global_message_bus[self.node_id]


class MockDataGenerator:
    """Mock data generator for testing."""
    
    @staticmethod
    def generate_image_data(num_samples: int, image_shape: Tuple[int, int, int], num_classes: int) -> MockTrainingData:
        """Generate mock image data."""
        X_train = np.random.randn(num_samples, *image_shape).astype(np.float32)
        y_train = np.random.randint(0, num_classes, num_samples)
        
        val_samples = max(1, num_samples // 5)
        X_val = np.random.randn(val_samples, *image_shape).astype(np.float32)
        y_val = np.random.randint(0, num_classes, val_samples)
        
        # Generate demographic data
        demographics = {
            'age_group': np.random.choice(['young', 'middle', 'old'], num_samples),
            'gender': np.random.choice(['male', 'female'], num_samples),
            'ethnicity': np.random.choice(['A', 'B', 'C', 'D'], num_samples)
        }
        
        return MockTrainingData(X_train, y_train, X_val, y_val, demographics)
    
    @staticmethod
    def generate_non_iid_data(num_clients: int, samples_per_client: List[int], image_shape: Tuple[int, int, int], num_classes: int) -> List[MockTrainingData]:
        """Generate non-IID data for multiple clients."""
        client_datasets = []
        
        for i, num_samples in enumerate(samples_per_client):
            # Each client gets different class distributions
            if i == 0:
                # Client 0: mainly classes 0, 1, 2
                class_probs = [0.4, 0.3, 0.2, 0.05, 0.05] + [0.0] * (num_classes - 5)
            elif i == 1:
                # Client 1: mainly classes 2, 3, 4
                class_probs = [0.05, 0.1, 0.3, 0.3, 0.25] + [0.0] * (num_classes - 5)
            else:
                # Other clients: more uniform but still skewed
                probs = np.random.dirichlet([1.0] * num_classes)
                class_probs = probs.tolist()
            
            # Ensure probabilities sum to 1
            class_probs = class_probs[:num_classes]
            class_probs = [p / sum(class_probs) for p in class_probs]
            
            X_train = np.random.randn(num_samples, *image_shape).astype(np.float32)
            y_train = np.random.choice(num_classes, num_samples, p=class_probs)
            
            val_samples = max(1, num_samples // 5)
            X_val = np.random.randn(val_samples, *image_shape).astype(np.float32)
            y_val = np.random.choice(num_classes, val_samples, p=class_probs)
            
            # Generate demographics with some bias
            age_bias = ['young', 'middle', 'old'][i % 3]
            demographics = {
                'age_group': np.random.choice(
                    ['young', 'middle', 'old'], 
                    num_samples, 
                    p=[0.6 if age_bias == 'young' else 0.2,
                       0.6 if age_bias == 'middle' else 0.2,
                       0.6 if age_bias == 'old' else 0.2]
                ),
                'gender': np.random.choice(['male', 'female'], num_samples),
                'ethnicity': np.random.choice(['A', 'B', 'C', 'D'], num_samples)
            }
            
            client_datasets.append(MockTrainingData(X_train, y_train, X_val, y_val, demographics))
        
        return client_datasets


class MockMetricsTracker:
    """Mock metrics tracker for testing."""
    
    def __init__(self):
        self.model_metrics = []
        self.federated_metrics = []
        self.custom_metrics = {}
    
    def add_model_metrics(self, accuracy: float, loss: float, f1_score: float = None):
        """Add mock model metrics."""
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=accuracy + np.random.normal(0, 0.05),
            recall=accuracy + np.random.normal(0, 0.05),
            f1_score=f1_score or accuracy + np.random.normal(0, 0.03),
            loss=loss,
            auc_roc=accuracy + np.random.normal(0, 0.08),
            num_samples=100 + np.random.randint(-20, 20),
            computation_time=2.5 + np.random.normal(0, 0.5),
            timestamp=time.time()
        )
        self.model_metrics.append(metrics)
        return metrics
    
    def add_federated_metrics(self, round_num: int, num_clients: int, participation_rate: float):
        """Add mock federated metrics."""
        metrics = FederatedMetrics(
            round_number=round_num,
            participating_clients=num_clients,
            total_clients=int(num_clients / participation_rate),
            client_participation_rate=participation_rate,
            aggregation_time=1.5 + np.random.normal(0, 0.3),
            communication_overhead=0.8 + np.random.normal(0, 0.2),
            convergence_rate=0.1 - round_num * 0.01,
            client_data_sizes=[100 + np.random.randint(-20, 20) for _ in range(num_clients)]
        )
        self.federated_metrics.append(metrics)
        return metrics
    
    def get_latest_model_metrics(self) -> Optional[ModelMetrics]:
        """Get latest model metrics."""
        return self.model_metrics[-1] if self.model_metrics else None
    
    def get_latest_federated_metrics(self) -> Optional[FederatedMetrics]:
        """Get latest federated metrics."""
        return self.federated_metrics[-1] if self.federated_metrics else None


# Utility functions for creating mock scenarios

def create_mock_federated_scenario(
    num_clients: int = 3,
    data_samples_per_client: List[int] = None,
    image_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 5,
    non_iid: bool = True
) -> Tuple[MockFederatedServer, List[MockFederatedClient], List[MockTrainingData]]:
    """Create a complete mock federated learning scenario."""
    
    if data_samples_per_client is None:
        data_samples_per_client = [100] * num_clients
    
    # Create server
    server = MockFederatedServer(max_clients=num_clients + 2, min_clients=max(1, num_clients // 2))
    
    # Create global model
    model_config = ModelConfig(
        input_shape=image_shape,
        num_classes=num_classes,
        dropout_rate=0.3
    )
    global_model = MockModel(model_config)
    server.set_global_model(global_model)
    
    # Generate data
    if non_iid:
        datasets = MockDataGenerator.generate_non_iid_data(
            num_clients, data_samples_per_client, image_shape, num_classes
        )
    else:
        datasets = [
            MockDataGenerator.generate_image_data(samples, image_shape, num_classes)
            for samples in data_samples_per_client
        ]
    
    # Create clients
    clients = []
    for i in range(num_clients):
        client = MockFederatedClient(
            client_id=f"mock_client_{i}",
            data_samples=data_samples_per_client[i],
            auto_participate=True
        )
        
        # Set training data and model
        client.set_training_data(datasets[i])
        client_model = MockModel(model_config)
        client.set_model(client_model)
        
        clients.append(client)
    
    return server, clients, datasets


async def run_mock_training_simulation(
    server: MockFederatedServer,
    clients: List[MockFederatedClient],
    num_rounds: int = 5,
    client_participation_rate: float = 0.8
) -> List[Dict[str, Any]]:
    """Run a mock federated training simulation."""
    
    # Register all clients
    for client in clients:
        await server.register_client(client)
    
    # Run training rounds
    round_results = []
    
    for round_num in range(num_rounds):
        # Simulate some clients going offline
        active_clients = []
        for client in clients:
            if random.random() < client_participation_rate:
                client.is_active = True
                active_clients.append(client)
            else:
                client.is_active = False
        
        # Update server's active clients
        server.active_clients = {client.client_id for client in active_clients}
        
        # Execute training round
        result = await server.execute_training_round(client_fraction=1.0)
        round_results.append(result)
        
        # Simulate network delays
        await asyncio.sleep(0.1)
    
    return round_results


if __name__ == "__main__":
    # Example usage
    async def test_mock_scenario():
        server, clients, datasets = create_mock_federated_scenario(
            num_clients=3,
            data_samples_per_client=[80, 120, 100],
            non_iid=True
        )
        
        results = await run_mock_training_simulation(
            server, clients, num_rounds=3
        )
        
        print(f"Completed {len(results)} training rounds")
        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"Round {i+1}: {result['num_participants']} participants, "
                      f"avg accuracy: {result['aggregation_metrics']['avg_accuracy']:.3f}")
            else:
                print(f"Round {i+1}: {result['error']}")
        
        print(f"Server statistics: {server.get_server_statistics()}")
    
    # Run the test
    asyncio.run(test_mock_scenario())
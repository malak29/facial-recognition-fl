import pytest
import asyncio
import numpy as np
import tempfile
import os
import time
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import threading

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.federated.server import FederatedServer, FederatedServerConfig
from src.federated.client import FederatedClient, FederatedClientConfig
from src.federated.communication import CommunicationManager, CommunicationConfig, Message
from src.models.base_model import ModelConfig
from src.models.server_model import ServerModelConfig
from src.models.client_model import ClientModelConfig
from src.federated.aggregation import AggregationConfig
from src.utils.metrics import PerformanceTracker, ModelMetrics


class TestFederatedCommunication:
    """Test federated communication between clients and server."""
    
    @pytest.fixture
    def communication_config(self):
        """Communication configuration fixture."""
        return CommunicationConfig(
            protocol="http",
            encryption_enabled=False,  # Disabled for testing
            timeout_seconds=10,
            max_retries=2
        )
    
    @pytest.mark.asyncio
    async def test_message_creation_and_validation(self):
        """Test message creation and validation."""
        message = Message(
            message_id="test_123",
            message_type="test_message",
            sender_id="client_1",
            recipient_id="server",
            payload={"data": "test"},
            timestamp=time.time()
        )
        
        # Test checksum calculation
        checksum = message.calculate_checksum()
        message.checksum = checksum
        
        assert message.verify_checksum() == True
        
        # Test serialization
        message_dict = message.to_dict()
        reconstructed = Message.from_dict(message_dict)
        
        assert reconstructed.message_id == message.message_id
        assert reconstructed.message_type == message.message_type
    
    @pytest.mark.asyncio
    async def test_communication_manager_setup(self, communication_config):
        """Test communication manager initialization."""
        comm_manager = CommunicationManager("test_node", communication_config)
        
        assert comm_manager.node_id == "test_node"
        assert comm_manager.config == communication_config
        
        # Test peer registration
        comm_manager.register_peer("peer_1", "http://localhost:8001")
        assert "peer_1" in comm_manager.peer_endpoints
    
    @pytest.mark.asyncio
    async def test_message_handler_registration(self, communication_config):
        """Test message handler registration."""
        comm_manager = CommunicationManager("test_node", communication_config)
        
        handler_called = False
        
        async def test_handler(message):
            nonlocal handler_called
            handler_called = True
        
        comm_manager.register_message_handler("test_type", test_handler)
        
        assert "test_type" in comm_manager.message_handlers
        
        # Simulate message processing
        test_message = Message(
            message_id="test",
            message_type="test_type",
            sender_id="sender",
            recipient_id="test_node",
            payload={},
            timestamp=time.time()
        )
        test_message.checksum = test_message.calculate_checksum()
        
        await comm_manager._process_incoming_message(test_message)
        assert handler_called == True


class TestFederatedClientServer:
    """Test federated client-server interactions."""
    
    @pytest.fixture
    def model_config(self):
        """Model configuration fixture."""
        return ModelConfig(
            input_shape=(32, 32, 3),
            num_classes=2,
            dropout_rate=0.3
        )
    
    @pytest.fixture
    def server_config(self, model_config):
        """Server configuration fixture."""
        comm_config = CommunicationConfig(
            protocol="http",
            encryption_enabled=False,
            timeout_seconds=30
        )
        
        server_model_config = ServerModelConfig(
            base_config=model_config,
            num_clients=3,
            aggregation_strategy="fedavg",
            min_clients=2,
            max_rounds=5
        )
        
        agg_config = AggregationConfig(strategy="fedavg")
        
        return FederatedServerConfig(
            server_id="test_server",
            communication_config=comm_config,
            model_config=server_model_config,
            aggregation_config=agg_config,
            min_clients=2,
            max_clients=5
        )
    
    @pytest.fixture
    def client_config(self, model_config):
        """Client configuration fixture."""
        comm_config = CommunicationConfig(
            protocol="http",
            encryption_enabled=False
        )
        
        client_model_config = ClientModelConfig(
            base_config=model_config,
            client_id=1,
            local_epochs=2,
            local_batch_size=16
        )
        
        return FederatedClientConfig(
            client_id="test_client_1",
            server_endpoint="http://localhost:8080",
            communication_config=comm_config,
            model_config=client_model_config,
            auto_participate=False,  # Manual control for testing
            min_data_samples=10
        )
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, server_config):
        """Test federated server initialization."""
        server = FederatedServer(server_config)
        
        assert server.server_id == "test_server"
        assert server.config == server_config
        assert len(server.registered_clients) == 0
        assert server.current_round == 0
        assert server.is_training == False
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client_config):
        """Test federated client initialization."""
        client = FederatedClient(client_config)
        
        assert client.client_id == "test_client_1"
        assert client.config == client_config
        assert client.current_round == 0
        assert client.is_participating == False
    
    @pytest.mark.asyncio
    async def test_client_registration_flow(self, server_config, client_config):
        """Test client registration with server."""
        # Mock communication for testing
        server = FederatedServer(server_config)
        client = FederatedClient(client_config)
        
        # Mock the registration message handling
        registration_message = Message(
            message_id="reg_001",
            message_type="client_registration",
            sender_id=client_config.client_id,
            recipient_id=server_config.server_id,
            payload={
                'client_id': client_config.client_id,
                'capabilities': {
                    'privacy_enabled': False,
                    'fairness_enabled': False,
                    'data_samples': 100
                },
                'client_config': {'auto_participate': False}
            },
            timestamp=time.time()
        )
        
        # Process registration
        await server._handle_client_registration(registration_message)
        
        # Verify client is registered
        assert client_config.client_id in server.registered_clients
        assert client_config.client_id in server.active_clients
    
    @pytest.mark.asyncio
    async def test_training_data_setup(self, client_config):
        """Test setting up training data on client."""
        client = FederatedClient(client_config)
        
        # Create mock training data
        X_train = np.random.randn(50, 32, 32, 3)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(10, 32, 32, 3)
        y_val = np.random.randint(0, 2, 10)
        
        # Set training data
        client.set_training_data(X_train, y_train, X_val, y_val)
        
        assert client.training_data is not None
        assert client.validation_data is not None
        assert len(client.training_data[0]) == 50
        assert len(client.validation_data[0]) == 10
    
    @pytest.mark.asyncio
    async def test_training_request_handling(self, server_config, client_config):
        """Test training request and response flow."""
        server = FederatedServer(server_config)
        client = FederatedClient(client_config)
        
        # Setup client data
        X_train = np.random.randn(30, 32, 32, 3)
        y_train = np.random.randint(0, 2, 30)
        client.set_training_data(X_train, y_train)
        
        # Mock training request message
        training_request = Message(
            message_id="train_001",
            message_type="training_request",
            sender_id=server_config.server_id,
            recipient_id=client_config.client_id,
            payload={
                'round_number': 1,
                'timeout': 300,
                'local_epochs': 2
            },
            timestamp=time.time(),
            round_number=1
        )
        
        # Process training request
        await client._handle_training_request(training_request)
        
        # Verify client accepted training
        assert client.current_round == 1
        assert client.is_participating == True
    
    @pytest.mark.asyncio
    async def test_model_weights_distribution(self, server_config):
        """Test distributing global weights to clients."""
        server = FederatedServer(server_config)
        
        # Mock global weights
        global_weights = {
            'layer1': np.random.randn(10, 5),
            'layer2': np.random.randn(5, 2)
        }
        
        with patch.object(server.model, 'get_global_weights', return_value=global_weights):
            selected_clients = ['client_1', 'client_2']
            
            # Mock communication
            with patch.object(server.comm_manager, 'send_message') as mock_send:
                mock_send.return_value = True
                
                await server._distribute_global_weights(selected_clients)
                
                # Verify weights were sent to all selected clients
                assert mock_send.call_count == len(selected_clients)
    
    @pytest.mark.asyncio
    async def test_client_update_aggregation(self, server_config):
        """Test aggregating client updates on server."""
        server = FederatedServer(server_config)
        
        # Mock client updates
        server.client_results = {
            'client_1': {
                'training_result': {
                    'weight_updates': {
                        'layer1': np.random.randn(10, 5) * 0.1,
                        'layer2': np.random.randn(5, 2) * 0.1
                    },
                    'num_samples': 50,
                    'final_loss': 0.5
                },
                'data_samples': 50,
                'timestamp': time.time()
            },
            'client_2': {
                'training_result': {
                    'weight_updates': {
                        'layer1': np.random.randn(10, 5) * 0.1,
                        'layer2': np.random.randn(5, 2) * 0.1
                    },
                    'num_samples': 30,
                    'final_loss': 0.6
                },
                'data_samples': 30,
                'timestamp': time.time()
            }
        }
        
        # Mock aggregation
        with patch.object(server, '_aggregate_sync') as mock_agg:
            mock_agg.return_value = {
                'aggregated_weights': {
                    'layer1': np.random.randn(10, 5),
                    'layer2': np.random.randn(5, 2)
                },
                'metrics': {
                    'aggregation_method': 'fedavg',
                    'num_clients': 2,
                    'total_samples': 80
                }
            }
            
            result = await server._aggregate_client_results()
            
            assert 'aggregated_weights' in result
            assert 'metrics' in result
            assert result['metrics']['num_clients'] == 2


class TestFederatedTrainingScenarios:
    """Test complete federated training scenarios."""
    
    @pytest.fixture
    def training_setup(self):
        """Setup for federated training tests."""
        # Model configuration
        model_config = ModelConfig(
            input_shape=(28, 28, 1),
            num_classes=10,
            dropout_rate=0.2
        )
        
        # Server configuration
        comm_config = CommunicationConfig(
            protocol="http",
            encryption_enabled=False,
            timeout_seconds=60
        )
        
        server_model_config = ServerModelConfig(
            base_config=model_config,
            num_clients=3,
            aggregation_strategy="fedavg",
            min_clients=2,
            max_rounds=3,
            convergence_threshold=0.01
        )
        
        server_config = FederatedServerConfig(
            server_id="training_server",
            communication_config=comm_config,
            model_config=server_model_config,
            min_clients=2,
            max_clients=3,
            round_timeout=30
        )
        
        # Client configurations
        client_configs = []
        for i in range(3):
            client_model_config = ClientModelConfig(
                base_config=model_config,
                client_id=i,
                local_epochs=1,
                local_batch_size=8
            )
            
            client_config = FederatedClientConfig(
                client_id=f"client_{i}",
                server_endpoint="http://localhost:8080",
                communication_config=comm_config,
                model_config=client_model_config,
                auto_participate=False,
                min_data_samples=5
            )
            client_configs.append(client_config)
        
        return {
            'server_config': server_config,
            'client_configs': client_configs,
            'model_config': model_config
        }
    
    @pytest.mark.asyncio
    async def test_multi_client_registration(self, training_setup):
        """Test multiple clients registering with server."""
        server = FederatedServer(training_setup['server_config'])
        clients = []
        
        for config in training_setup['client_configs']:
            client = FederatedClient(config)
            clients.append(client)
        
        # Mock registration for all clients
        for i, client in enumerate(clients):
            registration_message = Message(
                message_id=f"reg_{i}",
                message_type="client_registration",
                sender_id=client.client_id,
                recipient_id=server.server_id,
                payload={
                    'client_id': client.client_id,
                    'capabilities': {'data_samples': 20},
                    'client_config': {}
                },
                timestamp=time.time()
            )
            
            await server._handle_client_registration(registration_message)
        
        # Verify all clients registered
        assert len(server.registered_clients) == 3
        assert len(server.active_clients) == 3
    
    @pytest.mark.asyncio
    async def test_client_selection_process(self, training_setup):
        """Test client selection for training rounds."""
        server = FederatedServer(training_setup['server_config'])
        
        # Register clients
        for i, config in enumerate(training_setup['client_configs']):
            server.registered_clients[config.client_id] = {
                'client_id': config.client_id,
                'capabilities': {'data_samples': 20},
                'config': config
            }
            server.active_clients.add(config.client_id)
        
        # Test client selection
        selected_clients = server._select_clients_for_round()
        
        assert len(selected_clients) >= server.config.min_clients
        assert len(selected_clients) <= len(server.active_clients)
        assert all(client_id in server.active_clients for client_id in selected_clients)
    
    @pytest.mark.asyncio
    async def test_round_timeout_handling(self, training_setup):
        """Test handling of client timeouts during training rounds."""
        server = FederatedServer(training_setup['server_config'])
        server.config.round_timeout = 1  # Very short timeout for testing
        
        # Setup selected clients but don't provide results
        server.selected_clients = {'client_1', 'client_2'}
        server.client_results = {}  # No results provided
        
        # Test timeout handling
        start_time = time.time()
        result = await server._wait_for_client_results()
        elapsed = time.time() - start_time
        
        # Should timeout quickly and return False
        assert result == False
        assert elapsed >= 1.0  # Should wait at least the timeout period
    
    @pytest.mark.asyncio
    async def test_partial_client_participation(self, training_setup):
        """Test handling partial client participation."""
        server = FederatedServer(training_setup['server_config'])
        
        # Setup scenario where only some clients participate
        server.selected_clients = {'client_1', 'client_2', 'client_3'}
        server.client_results = {
            'client_1': {
                'training_result': {
                    'weight_updates': {'layer1': np.random.randn(5, 3)},
                    'num_samples': 20
                },
                'data_samples': 20,
                'timestamp': time.time()
            },
            'client_2': {
                'training_result': {
                    'weight_updates': {'layer1': np.random.randn(5, 3)},
                    'num_samples': 15
                },
                'data_samples': 15,
                'timestamp': time.time()
            }
            # client_3 doesn't provide results
        }
        
        # Should proceed with available clients
        with patch.object(server, '_wait_for_client_results', return_value=True):
            success = await server._execute_training_round()
        
        # Should handle partial participation gracefully
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_convergence_detection(self, training_setup):
        """Test training convergence detection."""
        server = FederatedServer(training_setup['server_config'])
        
        # Simulate convergence by adding small changes
        server.convergence_history.extend([0.1, 0.05, 0.008, 0.005, 0.003])
        
        converged = server._check_convergence()
        
        # Should detect convergence
        assert converged == True
        
        # Test non-convergence
        server.convergence_history.clear()
        server.convergence_history.extend([0.5, 0.4, 0.3, 0.25, 0.2])
        
        converged = server._check_convergence()
        assert converged == False


class TestFederatedLearningWithData:
    """Test federated learning with actual data flows."""
    
    @pytest.fixture
    def mnist_like_data(self):
        """Generate MNIST-like synthetic data."""
        def generate_client_data(client_id, num_samples=100):
            # Generate slightly different distributions for each client
            np.random.seed(42 + client_id)
            X = np.random.randn(num_samples, 28, 28, 1)
            y = np.random.randint(0, 10, num_samples)
            return X, y
        
        return generate_client_data
    
    @pytest.mark.asyncio
    async def test_end_to_end_training_simulation(self, mnist_like_data):
        """Test complete end-to-end training simulation."""
        # Setup configuration
        model_config = ModelConfig(
            input_shape=(28, 28, 1),
            num_classes=10,
            dropout_rate=0.1
        )
        
        comm_config = CommunicationConfig(
            protocol="http",
            encryption_enabled=False,
            timeout_seconds=30
        )
        
        server_model_config = ServerModelConfig(
            base_config=model_config,
            num_clients=2,
            aggregation_strategy="fedavg",
            min_clients=1,
            max_rounds=2
        )
        
        server_config = FederatedServerConfig(
            server_id="e2e_server",
            communication_config=comm_config,
            model_config=server_model_config,
            min_clients=1,
            max_clients=2
        )
        
        # Create server and clients
        server = FederatedServer(server_config)
        clients = []
        
        for i in range(2):
            client_model_config = ClientModelConfig(
                base_config=model_config,
                client_id=i,
                local_epochs=1,
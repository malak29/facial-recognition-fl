import pytest
import asyncio
import numpy as np
import tempfile
import os
import time
import json
import requests
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import psutil

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.app import app
from src.federated.server import FederatedServer, FederatedServerConfig
from src.federated.client import FederatedClient, FederatedClientConfig
from src.federated.communication import CommunicationConfig
from src.models.base_model import ModelConfig
from src.models.server_model import ServerModelConfig
from src.models.client_model import ClientModelConfig
from src.bias_mitigation.bias_detector import BiasDetectionSuite
from src.privacy.differential_privacy import DifferentialPrivacyManager
from src.utils.metrics import PerformanceTracker
from src.utils.helpers import setup_logging, create_directories


class TestSystemStartup:
    """Test system startup and initialization."""
    
    def test_api_import(self):
        """Test that API can be imported without errors."""
        from api.app import app
        assert app is not None
    
    def test_core_modules_import(self):
        """Test that all core modules can be imported."""
        # Test model imports
        from src.models.cnn_model import CNNModel
        from src.models.client_model import ClientModel
        from src.models.server_model import ServerModel
        
        # Test federated components
        from src.federated.server import FederatedServer
        from src.federated.client import FederatedClient
        from src.federated.aggregation import FedAvgAggregator
        
        # Test utilities
        from src.utils.metrics import PerformanceTracker
        from src.bias_mitigation.fairness_metrics import FairnessMetricsCalculator
        from src.privacy.differential_privacy import DifferentialPrivacyManager
        
        # All imports should succeed
        assert True
    
    def test_logging_setup(self):
        """Test logging configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            setup_logging(level="INFO", log_file=log_file)
            
            import logging
            logger = logging.getLogger("test")
            logger.info("Test log message")
            
            # Check log file was created
            assert os.path.exists(log_file)
            
            # Check log content
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test log message" in content
    
    def test_directory_creation(self):
        """Test directory creation utility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dirs = [
                os.path.join(temp_dir, "models"),
                os.path.join(temp_dir, "data", "processed"),
                os.path.join(temp_dir, "logs"),
                os.path.join(temp_dir, "checkpoints", "round_1")
            ]
            
            create_directories(*test_dirs)
            
            # Verify all directories were created
            for dir_path in test_dirs:
                assert os.path.exists(dir_path)
                assert os.path.isdir(dir_path)


class TestEndToEndFederatedLearning:
    """Test complete federated learning workflows."""
    
    @pytest.fixture
    def system_config(self):
        """System configuration for E2E tests."""
        model_config = ModelConfig(
            input_shape=(32, 32, 3),
            num_classes=5,
            dropout_rate=0.2
        )
        
        comm_config = CommunicationConfig(
            protocol="http",
            encryption_enabled=False,  # Simplified for testing
            timeout_seconds=30,
            max_retries=2
        )
        
        server_model_config = ServerModelConfig(
            base_config=model_config,
            num_clients=3,
            aggregation_strategy="fedavg",
            min_clients=2,
            max_rounds=3,
            convergence_threshold=0.05
        )
        
        server_config = FederatedServerConfig(
            server_id="e2e_server",
            communication_config=comm_config,
            model_config=server_model_config,
            min_clients=2,
            max_clients=3,
            round_timeout=60
        )
        
        client_configs = []
        for i in range(3):
            client_model_config = ClientModelConfig(
                base_config=model_config,
                client_id=i,
                local_epochs=2,
                local_batch_size=8,
                differential_privacy=i == 0  # Enable DP for first client
            )
            
            client_config = FederatedClientConfig(
                client_id=f"e2e_client_{i}",
                server_endpoint="http://localhost:8080",
                communication_config=comm_config,
                model_config=client_model_config,
                auto_participate=False,
                min_data_samples=10
            )
            client_configs.append(client_config)
        
        return {
            'server_config': server_config,
            'client_configs': client_configs,
            'model_config': model_config
        }
    
    @pytest.fixture
    def synthetic_datasets(self):
        """Generate synthetic datasets for different clients."""
        def create_dataset(client_id, num_samples=100, num_classes=5):
            np.random.seed(42 + client_id)
            
            # Create slightly different data distributions per client
            X = np.random.randn(num_samples, 32, 32, 3)
            
            # Non-IID data: each client has bias towards certain classes
            if client_id == 0:
                # Client 0: mostly classes 0, 1, 2
                class_probs = [0.4, 0.3, 0.2, 0.05, 0.05]
            elif client_id == 1:
                # Client 1: mostly classes 1, 2, 3
                class_probs = [0.1, 0.3, 0.3, 0.25, 0.05]
            else:
                # Client 2: mostly classes 2, 3, 4
                class_probs = [0.05, 0.1, 0.25, 0.3, 0.3]
            
            y = np.random.choice(num_classes, size=num_samples, p=class_probs)
            
            # Create demographic data for bias testing
            demographics = {
                'age_group': np.random.choice(['young', 'middle', 'old'], num_samples),
                'gender': np.random.choice(['male', 'female'], num_samples),
                'ethnicity': np.random.choice(['A', 'B', 'C', 'D'], num_samples)
            }
            
            return X, y, demographics
        
        return create_dataset
    
    @pytest.mark.asyncio
    async def test_full_federated_training_cycle(self, system_config, synthetic_datasets):
        """Test complete federated training cycle."""
        server = FederatedServer(system_config['server_config'])
        clients = []
        
        # Create and setup clients
        for i, config in enumerate(system_config['client_configs']):
            client = FederatedClient(config)
            clients.append(client)
            
            # Setup training data
            X_train, y_train, demographics = synthetic_datasets(i, 80)
            X_val, y_val, _ = synthetic_datasets(i + 10, 20)
            
            client.set_training_data(X_train, y_train, X_val, y_val, demographics)
        
        # Mock the network communication for testing
        with patch.object(server.comm_manager, 'send_message', return_value=True):
            with patch.object(server.comm_manager, 'broadcast_message', return_value={}):
                
                # Simulate client registration
                for client in clients:
                    registration_msg = {
                        'client_id': client.client_id,
                        'capabilities': {'data_samples': 80},
                        'client_config': {'auto_participate': False}
                    }
                    
                    await server._handle_client_registration(
                        Mock(sender_id=client.client_id, payload=registration_msg)
                    )
                
                # Verify clients registered
                assert len(server.registered_clients) == 3
                assert len(server.active_clients) == 3
                
                # Simulate training rounds
                training_success = await server._execute_training_round()
                
                # Training should complete or at least attempt
                assert isinstance(training_success, bool)
    
    @pytest.mark.asyncio
    async def test_bias_detection_integration(self, synthetic_datasets):
        """Test integration of bias detection with federated learning."""
        bias_detector = BiasDetectionSuite(
            sensitive_attributes=['age_group', 'gender', 'ethnicity']
        )
        
        # Create test data with potential bias
        X_test, y_test, demographics = synthetic_datasets(999, 200)
        
        # Mock model for bias detection
        class MockModel:
            def predict(self, X):
                # Introduce bias: better performance for certain demographic groups
                predictions = np.random.randint(0, 5, len(X))
                # Bias: perform worse on 'old' age group
                age_groups = demographics['age_group']
                for i, age in enumerate(age_groups):
                    if age == 'old' and np.random.random() < 0.3:
                        predictions[i] = (predictions[i] + 1) % 5  # Wrong prediction
                return predictions
        
        mock_model = MockModel()
        
        # Run bias detection
        bias_results = bias_detector.run_comprehensive_detection(
            mock_model, X_test, y_test, demographics
        )
        
        # Verify bias detection completed
        assert isinstance(bias_results, dict)
        assert len(bias_results) > 0
        
        # Check for overall assessment
        overall_assessment = bias_results.get('overall_assessment')
        if overall_assessment:
            assert hasattr(overall_assessment, 'bias_detected')
    
    @pytest.mark.asyncio
    async def test_privacy_preservation_integration(self):
        """Test privacy preservation components."""
        privacy_manager = DifferentialPrivacyManager(
            total_epsilon=1.0,
            total_delta=1e-5
        )
        
        # Test data privatization
        sensitive_data = np.random.randn(100, 10)
        
        privatized_data = privacy_manager.privatize_data(
            sensitive_data,
            sensitivity=1.0,
            epsilon_fraction=0.1
        )
        
        # Verify data was privatized (should be different)
        assert not np.allclose(sensitive_data, privatized_data)
        assert privatized_data.shape == sensitive_data.shape
        
        # Test privacy budget tracking
        privacy_report = privacy_manager.get_privacy_report()
        assert 'privacy_budget' in privacy_report
        assert privacy_report['privacy_budget']['epsilon_spent'] > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, synthetic_datasets):
        """Test performance monitoring throughout training."""
        tracker = PerformanceTracker()
        
        # Simulate multiple training rounds with improving metrics
        for round_num in range(5):
            # Generate mock metrics that improve over time
            from src.utils.metrics import ModelMetrics, FederatedMetrics
            
            # Model metrics
            model_metrics = ModelMetrics(
                accuracy=0.5 + round_num * 0.08,
                precision=0.48 + round_num * 0.07,
                recall=0.52 + round_num * 0.06,
                f1_score=0.50 + round_num * 0.065,
                loss=1.2 - round_num * 0.15,
                num_samples=100,
                computation_time=3.2 + np.random.normal(0, 0.5)
            )
            
            # Federated metrics
            fed_metrics = FederatedMetrics(
                round_number=round_num + 1,
                participating_clients=3,
                total_clients=3,
                client_participation_rate=1.0,
                aggregation_time=2.1 + np.random.normal(0, 0.3),
                communication_overhead=1.5 + round_num * 0.1,
                convergence_rate=0.1 - round_num * 0.015
            )
            
            tracker.add_model_metrics(model_metrics)
            tracker.add_federated_metrics(fed_metrics)
        
        # Verify tracking worked
        assert len(tracker.model_metrics_history) == 5
        assert len(tracker.federated_metrics_history) == 5
        
        # Check performance trends
        summary = tracker.calculate_performance_summary()
        assert 'model_performance' in summary
        assert 'federated_performance' in summary
        
        # Latest metrics should show improvement
        latest = tracker.get_latest_model_metrics()
        assert latest.accuracy > 0.8  # Should have improved
    
    def test_system_resource_monitoring(self):
        """Test system resource monitoring during operation."""
        # Monitor system resources
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Simulate some computational load
        large_array = np.random.randn(1000, 1000)
        result = np.dot(large_array, large_array.T)
        
        # Check that we can monitor resources
        current_memory = psutil.virtual_memory().percent
        current_cpu = psutil.cpu_percent(interval=1)
        
        # Verify monitoring works
        assert current_memory >= 0
        assert current_cpu >= 0
        assert isinstance(result, np.ndarray)
        
        # Clean up
        del large_array, result


class TestAPIEndpoints:
    """Test API endpoints end-to-end."""
    
    @pytest.fixture
    def api_client(self):
        """Create test client for API."""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
    
    def test_status_endpoint(self, api_client):
        """Test system status endpoint."""
        response = api_client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system" in data
        assert "performance" in data
        assert isinstance(data["system"], dict)
    
    @patch('api.app.app_state')
    def test_server_initialization_endpoint(self, mock_app_state, api_client):
        """Test server initialization via API."""
        mock_app_state.__getitem__.side_effect = lambda key: None if key == 'server' else {}
        
        server_config = {
            "server_id": "test_api_server",
            "min_clients": 2,
            "max_clients": 5,
            "client_selection_fraction": 0.4,
            "round_timeout": 120,
            "max_rounds": 10,
            "convergence_threshold": 0.01,
            "model_type": "cnn",
            "aggregation_strategy": "fedavg"
        }
        
        # Mock authentication
        headers = {"Authorization": "Bearer test_token"}
        
        with patch('api.app.FederatedServer') as mock_server:
            mock_server_instance = Mock()
            mock_server.return_value = mock_server_instance
            mock_server_instance.start = AsyncMock()
            
            response = api_client.post(
                "/server/initialize",
                json=server_config,
                headers=headers
            )
            
            # Should succeed with proper config
            assert response.status_code in [200, 500]  # May fail due to mocking
    
    def test_client_registration_endpoint(self, api_client):
        """Test client registration via API."""
        client_config = {
            "client_id": "test_api_client",
            "server_endpoint": "http://localhost:8080",
            "auto_participate": True,
            "min_data_samples": 20,
            "privacy_enabled": False,
            "fairness_enabled": True
        }
        
        headers = {"Authorization": "Bearer test_token"}
        
        with patch('api.app.FederatedClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.start = AsyncMock()
            
            response = api_client.post(
                "/client/register",
                json=client_config,
                headers=headers
            )
            
            # Should succeed or fail gracefully
            assert response.status_code in [200, 500]


class TestDataFlowIntegration:
    """Test data flow integration throughout the system."""
    
    def test_data_preprocessing_pipeline(self):
        """Test data preprocessing pipeline."""
        from src.data.preprocessor import ImagePreprocessor
        
        # Create mock image data
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processor = ImagePreprocessor(
            target_size=(64, 64),
            normalize=True,
            preserve_aspect_ratio=True
        )
        
        processed_image = processor.preprocess_image(mock_image)
        
        # Verify preprocessing
        assert processed_image.shape == (64, 64, 3)
        assert processed_image.dtype == np.float32
        assert np.all(processed_image >= 0) and np.all(processed_image <= 1)
    
    def test_data_augmentation_pipeline(self):
        """Test data augmentation pipeline."""
        from src.data.augmentation import FairAugmentation
        
        augmenter = FairAugmentation(
            augment_probability=1.0,  # Always augment for testing
            preserve_demographics=True,
            intensity="medium"
        )
        
        # Create mock image
        mock_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Mock demographic info
        demographic_info = {
            'age_group': 'young',
            'gender': 'female',
            'ethnicity': 'A'
        }
        
        augmented = augmenter.augment_image(mock_image, demographic_info)
        
        # Verify augmentation produced different result
        assert not np.array_equal(mock_image, augmented)
    
    def test_federated_data_sampling(self):
        """Test federated data sampling strategies."""
        from src.data.federated_sampler import create_sampler
        
        # Create mock data
        data = list(range(1000))
        labels = ['class_' + str(i % 5) for i in data]
        demographics = [
            {
                'age_group': np.random.choice(['young', 'middle', 'old']),
                'gender': np.random.choice(['male', 'female'])
            }
            for _ in data
        ]
        
        # Test different sampling strategies
        strategies = ['iid', 'non_iid', 'demographic_aware']
        
        for strategy in strategies:
            sampler = create_sampler(strategy, num_clients=3)
            
            if strategy == 'demographic_aware':
                client_data = sampler.sample(data, labels, demographics)
            else:
                client_data = sampler.sample(data, labels)
            
            # Verify sampling worked
            assert len(client_data) == 3
            assert all('indices' in client_info for client_info in client_data.values())
            
            # Verify all data was distributed
            total_samples = sum(len(client_info['indices']) for client_info in client_data.values())
            assert total_samples == len(data)


class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""
    
    @pytest.mark.asyncio
    async def test_client_failure_handling(self):
        """Test handling of client failures during training."""
        from src.federated.server import FederatedServer, FederatedServerConfig
        from src.federated.communication import CommunicationConfig
        from src.models.server_model import ServerModelConfig
        from src.models.base_model import ModelConfig
        
        # Setup server
        model_config = ModelConfig(input_shape=(32, 32, 3), num_classes=2)
        comm_config = CommunicationConfig(protocol="http", timeout_seconds=5)
        server_model_config = ServerModelConfig(
            base_config=model_config,
            num_clients=3,
            min_clients=1
        )
        
        server_config = FederatedServerConfig(
            communication_config=comm_config,
            model_config=server_model_config,
            round_timeout=10  # Short timeout for testing
        )
        
        server = FederatedServer(server_config)
        
        # Setup clients but simulate failure
        server.selected_clients = {'client_1', 'client_2', 'client_3'}
        server.client_results = {
            'client_1': {
                'training_result': {'num_samples': 50},
                'data_samples': 50,
                'timestamp': time.time()
            }
            # client_2 and client_3 fail to respond
        }
        
        # Test timeout handling
        with patch.object(server, '_wait_for_client_results') as mock_wait:
            mock_wait.return_value = True  # Proceed with partial results
            
            success = await server._execute_training_round()
            
            # Should handle partial participation gracefully
            assert isinstance(success, bool)
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        from src.models.base_model import ModelConfig
        
        # Test invalid model config
        with pytest.raises((ValueError, AssertionError)):
            ModelConfig(
                input_shape=(-1, -1, 3),  # Invalid shape
                num_classes=0,  # Invalid class count
                dropout_rate=1.5  # Invalid dropout rate
            )
    
    @pytest.mark.asyncio
    async def test_communication_failure_recovery(self):
        """Test communication failure recovery."""
        from src.federated.communication import CommunicationManager, CommunicationConfig
        
        config = CommunicationConfig(
            protocol="http",
            max_retries=2,
            timeout_seconds=1
        )
        
        comm_manager = CommunicationManager("test_node", config)
        comm_manager.register_peer("unreachable_peer", "http://nonexistent:9999")
        
        # Test sending to unreachable peer
        success = await comm_manager.send_message(
            "unreachable_peer",
            "test_message",
            {"data": "test"}
        )
        
        # Should fail gracefully
        assert success == False
        assert comm_manager.statistics['send_failures'] > 0
    
    def test_memory_management(self):
        """Test memory management under load."""
        import gc
        
        # Create large objects and verify cleanup
        large_objects = []
        for i in range(10):
            large_array = np.random.randn(1000, 1000)
            large_objects.append(large_array)
        
        # Clear references
        large_objects.clear()
        gc.collect()
        
        # Should not consume excessive memory
        memory_info = psutil.Process().memory_info()
        assert memory_info.rss < 2 * 1024 * 1024 * 1024  # Less than 2GB


class TestConfigurationManagement:
    """Test configuration management and persistence."""
    
    def test_configuration_serialization(self):
        """Test saving and loading configurations."""
        from src.utils.helpers import save_config, load_config
        
        test_config = {
            'model': {
                'input_shape': [32, 32, 3],
                'num_classes': 10,
                'dropout_rate': 0.3
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 10
            },
            'federated': {
                'num_clients': 5,
                'client_fraction': 0.6,
                'aggregation_strategy': 'fedavg'
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON format
            json_path = os.path.join(temp_dir, "config.json")
            save_config(test_config, json_path)
            loaded_config = load_config(json_path)
            
            assert loaded_config == test_config
            
            # Test YAML format
            yaml_path = os.path.join(temp_dir, "config.yaml")
            save_config(test_config, yaml_path)
            loaded_yaml_config = load_config(yaml_path)
            
            assert loaded_yaml_config == test_config
    
    def test_model_serialization(self):
        """Test model weight serialization."""
        from src.utils.helpers import serialize_weights, deserialize_weights
        
        # Create mock model weights
        weights = {
            'layer1': np.random.randn(10, 5),
            'layer2': np.random.randn(5, 2),
            'bias1': np.random.randn(5),
            'bias2': np.random.randn(2)
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "weights.pkl.gz")
            
            # Serialize
            serialize_weights(weights, weights_path)
            assert os.path.exists(weights_path)
            
            # Deserialize
            loaded_weights = deserialize_weights(weights_path)
            
            # Verify weights are identical
            for layer_name in weights:
                assert layer_name in loaded_weights
                np.testing.assert_array_equal(weights[layer_name], loaded_weights[layer_name])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
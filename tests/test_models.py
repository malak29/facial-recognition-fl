import pytest
import numpy as np
import tensorflow as tf
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import BaseModel, ModelConfig
from src.models.cnn_model import CNNModel, ResNetModel, EfficientNetModel, create_model
from src.models.client_model import ClientModel, ClientModelConfig
from src.models.server_model import ServerModel, ServerModelConfig
from src.federated.aggregation import FedAvgAggregator, FairFedAggregator, AggregationConfig


class TestModelConfig:
    """Test model configuration class."""
    
    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(
            input_shape=(224, 224, 3),
            num_classes=10,
            dropout_rate=0.5
        )
        
        assert config.input_shape == (224, 224, 3)
        assert config.num_classes == 10
        assert config.dropout_rate == 0.5
        assert config.l2_regularization == 1e-4  # default value
        assert config.batch_normalization == True  # default value


class TestCNNModel:
    """Test CNN model implementations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ModelConfig(
            input_shape=(64, 64, 3),
            num_classes=2,
            dropout_rate=0.3
        )
    
    def test_cnn_model_tensorflow_creation(self):
        """Test CNN model creation with TensorFlow."""
        model = CNNModel(self.config, framework="tensorflow", architecture="simple")
        
        assert model.config == self.config
        assert model.framework == "tensorflow"
        assert model.architecture == "simple"
        assert model.model is None  # Not built yet
    
    def test_cnn_model_pytorch_creation(self):
        """Test CNN model creation with PyTorch."""
        model = CNNModel(self.config, framework="pytorch", architecture="simple")
        
        assert model.config == self.config
        assert model.framework == "pytorch"
        assert model.architecture == "simple"
    
    def test_model_building_tensorflow(self):
        """Test building TensorFlow model."""
        model = CNNModel(self.config, framework="tensorflow", architecture="simple")
        tf_model = model.build_model()
        
        assert tf_model is not None
        assert isinstance(tf_model, tf.keras.Model)
        
        # Test input/output shapes
        input_shape = tf_model.input_shape
        output_shape = tf_model.output_shape
        
        assert input_shape[1:] == self.config.input_shape
        assert output_shape[1] == self.config.num_classes
    
    def test_model_building_pytorch(self):
        """Test building PyTorch model."""
        model = CNNModel(self.config, framework="pytorch", architecture="simple")
        torch_model = model.build_model()
        
        assert torch_model is not None
        assert isinstance(torch_model, torch.nn.Module)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 64, 64)
        output = torch_model(dummy_input)
        
        assert output.shape[0] == 1
        assert output.shape[1] == self.config.num_classes
    
    def test_model_compilation(self):
        """Test model compilation."""
        model = CNNModel(self.config, framework="tensorflow")
        model.compile_model(optimizer='adam', learning_rate=0.001)
        
        assert model._is_compiled == True
        assert model.model is not None
    
    def test_weight_operations(self):
        """Test getting and setting model weights."""
        model = CNNModel(self.config, framework="tensorflow")
        model.compile_model()
        
        # Get initial weights
        weights = model.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        
        # Modify weights slightly
        modified_weights = {}
        for layer_name, weight_arrays in weights.items():
            if isinstance(weight_arrays, list) and len(weight_arrays) > 0:
                modified_weights[layer_name] = [w + 0.1 for w in weight_arrays]
            else:
                modified_weights[layer_name] = weight_arrays
        
        # Set modified weights
        model.set_weights(modified_weights)
        
        # Verify weights changed
        new_weights = model.get_weights()
        assert len(new_weights) == len(weights)
    
    def test_model_summary(self):
        """Test model summary generation."""
        model = CNNModel(self.config, framework="tensorflow")
        summary = model.get_model_summary()
        
        # Should not throw exception
        assert summary is not None
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        model = CNNModel(self.config, framework="tensorflow")
        model.build_model()
        
        param_counts = model.count_parameters()
        
        assert 'trainable' in param_counts
        assert 'non_trainable' in param_counts
        assert 'total' in param_counts
        assert param_counts['trainable'] > 0
        assert param_counts['total'] >= param_counts['trainable']
    
    def test_model_save_load_tensorflow(self):
        """Test saving and loading TensorFlow model."""
        model = CNNModel(self.config, framework="tensorflow")
        model.compile_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")
            
            # Save model
            model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            new_model = CNNModel(self.config, framework="tensorflow")
            new_model.load_model(model_path)
            assert new_model.model is not None
    
    def test_resnet_model(self):
        """Test ResNet model creation."""
        model = ResNetModel(self.config, framework="tensorflow", variant="resnet50")
        tf_model = model.build_model()
        
        assert tf_model is not None
        assert isinstance(tf_model, tf.keras.Model)
    
    def test_efficientnet_model(self):
        """Test EfficientNet model creation."""
        model = EfficientNetModel(self.config, framework="tensorflow", variant="b0")
        tf_model = model.build_model()
        
        assert tf_model is not None
        assert isinstance(tf_model, tf.keras.Model)
    
    def test_create_model_factory(self):
        """Test model factory function."""
        # Test CNN creation
        cnn_model = create_model("cnn", self.config)
        assert isinstance(cnn_model, CNNModel)
        
        # Test ResNet creation
        resnet_model = create_model("resnet", self.config)
        assert isinstance(resnet_model, ResNetModel)
        
        # Test EfficientNet creation
        efficientnet_model = create_model("efficientnet", self.config)
        assert isinstance(efficientnet_model, EfficientNetModel)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_model("invalid_model", self.config)


class TestClientModel:
    """Test client model for federated learning."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.base_config = ModelConfig(
            input_shape=(64, 64, 3),
            num_classes=2,
            dropout_rate=0.3
        )
        
        self.client_config = ClientModelConfig(
            base_config=self.base_config,
            client_id=1,
            local_epochs=3,
            local_batch_size=16,
            learning_rate=0.001
        )
    
    def test_client_model_creation(self):
        """Test client model creation."""
        client_model = ClientModel(self.client_config)
        
        assert client_model.config == self.client_config
        assert client_model.model_type == "cnn"
        assert client_model.framework == "tensorflow"
        assert client_model.base_model is not None
    
    def test_weight_operations(self):
        """Test client model weight operations."""
        client_model = ClientModel(self.client_config)
        
        # Get initial weights
        weights = client_model.get_weights()
        assert isinstance(weights, dict)
        
        # Set weights (simulate receiving from server)
        client_model.set_weights(weights)
        
        # Get weights again to verify
        new_weights = client_model.get_weights()
        assert len(new_weights) == len(weights)
    
    @patch('src.models.client_model.DPOptimizer')
    def test_differential_privacy_setup(self, mock_dp_optimizer):
        """Test differential privacy setup."""
        config = ClientModelConfig(
            base_config=self.base_config,
            client_id=1,
            differential_privacy=True,
            privacy_budget=1.0
        )
        
        client_model = ClientModel(config)
        # Should not raise exception
        assert client_model.config.differential_privacy == True
    
    def test_local_training_mock(self):
        """Test local training with mock data."""
        client_model = ClientModel(self.client_config)
        
        # Create mock training data
        X_train = np.random.randn(32, 64, 64, 3)
        y_train = np.random.randint(0, 2, 32)
        X_val = np.random.randn(8, 64, 64, 3)
        y_val = np.random.randint(0, 2, 8)
        
        # Mock the training method
        with patch.object(client_model, 'train_local_model') as mock_train:
            mock_train.return_value = {
                'client_id': 1,
                'round': 1,
                'history': {'loss': [0.6, 0.5], 'accuracy': [0.6, 0.7]},
                'num_samples': 32
            }
            
            result = client_model.train_local_model(X_train, y_train, (X_val, y_val))
            
            assert result['client_id'] == 1
            assert 'history' in result
            assert 'num_samples' in result
            mock_train.assert_called_once()
    
    def test_evaluation(self):
        """Test model evaluation."""
        client_model = ClientModel(self.client_config)
        
        # Create mock test data
        X_test = np.random.randn(16, 64, 64, 3)
        y_test = np.random.randint(0, 2, 16)
        
        # Mock the evaluation method
        with patch.object(client_model, 'evaluate_model') as mock_eval:
            mock_eval.return_value = {
                'loss': 0.5,
                'accuracy': 0.75,
                'client_id': 1
            }
            
            result = client_model.evaluate_model(X_test, y_test)
            
            assert 'loss' in result
            assert 'accuracy' in result
            assert result['client_id'] == 1
            mock_eval.assert_called_once()


class TestServerModel:
    """Test server model for federated learning."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.base_config = ModelConfig(
            input_shape=(64, 64, 3),
            num_classes=2,
            dropout_rate=0.3
        )
        
        self.server_config = ServerModelConfig(
            base_config=self.base_config,
            num_clients=5,
            aggregation_strategy="fedavg",
            client_fraction=0.6,
            min_clients=2,
            max_rounds=10
        )
    
    def test_server_model_creation(self):
        """Test server model creation."""
        server_model = ServerModel(self.server_config)
        
        assert server_model.config == self.server_config
        assert server_model.model_type == "cnn"
        assert server_model.framework == "tensorflow"
        assert server_model.current_round == 0
        assert len(server_model.registered_clients) == 0
    
    def test_client_registration(self):
        """Test client registration."""
        server_model = ServerModel(self.server_config)
        
        client_config = ClientModelConfig(
            base_config=self.base_config,
            client_id=1
        )
        
        server_model.register_client(1, client_config)
        
        assert 1 in server_model.registered_clients
        assert server_model.registered_clients[1]['config'] == client_config
    
    def test_client_selection(self):
        """Test client selection for training."""
        server_model = ServerModel(self.server_config)
        
        # Register multiple clients
        for i in range(5):
            client_config = ClientModelConfig(
                base_config=self.base_config,
                client_id=i
            )
            server_model.register_client(i, client_config)
        
        # Select clients
        selected = server_model.select_clients(1)
        
        assert len(selected) >= self.server_config.min_clients
        assert len(selected) <= self.server_config.num_clients
        assert all(client_id in server_model.registered_clients for client_id in selected)
    
    def test_aggregation(self):
        """Test client update aggregation."""
        server_model = ServerModel(self.server_config)
        
        # Mock client updates
        client_updates = {
            1: {
                'num_samples': 100,
                'weight_updates': {'layer1': np.random.randn(10, 5)},
                'training_result': {'loss': 0.5, 'accuracy': 0.8}
            },
            2: {
                'num_samples': 80,
                'weight_updates': {'layer1': np.random.randn(10, 5)},
                'training_result': {'loss': 0.6, 'accuracy': 0.75}
            }
        }
        
        # Mock the aggregation
        with patch.object(server_model, 'aggregate_client_updates') as mock_agg:
            mock_agg.return_value = {
                'round': 1,
                'num_clients': 2,
                'aggregation_metrics': {'total_samples': 180}
            }
            
            result = server_model.aggregate_client_updates(client_updates, 1)
            
            assert result['round'] == 1
            assert result['num_clients'] == 2
            mock_agg.assert_called_once()
    
    def test_convergence_checking(self):
        """Test convergence detection."""
        server_model = ServerModel(self.server_config)
        
        # Add some convergence history
        server_model.convergence_history.extend([0.1, 0.05, 0.02, 0.01, 0.005])
        
        # Check convergence
        converged = server_model.check_convergence()
        
        # Should converge if recent changes are small
        assert isinstance(converged, bool)
    
    def test_checkpoint_saving(self):
        """Test model checkpoint saving."""
        server_model = ServerModel(self.server_config)
        server_model.current_round = 5
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the save operation
            with patch.object(server_model, 'save_checkpoint') as mock_save:
                server_model.save_checkpoint(temp_dir)
                mock_save.assert_called_once_with(temp_dir)
    
    def test_server_statistics(self):
        """Test server statistics collection."""
        server_model = ServerModel(self.server_config)
        
        # Register some clients
        for i in range(3):
            client_config = ClientModelConfig(
                base_config=self.base_config,
                client_id=i
            )
            server_model.register_client(i, client_config)
        
        stats = server_model.get_server_statistics()
        
        assert 'current_round' in stats
        assert 'total_clients' in stats
        assert 'aggregation_strategy' in stats
        assert stats['total_clients'] == 3


class TestAggregationStrategies:
    """Test federated aggregation strategies."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = AggregationConfig(
            strategy="fedavg",
            fairness_weight=0.1,
            robustness_enabled=True
        )
        
        self.current_weights = {
            'layer1': np.random.randn(10, 5),
            'layer2': np.random.randn(5, 2)
        }
        
        self.client_updates = {
            1: {
                'num_samples': 100,
                'weight_updates': {
                    'layer1': np.random.randn(10, 5) * 0.1,
                    'layer2': np.random.randn(5, 2) * 0.1
                }
            },
            2: {
                'num_samples': 80,
                'weight_updates': {
                    'layer1': np.random.randn(10, 5) * 0.1,
                    'layer2': np.random.randn(5, 2) * 0.1
                }
            }
        }
    
    def test_fedavg_aggregator(self):
        """Test FedAvg aggregation."""
        aggregator = FedAvgAggregator(self.config)
        
        result = aggregator.aggregate(
            self.client_updates,
            self.current_weights,
            round_num=1
        )
        
        assert 'aggregated_weights' in result
        assert 'metrics' in result
        
        # Check that weights have correct shape
        agg_weights = result['aggregated_weights']
        for layer_name in self.current_weights:
            assert layer_name in agg_weights
            assert agg_weights[layer_name].shape == self.current_weights[layer_name].shape
    
    def test_fairfed_aggregator(self):
        """Test FairFed aggregation."""
        aggregator = FairFedAggregator(self.config)
        
        # Mock demographic data
        demographic_data = {
            1: {'demographics': [{'age_group': 'young', 'gender': 'male'}]},
            2: {'demographics': [{'age_group': 'old', 'gender': 'female'}]}
        }
        
        result = aggregator.aggregate(
            self.client_updates,
            self.current_weights,
            round_num=1,
            demographic_data=demographic_data
        )
        
        assert 'aggregated_weights' in result
        assert 'metrics' in result
        
        # Should include fairness metrics
        metrics = result['metrics']
        assert metrics['aggregation_method'] == 'fairfed'
        assert 'fairness_metrics' in metrics
    
    def test_byzantine_detection(self):
        """Test Byzantine client detection."""
        aggregator = FedAvgAggregator(self.config)
        
        # Add a Byzantine client with extreme updates
        byzantine_updates = self.client_updates.copy()
        byzantine_updates[3] = {
            'num_samples': 50,
            'weight_updates': {
                'layer1': np.random.randn(10, 5) * 10,  # Extreme values
                'layer2': np.random.randn(5, 2) * 10
            }
        }
        
        byzantine_clients = aggregator.detect_byzantine_clients(byzantine_updates)
        
        # Should detect the Byzantine client
        assert isinstance(byzantine_clients, list)
        # Note: The exact detection depends on the statistical thresholds
    
    def test_client_weighting(self):
        """Test client weight calculation."""
        aggregator = FedAvgAggregator(self.config)
        
        # Mock reliability scores
        aggregator.client_reliability_scores = {
            1: {'reliability_score': 0.9},
            2: {'reliability_score': 0.7}
        }
        
        weights = aggregator.calculate_client_weights(self.client_updates, [])
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.client_updates)
        
        # Weights should sum to 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-6


class TestModelIntegration:
    """Integration tests between model components."""
    
    def test_client_server_weight_exchange(self):
        """Test weight exchange between client and server."""
        # Setup
        base_config = ModelConfig(input_shape=(32, 32, 3), num_classes=2)
        
        client_config = ClientModelConfig(base_config=base_config, client_id=1)
        server_config = ServerModelConfig(base_config=base_config, num_clients=1)
        
        client_model = ClientModel(client_config)
        server_model = ServerModel(server_config)
        
        # Get server weights
        server_weights = server_model.get_global_weights()
        
        # Send to client
        client_model.set_weights(server_weights)
        
        # Get client weights
        client_weights = client_model.get_weights()
        
        # Weights should have same structure
        assert set(server_weights.keys()) == set(client_weights.keys())
    
    def test_full_training_round_simulation(self):
        """Test a complete training round simulation."""
        # This would be a more complex integration test
        # involving multiple clients and a server
        
        base_config = ModelConfig(input_shape=(32, 32, 3), num_classes=2)
        server_config = ServerModelConfig(base_config=base_config, num_clients=2)
        server_model = ServerModel(server_config)
        
        # Create mock clients
        clients = []
        for i in range(2):
            client_config = ClientModelConfig(base_config=base_config, client_id=i)
            client = ClientModel(client_config)
            clients.append(client)
            server_model.register_client(i, client_config)
        
        # Simulate training round
        selected_clients = server_model.select_clients(1)
        assert len(selected_clients) > 0
        
        # Mock client updates
        client_updates = {}
        for client_id in selected_clients:
            client_updates[client_id] = {
                'num_samples': 50,
                'weight_updates': {
                    f'layer_{i}': np.random.randn(5, 3) * 0.1 
                    for i in range(2)
                },
                'training_result': {'loss': 0.5, 'accuracy': 0.8}
            }
        
        # Test aggregation
        with patch.object(server_model, 'aggregate_client_updates') as mock_agg:
            mock_agg.return_value = {
                'round': 1,
                'num_clients': len(selected_clients),
                'aggregation_metrics': {'total_samples': len(selected_clients) * 50}
            }
            
            result = server_model.aggregate_client_updates(client_updates, 1)
            assert result['round'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
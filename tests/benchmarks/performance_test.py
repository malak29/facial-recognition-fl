import pytest
import time
import psutil
import asyncio
import numpy as np
import memory_profiler
import cProfile
import pstats
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
from pathlib import Path
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.mocks.mock_components import (
    MockFederatedServer, MockFederatedClient, MockModel, MockDataGenerator,
    create_mock_federated_scenario, run_mock_training_simulation
)
from src.models.base_model import ModelConfig
from src.models.cnn_model import CNNModel
from src.federated.aggregation import FedAvgAggregator, AggregationConfig
from src.utils.metrics import PerformanceTracker, calculate_model_metrics
from src.data.preprocessor import ImagePreprocessor
from src.privacy.differential_privacy import DifferentialPrivacyManager


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def start_benchmark(self):
        """Start timing the benchmark."""
        self.start_time = time.time()
        return self
    
    def end_benchmark(self):
        """End timing the benchmark."""
        self.end_time = time.time()
        self.results['execution_time'] = self.end_time - self.start_time
        return self.results['execution_time']
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        self.results['memory_usage_mb'] = memory_used
        self.results['initial_memory_mb'] = initial_memory
        self.results['final_memory_mb'] = final_memory
        
        return result


class ModelPerformanceBenchmarks:
    """Benchmark model operations."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_model_creation(self, model_configs: List[Dict[str, Any]], iterations: int = 10):
        """Benchmark model creation time."""
        benchmark_results = {}
        
        for config_name, config_params in model_configs:
            model_config = ModelConfig(**config_params['model_config'])
            
            times = []
            memory_usage = []
            
            for _ in range(iterations):
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                
                if config_params['model_type'] == 'cnn':
                    model = CNNModel(model_config, **config_params.get('model_kwargs', {}))
                    model.build_model()
                else:
                    model = MockModel(model_config)
                    model.build_model()
                
                end_time = time.time()
                
                final_memory = process.memory_info().rss / 1024 / 1024
                
                times.append(end_time - start_time)
                memory_usage.append(final_memory - initial_memory)
                
                # Clean up
                del model
            
            benchmark_results[config_name] = {
                'avg_creation_time': np.mean(times),
                'std_creation_time': np.std(times),
                'avg_memory_mb': np.mean(memory_usage),
                'std_memory_mb': np.std(memory_usage),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        self.results['model_creation'] = benchmark_results
        return benchmark_results
    
    def benchmark_model_training(self, model_configs: List[Dict[str, Any]], data_sizes: List[int]):
        """Benchmark model training performance."""
        benchmark_results = {}
        
        for config_name, config_params in model_configs:
            model_config = ModelConfig(**config_params['model_config'])
            config_results = {}
            
            for data_size in data_sizes:
                # Generate training data
                X_train = np.random.randn(data_size, *model_config.input_shape)
                y_train = np.random.randint(0, model_config.num_classes, data_size)
                
                # Create model
                if config_params['model_type'] == 'mock':
                    model = MockModel(model_config)
                    model.build_model()
                else:
                    model = CNNModel(model_config, framework="tensorflow")
                    model.compile_model()
                
                # Benchmark training
                start_time = time.time()
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                
                if hasattr(model.model, 'fit'):
                    history = model.model.fit(
                        X_train, y_train,
                        epochs=1,
                        batch_size=32,
                        verbose=0
                    )
                else:
                    # Mock training
                    history = model._mock_fit(X_train, y_train, epochs=1)
                
                end_time = time.time()
                final_memory = process.memory_info().rss / 1024 / 1024
                
                config_results[f'data_size_{data_size}'] = {
                    'training_time': end_time - start_time,
                    'memory_usage_mb': final_memory - initial_memory,
                    'samples_per_second': data_size / (end_time - start_time)
                }
                
                # Clean up
                del model, X_train, y_train
            
            benchmark_results[config_name] = config_results
        
        self.results['model_training'] = benchmark_results
        return benchmark_results
    
    def benchmark_model_inference(self, model_configs: List[Dict[str, Any]], batch_sizes: List[int]):
        """Benchmark model inference performance."""
        benchmark_results = {}
        
        for config_name, config_params in model_configs:
            model_config = ModelConfig(**config_params['model_config'])
            
            # Create and compile model
            if config_params['model_type'] == 'mock':
                model = MockModel(model_config)
                model.build_model()
            else:
                model = CNNModel(model_config, framework="tensorflow")
                model.compile_model()
            
            config_results = {}
            
            for batch_size in batch_sizes:
                # Generate test data
                X_test = np.random.randn(batch_size, *model_config.input_shape)
                
                # Warm-up runs
                for _ in range(3):
                    if hasattr(model.model, 'predict'):
                        _ = model.model.predict(X_test[:1], verbose=0)
                    else:
                        _ = model._mock_predict(X_test[:1])
                
                # Benchmark inference
                times = []
                for _ in range(10):  # Multiple runs for accuracy
                    start_time = time.time()
                    
                    if hasattr(model.model, 'predict'):
                        predictions = model.model.predict(X_test, verbose=0)
                    else:
                        predictions = model._mock_predict(X_test)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                config_results[f'batch_size_{batch_size}'] = {
                    'avg_inference_time': np.mean(times),
                    'std_inference_time': np.std(times),
                    'throughput_samples_per_sec': batch_size / np.mean(times),
                    'latency_per_sample_ms': (np.mean(times) / batch_size) * 1000
                }
            
            benchmark_results[config_name] = config_results
            del model
        
        self.results['model_inference'] = benchmark_results
        return benchmark_results


class FederatedLearningBenchmarks:
    """Benchmark federated learning operations."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_client_server_scalability(self, client_counts: List[int], rounds: int = 3):
        """Benchmark scalability with different numbers of clients."""
        scalability_results = {}
        
        for num_clients in client_counts:
            print(f"Benchmarking with {num_clients} clients...")
            
            # Create federated scenario
            server, clients, datasets = create_mock_federated_scenario(
                num_clients=num_clients,
                data_samples_per_client=[100] * num_clients,
                non_iid=True
            )
            
            # Benchmark training rounds
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Run training simulation
            async def run_benchmark():
                return await run_mock_training_simulation(
                    server, clients, num_rounds=rounds
                )
            
            round_results = asyncio.run(run_benchmark())
            
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            successful_rounds = len([r for r in round_results if 'error' not in r])
            total_participants = sum(
                r.get('num_participants', 0) for r in round_results if 'error' not in r
            )
            
            scalability_results[num_clients] = {
                'total_time': end_time - start_time,
                'memory_usage_mb': final_memory - initial_memory,
                'successful_rounds': successful_rounds,
                'total_participants': total_participants,
                'avg_participants_per_round': total_participants / max(successful_rounds, 1),
                'time_per_round': (end_time - start_time) / max(successful_rounds, 1),
                'throughput_clients_per_second': total_participants / (end_time - start_time)
            }
            
            # Clean up
            del server, clients, datasets
        
        self.results['scalability'] = scalability_results
        return scalability_results
    
    def benchmark_aggregation_algorithms(self, algorithms: List[str], client_counts: List[int]):
        """Benchmark different aggregation algorithms."""
        aggregation_results = {}
        
        for algorithm in algorithms:
            algorithm_results = {}
            
            for num_clients in client_counts:
                # Create mock client updates
                client_updates = {}
                current_weights = {
                    'layer1': np.random.randn(100, 50),
                    'layer2': np.random.randn(50, 10)
                }
                
                for i in range(num_clients):
                    client_updates[i] = {
                        'num_samples': 100 + np.random.randint(-20, 20),
                        'weight_updates': {
                            'layer1': np.random.randn(100, 50) * 0.01,
                            'layer2': np.random.randn(50, 10) * 0.01
                        }
                    }
                
                # Create aggregator
                config = AggregationConfig(strategy=algorithm)
                
                if algorithm == 'fedavg':
                    aggregator = FedAvgAggregator(config)
                else:
                    # Use FedAvg as fallback for unknown algorithms
                    aggregator = FedAvgAggregator(config)
                
                # Benchmark aggregation
                times = []
                memory_usage = []
                
                for _ in range(10):  # Multiple runs
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024
                    
                    start_time = time.time()
                    result = aggregator.aggregate(client_updates, current_weights, round_num=1)
                    end_time = time.time()
                    
                    final_memory = process.memory_info().rss / 1024 / 1024
                    
                    times.append(end_time - start_time)
                    memory_usage.append(final_memory - initial_memory)
                
                algorithm_results[num_clients] = {
                    'avg_aggregation_time': np.mean(times),
                    'std_aggregation_time': np.std(times),
                    'avg_memory_usage_mb': np.mean(memory_usage),
                    'throughput_clients_per_second': num_clients / np.mean(times)
                }
            
            aggregation_results[algorithm] = algorithm_results
        
        self.results['aggregation'] = aggregation_results
        return aggregation_results
    
    def benchmark_communication_overhead(self, message_sizes: List[int], num_messages: int = 100):
        """Benchmark communication overhead."""
        from tests.mocks.mock_components import MockCommunicationProtocol
        
        communication_results = {}
        
        for message_size in message_sizes:
            # Create mock communication protocols
            sender = MockCommunicationProtocol("sender")
            receiver = MockCommunicationProtocol("receiver")
            
            # Start listening
            asyncio.run(sender.start_listening("localhost", 8001))
            asyncio.run(receiver.start_listening("localhost", 8002))
            
            # Generate test payload
            payload = {'data': 'x' * message_size}
            
            async def send_messages():
                times = []
                
                for i in range(num_messages):
                    from src.federated.communication import Message
                    message = Message(
                        message_id=f"test_{i}",
                        message_type="benchmark",
                        sender_id="sender",
                        recipient_id="receiver",
                        payload=payload,
                        timestamp=time.time()
                    )
                    
                    start_time = time.time()
                    success = await sender.send_message(message, "receiver")
                    end_time = time.time()
                    
                    if success:
                        times.append(end_time - start_time)
                
                return times
            
            # Run benchmark
            send_times = asyncio.run(send_messages())
            
            communication_results[message_size] = {
                'avg_send_time': np.mean(send_times),
                'std_send_time': np.std(send_times),
                'successful_sends': len(send_times),
                'throughput_messages_per_second': len(send_times) / sum(send_times) if send_times else 0,
                'bandwidth_bytes_per_second': (message_size * len(send_times)) / sum(send_times) if send_times else 0
            }
            
            # Clean up
            asyncio.run(sender.stop_listening())
            asyncio.run(receiver.stop_listening())
        
        self.results['communication'] = communication_results
        return communication_results


class DataProcessingBenchmarks:
    """Benchmark data processing operations."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_data_preprocessing(self, image_sizes: List[Tuple[int, int, int]], batch_sizes: List[int]):
        """Benchmark image preprocessing performance."""
        preprocessing_results = {}
        
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            normalize=True,
            preserve_aspect_ratio=True
        )
        
        for image_size in image_sizes:
            size_results = {}
            
            for batch_size in batch_sizes:
                # Generate test images
                images = [
                    np.random.randint(0, 255, image_size, dtype=np.uint8)
                    for _ in range(batch_size)
                ]
                
                # Benchmark preprocessing
                times = []
                
                for _ in range(5):  # Multiple runs
                    start_time = time.time()
                    
                    processed_images = [
                        preprocessor.preprocess_image(img) for img in images
                    ]
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                size_results[batch_size] = {
                    'avg_processing_time': np.mean(times),
                    'std_processing_time': np.std(times),
                    'throughput_images_per_second': batch_size / np.mean(times),
                    'time_per_image_ms': (np.mean(times) / batch_size) * 1000
                }
            
            preprocessing_results[f"{image_size[0]}x{image_size[1]}x{image_size[2]}"] = size_results
        
        self.results['preprocessing'] = preprocessing_results
        return preprocessing_results
    
    def benchmark_data_augmentation(self, image_sizes: List[Tuple[int, int, int]], augmentation_intensities: List[str]):
        """Benchmark data augmentation performance."""
        from src.data.augmentation import FairAugmentation
        
        augmentation_results = {}
        
        for intensity in augmentation_intensities:
            augmenter = FairAugmentation(
                augment_probability=1.0,
                intensity=intensity
            )
            
            intensity_results = {}
            
            for image_size in image_sizes:
                # Generate test image
                test_image = np.random.randint(0, 255, image_size, dtype=np.uint8)
                
                # Benchmark augmentation
                times = []
                
                for _ in range(50):  # Multiple augmentations
                    start_time = time.time()
                    augmented = augmenter.augment_image(test_image)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                
                intensity_results[f"{image_size[0]}x{image_size[1]}x{image_size[2]}"] = {
                    'avg_augmentation_time': np.mean(times),
                    'std_augmentation_time': np.std(times),
                    'throughput_images_per_second': 1.0 / np.mean(times)
                }
            
            augmentation_results[intensity] = intensity_results
        
        self.results['augmentation'] = augmentation_results
        return augmentation_results


class PrivacyBenchmarks:
    """Benchmark privacy-preserving operations."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_differential_privacy(self, data_sizes: List[int], noise_levels: List[float]):
        """Benchmark differential privacy mechanisms."""
        privacy_results = {}
        
        for noise_level in noise_levels:
            privacy_manager = DifferentialPrivacyManager(
                total_epsilon=1.0,
                total_delta=1e-5,
                noise_mechanism="gaussian"
            )
            
            noise_results = {}
            
            for data_size in data_sizes:
                # Generate test data
                test_data = np.random
                test_data = np.random.randn(data_size, 100)
                
                # Benchmark privatization
                times = []
                memory_usage = []
                
                for _ in range(10):
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024
                    
                    start_time = time.time()
                    
                    privatized_data = privacy_manager.privatize_data(
                        test_data,
                        sensitivity=1.0,
                        epsilon_fraction=0.1
                    )
                    
                    end_time = time.time()
                    final_memory = process.memory_info().rss / 1024 / 1024
                    
                    times.append(end_time - start_time)
                    memory_usage.append(final_memory - initial_memory)
                
                noise_results[data_size] = {
                    'avg_privatization_time': np.mean(times),
                    'std_privatization_time': np.std(times),
                    'avg_memory_usage_mb': np.mean(memory_usage),
                    'throughput_samples_per_second': data_size / np.mean(times),
                    'privacy_overhead_factor': np.mean(times) / (data_size * 1e-6)  # Baseline comparison
                }
            
            privacy_results[noise_level] = noise_results
        
        self.results['differential_privacy'] = privacy_results
        return privacy_results
    
    def benchmark_secure_aggregation(self, num_clients_list: List[int], weight_sizes: List[int]):
        """Benchmark secure aggregation performance."""
        secure_agg_results = {}
        
        for num_clients in num_clients_list:
            client_results = {}
            
            for weight_size in weight_sizes:
                # Generate client weight updates
                client_updates = {}
                for i in range(num_clients):
                    client_updates[i] = {
                        'weight_updates': {
                            f'layer_{j}': np.random.randn(weight_size, weight_size // 2) * 0.01
                            for j in range(3)
                        },
                        'num_samples': 100
                    }
                
                # Benchmark secure aggregation simulation
                times = []
                
                for _ in range(5):
                    start_time = time.time()
                    
                    # Simulate secure aggregation steps
                    # 1. Add noise to updates (simplified)
                    for client_id in client_updates:
                        for layer_name, weights in client_updates[client_id]['weight_updates'].items():
                            noise = np.random.normal(0, 0.001, weights.shape)
                            client_updates[client_id]['weight_updates'][layer_name] += noise
                    
                    # 2. Aggregate (simplified FedAvg)
                    total_samples = sum(update['num_samples'] for update in client_updates.values())
                    aggregated = {}
                    
                    for layer_name in client_updates[0]['weight_updates']:
                        weighted_sum = np.zeros_like(client_updates[0]['weight_updates'][layer_name])
                        
                        for client_id, update in client_updates.items():
                            weight = update['num_samples'] / total_samples
                            weighted_sum += weight * update['weight_updates'][layer_name]
                        
                        aggregated[layer_name] = weighted_sum
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                client_results[weight_size] = {
                    'avg_secure_aggregation_time': np.mean(times),
                    'std_secure_aggregation_time': np.std(times),
                    'throughput_clients_per_second': num_clients / np.mean(times),
                    'overhead_per_client_ms': (np.mean(times) / num_clients) * 1000
                }
            
            secure_agg_results[num_clients] = client_results
        
        self.results['secure_aggregation'] = secure_agg_results
        return secure_agg_results


class StressTests:
    """Stress tests for system limits and edge cases."""
    
    def __init__(self):
        self.results = {}
    
    def stress_test_concurrent_clients(self, max_clients: int = 100, step: int = 10):
        """Stress test with increasing number of concurrent clients."""
        stress_results = {}
        
        for num_clients in range(step, max_clients + 1, step):
            print(f"Stress testing with {num_clients} concurrent clients...")
            
            try:
                # Create server
                server = MockFederatedServer(max_clients=num_clients + 10)
                clients = []
                
                # Create clients
                start_time = time.time()
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                initial_cpu = psutil.cpu_percent()
                
                for i in range(num_clients):
                    client = MockFederatedClient(f"stress_client_{i}")
                    clients.append(client)
                
                # Register all clients concurrently
                async def register_clients():
                    tasks = []
                    for client in clients:
                        task = server.register_client(client)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    successful_registrations = sum(1 for r in results if r is True)
                    return successful_registrations
                
                successful_registrations = asyncio.run(register_clients())
                
                # Execute one training round
                round_result = asyncio.run(server.execute_training_round())
                
                end_time = time.time()
                final_memory = process.memory_info().rss / 1024 / 1024
                final_cpu = psutil.cpu_percent()
                
                # Record results
                stress_results[num_clients] = {
                    'successful_registrations': successful_registrations,
                    'registration_success_rate': successful_registrations / num_clients,
                    'setup_time': end_time - start_time,
                    'memory_usage_mb': final_memory - initial_memory,
                    'cpu_usage_increase': final_cpu - initial_cpu,
                    'training_successful': 'error' not in round_result,
                    'participants': round_result.get('num_participants', 0) if 'error' not in round_result else 0
                }
                
                # Clean up
                del server, clients
                
            except Exception as e:
                stress_results[num_clients] = {
                    'error': str(e),
                    'test_failed': True
                }
                break  # Stop if we hit system limits
        
        self.results['concurrent_clients'] = stress_results
        return stress_results
    
    def stress_test_large_models(self, model_sizes: List[Dict[str, Any]]):
        """Stress test with increasingly large models."""
        model_stress_results = {}
        
        for size_config in model_sizes:
            config_name = size_config['name']
            model_config = ModelConfig(**size_config['config'])
            
            try:
                start_time = time.time()
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                
                # Create large model
                model = MockModel(model_config)
                built_model = model.build_model()
                
                # Get model size
                weights = model.get_weights()
                total_parameters = sum(np.prod(w.shape) for w in weights.values())
                
                # Test training with large batch
                large_X = np.random.randn(100, *model_config.input_shape)
                large_y = np.random.randint(0, model_config.num_classes, 100)
                
                training_start = time.time()
                training_result = model._mock_fit(large_X, large_y, epochs=1)
                training_end = time.time()
                
                end_time = time.time()
                final_memory = process.memory_info().rss / 1024 / 1024
                
                model_stress_results[config_name] = {
                    'total_parameters': total_parameters,
                    'model_creation_time': end_time - start_time,
                    'memory_usage_mb': final_memory - initial_memory,
                    'training_time': training_end - training_start,
                    'memory_per_parameter': (final_memory - initial_memory) / total_parameters * 1024 * 1024,  # bytes per parameter
                    'success': True
                }
                
                del model, built_model, large_X, large_y
                
            except Exception as e:
                model_stress_results[config_name] = {
                    'error': str(e),
                    'success': False
                }
        
        self.results['large_models'] = model_stress_results
        return model_stress_results
    
    def stress_test_memory_limits(self, data_sizes: List[int]):
        """Test system behavior under memory pressure."""
        memory_stress_results = {}
        
        for data_size in data_sizes:
            try:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                available_memory = psutil.virtual_memory().available / 1024 / 1024
                
                # Allocate large arrays
                start_time = time.time()
                large_arrays = []
                
                # Allocate in chunks to monitor memory usage
                chunk_size = data_size // 10
                for i in range(10):
                    chunk = np.random.randn(chunk_size, 100)
                    large_arrays.append(chunk)
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    if current_memory > available_memory * 0.8:  # Stop at 80% memory usage
                        break
                
                end_time = time.time()
                final_memory = process.memory_info().rss / 1024 / 1024
                
                # Test operations on large data
                if large_arrays:
                    combined_array = np.concatenate(large_arrays, axis=0)
                    
                    # Simulate model operations
                    operation_start = time.time()
                    result = np.dot(combined_array, combined_array.T)
                    operation_end = time.time()
                    
                    memory_stress_results[data_size] = {
                        'allocation_time': end_time - start_time,
                        'memory_allocated_mb': final_memory - initial_memory,
                        'chunks_allocated': len(large_arrays),
                        'operation_time': operation_end - operation_start,
                        'peak_memory_mb': final_memory,
                        'success': True
                    }
                    
                    del large_arrays, combined_array, result
                else:
                    memory_stress_results[data_size] = {
                        'error': 'Insufficient memory for allocation',
                        'success': False
                    }
                
            except MemoryError as e:
                memory_stress_results[data_size] = {
                    'error': f'MemoryError: {str(e)}',
                    'success': False
                }
            except Exception as e:
                memory_stress_results[data_size] = {
                    'error': f'Unexpected error: {str(e)}',
                    'success': False
                }
        
        self.results['memory_limits'] = memory_stress_results
        return memory_stress_results


class ProfiledBenchmark:
    """Benchmark with detailed profiling information."""
    
    def __init__(self, name: str):
        self.name = name
        self.profiler = None
        self.profile_stats = None
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a function execution."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            self.profiler.disable()
        
        # Get profiling statistics
        self.profile_stats = pstats.Stats(self.profiler)
        self.profile_stats.sort_stats('cumulative')
        
        return result
    
    def get_top_functions(self, n: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """Get top N functions by cumulative time."""
        if not self.profile_stats:
            return []
        
        top_functions = []
        for func_info in self.profile_stats.stats.items()[:n]:
            func_key, (cc, nc, tt, ct, callers) = func_info
            
            top_functions.append((
                f"{func_key[0]}:{func_key[1]}({func_key[2]})",
                {
                    'call_count': cc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'time_per_call': tt / cc if cc > 0 else 0
                }
            ))
        
        return top_functions
    
    def save_profile_report(self, filename: str):
        """Save detailed profile report to file."""
        if self.profile_stats:
            with open(filename, 'w') as f:
                self.profile_stats.print_stats(file=f)


@memory_profiler.profile
def memory_intensive_federated_round():
    """Memory profiled federated learning round."""
    # Create a scenario with moderate complexity
    server, clients, datasets = create_mock_federated_scenario(
        num_clients=5,
        data_samples_per_client=[200] * 5,
        non_iid=True
    )
    
    # Run training round
    async def run_round():
        return await run_mock_training_simulation(server, clients, num_rounds=2)
    
    results = asyncio.run(run_round())
    return results


class BenchmarkVisualizer:
    """Visualize benchmark results."""
    
    def __init__(self):
        self.figures = []
    
    def plot_scalability_results(self, scalability_data: Dict[int, Dict[str, Any]], save_path: str = None):
        """Plot scalability benchmark results."""
        client_counts = list(scalability_data.keys())
        times = [data['total_time'] for data in scalability_data.values()]
        memory_usage = [data['memory_usage_mb'] for data in scalability_data.values()]
        throughput = [data['throughput_clients_per_second'] for data in scalability_data.values()]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Execution time
        ax1.plot(client_counts, times, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clients')
        ax1.set_ylabel('Total Time (seconds)')
        ax1.set_title('Execution Time vs Number of Clients')
        ax1.grid(True, alpha=0.3)
        
        # Memory usage
        ax2.plot(client_counts, memory_usage, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clients')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Number of Clients')
        ax2.grid(True, alpha=0.3)
        
        # Throughput
        ax3.plot(client_counts, throughput, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Clients')
        ax3.set_ylabel('Throughput (clients/second)')
        ax3.set_title('Throughput vs Number of Clients')
        ax3.grid(True, alpha=0.3)
        
        # Efficiency (clients per MB)
        efficiency = [c / m if m > 0 else 0 for c, m in zip(client_counts, memory_usage)]
        ax4.plot(client_counts, efficiency, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Clients')
        ax4.set_ylabel('Efficiency (clients/MB)')
        ax4.set_title('Memory Efficiency vs Number of Clients')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_performance_comparison(self, comparison_data: Dict[str, Dict], metric: str, save_path: str = None):
        """Plot performance comparison across different configurations."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        configs = list(comparison_data.keys())
        x_pos = np.arange(len(configs))
        
        values = [data.get(metric, 0) for data in comparison_data.values()]
        colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def create_benchmark_dashboard(self, all_results: Dict[str, Any], save_dir: str = "benchmark_results"):
        """Create comprehensive benchmark dashboard."""
        Path(save_dir).mkdir(exist_ok=True)
        
        dashboard_plots = []
        
        # Scalability plots
        if 'scalability' in all_results:
            scalability_plot = self.plot_scalability_results(
                all_results['scalability'],
                save_path=f"{save_dir}/scalability_analysis.png"
            )
            dashboard_plots.append(scalability_plot)
        
        # Model performance comparison
        if 'model_performance' in all_results:
            for metric in ['avg_creation_time', 'avg_memory_mb']:
                if any(metric in data for data in all_results['model_performance'].values()):
                    comparison_plot = self.plot_performance_comparison(
                        all_results['model_performance'],
                        metric,
                        save_path=f"{save_dir}/model_{metric}_comparison.png"
                    )
                    dashboard_plots.append(comparison_plot)
        
        # Generate summary report
        self.generate_summary_report(all_results, f"{save_dir}/benchmark_summary.txt")
        
        return dashboard_plots
    
    def generate_summary_report(self, all_results: Dict[str, Any], save_path: str):
        """Generate comprehensive benchmark summary report."""
        with open(save_path, 'w') as f:
            f.write("FEDERATED LEARNING SYSTEM BENCHMARK REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System information
            f.write("SYSTEM INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"CPU Count: {psutil.cpu_count()}\n")
            f.write(f"Total Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB\n")
            f.write(f"Available Memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB\n\n")
            
            # Benchmark results summary
            for benchmark_type, results in all_results.items():
                f.write(f"{benchmark_type.upper()} RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                if isinstance(results, dict):
                    for key, value in results.items():
                        f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{results}\n")
                
                f.write("\n")
            
            # Performance recommendations
            f.write("PERFORMANCE RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("• Monitor memory usage when scaling beyond tested client counts\n")
            f.write("• Consider model compression for large model deployments\n")
            f.write("• Implement connection pooling for high client scenarios\n")
            f.write("• Use asynchronous communication for better throughput\n")
            f.write("• Regular profiling recommended for production deployments\n")


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self):
        self.model_benchmarks = ModelPerformanceBenchmarks()
        self.federated_benchmarks = FederatedLearningBenchmarks()
        self.data_benchmarks = DataProcessingBenchmarks()
        self.privacy_benchmarks = PrivacyBenchmarks()
        self.stress_tests = StressTests()
        self.visualizer = BenchmarkVisualizer()
        
        self.all_results = {}
    
    def run_model_benchmarks(self):
        """Run all model performance benchmarks."""
        print("Running model performance benchmarks...")
        
        # Define test configurations
        model_configs = [
            ("small_cnn", {
                'model_type': 'mock',
                'model_config': {
                    'input_shape': (32, 32, 3),
                    'num_classes': 5,
                    'dropout_rate': 0.3
                }
            }),
            ("medium_cnn", {
                'model_type': 'mock',
                'model_config': {
                    'input_shape': (64, 64, 3),
                    'num_classes': 10,
                    'dropout_rate': 0.4
                }
            }),
            ("large_cnn", {
                'model_type': 'mock',
                'model_config': {
                    'input_shape': (128, 128, 3),
                    'num_classes': 20,
                    'dropout_rate': 0.5
                }
            })
        ]
        
        # Run benchmarks
        creation_results = self.model_benchmarks.benchmark_model_creation(model_configs)
        training_results = self.model_benchmarks.benchmark_model_training(model_configs, [50, 100, 200])
        inference_results = self.model_benchmarks.benchmark_model_inference(model_configs, [1, 8, 32, 64])
        
        self.all_results['model_creation'] = creation_results
        self.all_results['model_training'] = training_results
        self.all_results['model_inference'] = inference_results
    
    def run_federated_benchmarks(self):
        """Run federated learning benchmarks."""
        print("Running federated learning benchmarks...")
        
        scalability_results = self.federated_benchmarks.benchmark_client_server_scalability([2, 5, 10, 20, 50])
        aggregation_results = self.federated_benchmarks.benchmark_aggregation_algorithms(['fedavg'], [2, 5, 10, 20])
        communication_results = self.federated_benchmarks.benchmark_communication_overhead([100, 1000, 10000])
        
        self.all_results['scalability'] = scalability_results
        self.all_results['aggregation'] = aggregation_results
        self.all_results['communication'] = communication_results
    
    def run_data_benchmarks(self):
        """Run data processing benchmarks."""
        print("Running data processing benchmarks...")
        
        preprocessing_results = self.data_benchmarks.benchmark_data_preprocessing(
            [(64, 64, 3), (128, 128, 3), (224, 224, 3)],
            [10, 50, 100]
        )
        augmentation_results = self.data_benchmarks.benchmark_data_augmentation(
            [(64, 64, 3), (128, 128, 3)],
            ['light', 'medium', 'heavy']
        )
        
        self.all_results['preprocessing'] = preprocessing_results
        self.all_results['augmentation'] = augmentation_results
    
    def run_privacy_benchmarks(self):
        """Run privacy preservation benchmarks."""
        print("Running privacy benchmarks...")
        
        dp_results = self.privacy_benchmarks.benchmark_differential_privacy([100, 500, 1000], [0.1, 0.5, 1.0])
        secure_agg_results = self.privacy_benchmarks.benchmark_secure_aggregation([2, 5, 10], [50, 100, 200])
        
        self.all_results['differential_privacy'] = dp_results
        self.all_results['secure_aggregation'] = secure_agg_results
    
    def run_stress_tests(self):
        """Run stress tests."""
        print("Running stress tests...")
        
        client_stress = self.stress_tests.stress_test_concurrent_clients(max_clients=50, step=10)
        
        model_stress = self.stress_tests.stress_test_large_models([
            {
                'name': 'small',
                'config': {'input_shape': (32, 32, 3), 'num_classes': 5, 'dropout_rate': 0.3}
            },
            {
                'name': 'medium',
                'config': {'input_shape': (64, 64, 3), 'num_classes': 10, 'dropout_rate': 0.4}
            },
            {
                'name': 'large',
                'config': {'input_shape': (128, 128, 3), 'num_classes': 20, 'dropout_rate': 0.5}
            }
        ])
        
        memory_stress = self.stress_tests.stress_test_memory_limits([1000, 5000, 10000])
        
        self.all_results['client_stress'] = client_stress
        self.all_results['model_stress'] = model_stress
        self.all_results['memory_stress'] = memory_stress
    
    def run_profiled_benchmark(self):
        """Run profiled benchmark for detailed analysis."""
        print("Running profiled benchmark...")
        
        profiled_benchmark = ProfiledBenchmark("federated_training_round")
        
        # Profile a federated training round
        result = profiled_benchmark.profile_function(memory_intensive_federated_round)
        
        # Get top functions
        top_functions = profiled_benchmark.get_top_functions(10)
        
        # Save profile report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.prof', delete=False) as f:
            profile_path = f.name
        
        profiled_benchmark.save_profile_report(profile_path)
        
        self.all_results['profiling'] = {
            'top_functions': top_functions,
            'profile_report_path': profile_path
        }
    
    def run_all_benchmarks(self, save_results: bool = True, results_dir: str = "benchmark_results"):
        """Run all benchmarks and generate reports."""
        start_time = time.time()
        
        print("Starting comprehensive benchmark suite...")
        print("This may take several minutes to complete.\n")
        
        # Run all benchmark categories
        try:
            self.run_model_benchmarks()
        except Exception as e:
            print(f"Model benchmarks failed: {e}")
        
        try:
            self.run_federated_benchmarks()
        except Exception as e:
            print(f"Federated benchmarks failed: {e}")
        
        try:
            self.run_data_benchmarks()
        except Exception as e:
            print(f"Data benchmarks failed: {e}")
        
        try:
            self.run_privacy_benchmarks()
        except Exception as e:
            print(f"Privacy benchmarks failed: {e}")
        
        try:
            self.run_stress_tests()
        except Exception as e:
            print(f"Stress tests failed: {e}")
        
        try:
            self.run_profiled_benchmark()
        except Exception as e:
            print(f"Profiled benchmark failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nBenchmark suite completed in {total_time:.2f} seconds")
        
        # Generate reports and visualizations
        if save_results:
            Path(results_dir).mkdir(exist_ok=True)
            
            # Save raw results
            import json
            with open(f"{results_dir}/raw_results.json", 'w') as f:
                json.dump(self.all_results, f, indent=2, default=str)
            
            # Generate visualizations
            self.visualizer.create_benchmark_dashboard(self.all_results, results_dir)
            
            print(f"Results saved to {results_dir}/")
        
        return self.all_results


# Test classes for pytest integration
class TestPerformanceBenchmarks:
    """Performance tests that can be run with pytest."""
    
    def test_model_creation_performance(self):
        """Test that model creation is within acceptable time limits."""
        config = ModelConfig(input_shape=(32, 32, 3), num_classes=5)
        
        start_time = time.time()
        model = MockModel(config)
        model.build_model()
        end_time = time.time()
        
        creation_time = end_time - start_time
        
        # Should create model in under 1 second
        assert creation_time < 1.0, f"Model creation took {creation_time:.2f}s, expected < 1.0s"
    
    def test_aggregation_performance(self):
        """Test that aggregation performance is acceptable."""
        config = AggregationConfig(strategy="fedavg")
        aggregator = FedAvgAggregator(config)
        
        # Create mock updates for 10 clients
        current_weights = {'layer1': np.random.randn(100, 50)}
        client_updates = {
            i: {
                'num_samples': 100,
                'weight_updates': {'layer1': np.random.randn(100, 50) * 0.01}
            }
            for i in range(10)
        }
        
        start_time = time.time()
        result = aggregator.aggregate(client_updates, current_weights, round_num=1)
        end_time = time.time()
        
        aggregation_time = end_time - start_time
        
        # Should aggregate in under 0.5 seconds for 10 clients
        assert aggregation_time < 0.5, f"Aggregation took {aggregation_time:.2f}s, expected < 0.5s"
    
    def test_memory_usage_reasonable(self):
        """Test that memory usage stays within reasonable bounds."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create federated scenario
        server, clients, datasets = create_mock_federated_scenario(num_clients=5)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should use less than 200MB for 5 clients
        assert memory_increase < 200, f"Memory usage increased by {memory_increase:.2f}MB, expected < 200MB"
        
        # Clean up
        del server, clients, datasets


if __name__ == "__main__":
    # Run comprehensive benchmarks
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks(save_results=True)
    
    print("\nBenchmark Summary:")
    print("=" * 50)
    
    # Print key metrics
    if 'scalability' in results:
        max_clients = max(results['scalability'].keys())
        max_throughput = max(data['throughput_clients_per_second'] for data in results['scalability'].values())
        print(f"Max clients tested: {max_clients}")
        print(f"Peak throughput: {max_throughput:.2f} clients/second")
    
    if 'model_creation' in results:
        fastest_creation = min(
            data['avg_creation_time'] 
            for model_results in results['model_creation'].values()
            for data in [model_results] if isinstance(model_results, dict) and 'avg_creation_time' in model_results
        )
        print(f"Fastest model creation: {fastest_creation:.4f} seconds")
    
    print("\nFor detailed results, check the benchmark_results/ directory")
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass
import numpy as np
import time
from collections import defaultdict, deque

from .communication import CommunicationManager, CommunicationConfig, Message
from .aggregation import create_aggregator, AggregationConfig
from ..models.server_model import ServerModel, ServerModelConfig
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FederatedServerConfig:
    """Configuration for federated learning server."""
    server_id: str = "federated_server"
    communication_config: CommunicationConfig = None
    model_config: ServerModelConfig = None
    aggregation_config: AggregationConfig = None
    min_clients: int = 2
    max_clients: int = 100
    client_selection_fraction: float = 0.3
    round_timeout: int = 300  # seconds
    max_rounds: int = 100
    convergence_threshold: float = 0.001
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    enable_client_validation: bool = True
    client_health_timeout: int = 120  # seconds


class FederatedServer:
    """High-level federated learning server with complete FL orchestration."""
    
    def __init__(self, config: FederatedServerConfig):
        """
        Initialize federated learning server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.server_id = config.server_id
        
        # Initialize components
        self.comm_manager = CommunicationManager(
            config.server_id,
            config.communication_config
        )
        
        self.model = ServerModel(
            config.model_config,
            model_type="cnn",  # Can be configurable
            framework="tensorflow"
        )
        
        self.aggregator = create_aggregator(
            config.aggregation_config.strategy,
            config.aggregation_config
        )
        
        # Client management
        self.registered_clients = {}  # client_id -> client_info
        self.active_clients = set()   # currently connected clients
        self.selected_clients = set() # clients selected for current round
        self.client_results = {}      # round results from clients
        self.client_health = {}       # client health monitoring
        
        # Training state
        self.current_round = 0
        self.is_training = False
        self.round_start_time = None
        self.training_complete = False
        
        # Performance tracking
        self.round_history = []
        self.global_performance_history = []
        self.convergence_history = deque(maxlen=10)
        
        # Background tasks
        self.round_manager_task = None
        self.health_monitor_task = None
        
        # Register message handlers
        self._register_message_handlers()
        
        logger.info(f"Federated server {self.server_id} initialized")
    
    def _register_message_handlers(self) -> None:
        """Register handlers for different message types."""
        handlers = {
            'client_registration': self._handle_client_registration,
            'client_disconnect': self._handle_client_disconnect,
            'training_accepted': self._handle_training_accepted,
            'training_declined': self._handle_training_declined,
            'training_results': self._handle_training_results,
            'training_error': self._handle_training_error,
            'evaluation_results': self._handle_evaluation_results,
            'architecture_ack': self._handle_architecture_ack,
            'heartbeat': self._handle_client_heartbeat
        }
        
        for message_type, handler in handlers.items():
            self.comm_manager.register_message_handler(message_type, handler)
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        Start the federated server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        # Start communication manager
        await self.comm_manager.start(host, port)
        
        # Start background tasks
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        # Initialize global model
        logger.info("Initializing global model...")
        # Model initialization would happen here
        
        logger.info(f"Federated server started on {host}:{port}")
        logger.info(f"Server ready to accept clients (min: {self.config.min_clients}, max: {self.config.max_clients})")
    
    async def stop(self) -> None:
        """Stop the federated server."""
        logger.info("Shutting down federated server...")
        
        # Notify all clients of shutdown
        if self.registered_clients:
            await self.comm_manager.broadcast_message('server_shutdown', {'reason': 'normal_shutdown'})
        
        # Cancel background tasks
        if self.round_manager_task:
            self.round_manager_task.cancel()
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        
        # Save final checkpoint
        if self.config.save_checkpoints:
            await self._save_checkpoint()
        
        # Stop communication
        await self.comm_manager.stop()
        
        logger.info("Federated server stopped")
    
    async def start_training(self, test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Start federated training process.
        
        Args:
            test_data: Optional global test data for evaluation
        """
        if self.is_training:
            logger.warning("Training already in progress")
            return
        
        if len(self.active_clients) < self.config.min_clients:
            logger.error(f"Insufficient clients: {len(self.active_clients)} < {self.config.min_clients}")
            return
        
        self.is_training = True
        self.training_complete = False
        self.current_round = 0
        
        logger.info(f"Starting federated training with {len(self.active_clients)} clients")
        
        # Start round manager
        self.round_manager_task = asyncio.create_task(
            self._training_loop(test_data)
        )
    
    async def stop_training(self) -> None:
        """Stop federated training process."""
        self.is_training = False
        
        if self.round_manager_task:
            self.round_manager_task.cancel()
        
        # Notify clients
        await self.comm_manager.broadcast_message('training_stopped', {'reason': 'manual_stop'})
        
        logger.info("Federated training stopped")
    
    async def _training_loop(self, test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """Main federated training loop."""
        try:
            while (self.is_training and 
                   self.current_round < self.config.max_rounds and 
                   not self.training_complete):
                
                self.current_round += 1
                logger.info(f"Starting training round {self.current_round}")
                
                # Execute training round
                round_successful = await self._execute_training_round()
                
                if not round_successful:
                    logger.warning(f"Round {self.current_round} failed, continuing...")
                    continue
                
                # Global model evaluation
                if test_data:
                    await self._evaluate_global_model(test_data)
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Training converged at round {self.current_round}")
                    self.training_complete = True
                    break
                
                # Save checkpoint
                if (self.config.save_checkpoints and 
                    self.current_round % self.config.checkpoint_frequency == 0):
                    await self._save_checkpoint()
                
                # Brief pause between rounds
                await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            logger.info("Training loop cancelled")
        except Exception as e:
            logger.error(f"Error in training loop: {e}")
        finally:
            self.is_training = False
            await self._training_complete_cleanup()
    
    async def _execute_training_round(self) -> bool:
        """Execute a single training round."""
        try:
            # Select clients for this round
            selected_clients = self._select_clients_for_round()
            if not selected_clients:
                logger.error("No clients selected for training round")
                return False
            
            self.selected_clients = set(selected_clients)
            self.client_results.clear()
            self.round_start_time = time.time()
            
            # Broadcast round start
            await self.comm_manager.broadcast_message(
                'round_start',
                {'round_number': self.current_round},
                round_number=self.current_round
            )
            
            # Send global weights to selected clients
            await self._distribute_global_weights(selected_clients)
            
            # Send training requests
            await self._send_training_requests(selected_clients)
            
            # Wait for client responses
            success = await self._wait_for_client_results()
            if not success:
                return False
            
            # Aggregate results
            aggregation_result = await self._aggregate_client_results()
            
            # Update global model
            self.model.aggregate_client_updates(
                self.client_results,
                self.current_round
            )
            
            # Broadcast round completion
            await self.comm_manager.broadcast_message(
                'round_end',
                {
                    'round_number': self.current_round,
                    'participants': list(selected_clients),
                    'aggregation_metrics': aggregation_result
                },
                round_number=self.current_round
            )
            
            # Record round statistics
            round_duration = time.time() - self.round_start_time
            self.round_history.append({
                'round': self.current_round,
                'participants': list(selected_clients),
                'duration': round_duration,
                'aggregation_result': aggregation_result,
                'timestamp': time.time()
            })
            
            logger.info(f"Round {self.current_round} completed in {round_duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error in training round {self.current_round}: {e}")
            return False
    
    def _select_clients_for_round(self) -> List[str]:
        """Select clients for the current training round."""
        available_clients = list(self.active_clients)
        
        if not available_clients:
            return []
        
        # Calculate number of clients to select
        num_select = max(
            self.config.min_clients,
            int(len(available_clients) * self.config.client_selection_fraction)
        )
        num_select = min(num_select, len(available_clients))
        
        # Simple random selection (could be enhanced with sophisticated strategies)
        selected = np.random.choice(
            available_clients,
            size=num_select,
            replace=False
        ).tolist()
        
        logger.info(f"Selected {len(selected)} clients for round {self.current_round}: {selected}")
        return selected
    
    async def _distribute_global_weights(self, clients: List[str]) -> None:
        """Distribute global model weights to selected clients."""
        global_weights = self.model.get_global_weights()
        
        weights_payload = {
            'weights': global_weights,
            'round_number': self.current_round
        }
        
        # Send to all selected clients
        for client_id in clients:
            await self.comm_manager.send_message(
                client_id,
                'global_weights',
                weights_payload,
                round_number=self.current_round
            )
        
        logger.debug(f"Global weights distributed to {len(clients)} clients")
    
    async def _send_training_requests(self, clients: List[str]) -> None:
        """Send training requests to selected clients."""
        training_payload = {
            'round_number': self.current_round,
            'timeout': self.config.round_timeout,
            'local_epochs': 5,  # Could be configurable
            'batch_size': 32   # Could be configurable
        }
        
        for client_id in clients:
            await self.comm_manager.send_message(
                client_id,
                'training_request',
                training_payload,
                round_number=self.current_round
            )
        
        logger.info(f"Training requests sent to {len(clients)} clients")
    
    async def _wait_for_client_results(self) -> bool:
        """Wait for training results from selected clients."""
        start_time = time.time()
        timeout = self.config.round_timeout
        
        while time.time() - start_time < timeout:
            # Check if we have results from enough clients
            if len(self.client_results) >= len(self.selected_clients) * 0.7:  # 70% threshold
                logger.info(f"Received results from {len(self.client_results)}/{len(self.selected_clients)} clients")
                return True
            
            await asyncio.sleep(1)
        
        # Timeout reached
        missing_clients = self.selected_clients - set(self.client_results.keys())
        logger.warning(f"Round timeout. Missing results from: {missing_clients}")
        
        # Check if we have minimum results
        if len(self.client_results) >= self.config.min_clients:
            logger.info(f"Proceeding with {len(self.client_results)} client results")
            return True
        
        logger.error("Insufficient client results for aggregation")
        return False
    
    async def _aggregate_client_results(self) -> Dict[str, Any]:
        """Aggregate client training results."""
        logger.info(f"Aggregating results from {len(self.client_results)} clients")
        
        # Run aggregation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        aggregation_result = await loop.run_in_executor(
            None,
            self._aggregate_sync,
            self.client_results
        )
        
        return aggregation_result
    
    def _aggregate_sync(self, client_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous aggregation (runs in executor)."""
        # Extract global weights and client updates
        current_weights = self.model.get_global_weights()
        
        # Prepare client updates for aggregator
        aggregation_input = {}
        for client_id, result in client_results.items():
            aggregation_input[client_id] = result['training_result']
        
        # Perform aggregation
        return self.aggregator.aggregate(
            aggregation_input,
            current_weights,
            self.current_round
        )
    
    async def _evaluate_global_model(self, test_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Evaluate global model on test data."""
        X_test, y_test = test_data
        
        logger.info("Evaluating global model...")
        
        # Run evaluation in executor
        loop = asyncio.get_event_loop()
        eval_result = await loop.run_in_executor(
            None,
            self._evaluate_sync,
            X_test, y_test
        )
        
        # Store evaluation results
        self.global_performance_history.append({
            'round': self.current_round,
            'evaluation': eval_result,
            'timestamp': time.time()
        })
        
        # Track convergence
        if 'accuracy' in eval_result:
            self.convergence_history.append(eval_result['accuracy'])
        
        logger.info(f"Global model evaluation - Round {self.current_round}: "
                   f"Loss: {eval_result.get('loss', 'N/A'):.4f}, "
                   f"Accuracy: {eval_result.get('accuracy', 'N/A'):.4f}")
    
    def _evaluate_sync(self, X_test, y_test):
        """Synchronous model evaluation (runs in executor)."""
        return self.model.evaluate_global_model(X_test, y_test)
    
    def _check_convergence(self) -> bool:
        """Check if training has converged."""
        if len(self.convergence_history) < 3:
            return False
        
        # Check if recent accuracy changes are below threshold
        recent_accuracies = list(self.convergence_history)[-3:]
        accuracy_changes = [abs(recent_accuracies[i] - recent_accuracies[i-1]) 
                           for i in range(1, len(recent_accuracies))]
        
        return all(change < self.config.convergence_threshold for change in accuracy_changes)
    
    async def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_dir = f"checkpoints/round_{self.current_round}"
        
        try:
            # Save model checkpoint
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.model.save_checkpoint,
                checkpoint_dir
            )
            
            logger.info(f"Checkpoint saved for round {self.current_round}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def _training_complete_cleanup(self) -> None:
        """Cleanup after training completion."""
        # Notify clients of training completion
        completion_payload = {
            'total_rounds': self.current_round,
            'reason': 'completed' if self.training_complete else 'stopped'
        }
        
        await self.comm_manager.broadcast_message(
            'training_complete',
            completion_payload
        )
        
        # Final checkpoint
        if self.config.save_checkpoints:
            await self._save_checkpoint()
        
        logger.info(f"Training completed after {self.current_round} rounds")
    
    async def _health_monitor_loop(self) -> None:
        """Monitor client health continuously."""
        while True:
            try:
                current_time = time.time()
                
                # Check client health
                inactive_clients = []
                for client_id in self.active_clients.copy():
                    last_seen = self.client_health.get(client_id, {}).get('last_seen', 0)
                    
                    if current_time - last_seen > self.config.client_health_timeout:
                        inactive_clients.append(client_id)
                
                # Remove inactive clients
                for client_id in inactive_clients:
                    self._remove_client(client_id, "health_timeout")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)  # Longer backoff on error
    
    def _remove_client(self, client_id: str, reason: str) -> None:
        """Remove client from active list."""
        if client_id in self.active_clients:
            self.active_clients.remove(client_id)
            logger.info(f"Client {client_id} removed: {reason}")
        
        # Remove from selected clients if currently selected
        self.selected_clients.discard(client_id)
        
        # Clean up client health tracking
        self.client_health.pop(client_id, None)
    
    # Message handlers
    async def _handle_client_registration(self, message: Message) -> None:
        """Handle client registration request."""
        try:
            client_id = message.payload['client_id']
            capabilities = message.payload.get('capabilities', {})
            client_config = message.payload.get('client_config', {})
            
            # Register client
            self.registered_clients[client_id] = {
                'client_id': client_id,
                'capabilities': capabilities,
                'config': client_config,
                'registered_at': time.time()
            }
            
            self.active_clients.add(client_id)
            self.comm_manager.register_peer(client_id, f"client_{client_id}")
            
            # Initialize client health tracking
            self.client_health[client_id] = {
                'last_seen': time.time(),
                'heartbeat_count': 0
            }
            
            logger.info(f"Client {client_id} registered. Total active clients: {len(self.active_clients)}")
            
            # Send registration acknowledgment
            await self.comm_manager.send_message(
                client_id,
                'registration_ack',
                {
                    'status': 'accepted',
                    'server_id': self.server_id,
                    'server_capabilities': {
                        'privacy_aware': True,
                        'fairness_aware': True
                    }
                }
            )
            
            # Send model architecture if available
            if self.model:
                await self._send_model_architecture(client_id)
            
        except Exception as e:
            logger.error(f"Error handling client registration: {e}")
    
    async def _send_model_architecture(self, client_id: str) -> None:
        """Send model architecture to client."""
        model_config = {
            'model_type': 'cnn',  # This should come from actual model config
            'framework': 'tensorflow',
            'architecture': 'custom'  # Model architecture details
        }
        
        await self.comm_manager.send_message(
            client_id,
            'model_architecture',
            {'model_config': model_config}
        )
    
    async def _handle_client_disconnect(self, message: Message) -> None:
        """Handle client disconnect notification."""
        client_id = message.payload['client_id']
        reason = message.payload.get('reason', 'unknown')
        
        self._remove_client(client_id, f"disconnect_{reason}")
    
    async def _handle_training_accepted(self, message: Message) -> None:
        """Handle training acceptance from client."""
        client_id = message.sender_id
        logger.debug(f"Training accepted by client {client_id}")
    
    async def _handle_training_declined(self, message: Message) -> None:
        """Handle training decline from client."""
        client_id = message.sender_id
        reason = message.payload.get('reason', 'unknown')
        
        logger.warning(f"Training declined by client {client_id}: {reason}")
        
        # Remove from selected clients
        self.selected_clients.discard(client_id)
    
    async def _handle_training_results(self, message: Message) -> None:
        """Handle training results from client."""
        client_id = message.sender_id
        training_result = message.payload['training_result']
        
        # Store client results
        self.client_results[client_id] = {
            'training_result': training_result,
            'data_samples': message.payload.get('data_samples', 0),
            'timestamp': message.payload.get('timestamp', time.time())
        }
        
        logger.debug(f"Training results received from client {client_id}")
    
    async def _handle_training_error(self, message: Message) -> None:
        """Handle training error from client."""
        client_id = message.sender_id
        error = message.payload.get('error', 'unknown')
        
        logger.error(f"Training error from client {client_id}: {error}")
        
        # Remove from selected clients
        self.selected_clients.discard(client_id)
    
    async def _handle_evaluation_results(self, message: Message) -> None:
        """Handle evaluation results from client."""
        client_id = message.sender_id
        eval_result = message.payload['evaluation_result']
        
        logger.debug(f"Evaluation results received from client {client_id}: {eval_result}")
    
    async def _handle_architecture_ack(self, message: Message) -> None:
        """Handle model architecture acknowledgment."""
        client_id = message.sender_id
        status = message.payload.get('status', 'unknown')
        
        logger.debug(f"Model architecture ack from client {client_id}: {status}")
    
    async def _handle_client_heartbeat(self, message: Message) -> None:
        """Handle heartbeat from client."""
        client_id = message.sender_id
        
        # Update client health
        if client_id in self.client_health:
            self.client_health[client_id]['last_seen'] = time.time()
            self.client_health[client_id]['heartbeat_count'] += 1
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status."""
        return {
            'server_id': self.server_id,
            'is_training': self.is_training,
            'current_round': self.current_round,
            'training_complete': self.training_complete,
            'client_counts': {
                'registered': len(self.registered_clients),
                'active': len(self.active_clients),
                'selected': len(self.selected_clients)
            },
            'active_clients': list(self.active_clients),
            'performance_history_length': len(self.global_performance_history),
            'round_history_length': len(self.round_history),
            'communication_stats': self.comm_manager.get_communication_statistics(),
            'model_stats': self.model.get_server_statistics() if self.model else None
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training progress summary."""
        if not self.global_performance_history:
            return {'message': 'No training data available'}
        
        # Calculate performance trends
        accuracies = [p['evaluation'].get('accuracy', 0) for p in self.global_performance_history]
        losses = [p['evaluation'].get('loss', float('inf')) for p in self.global_performance_history]
        
        summary = {
            'total_rounds': len(self.global_performance_history),
            'current_round': self.current_round,
            'training_active': self.is_training,
            'performance_trend': {
                'accuracy': {
                    'initial': accuracies[0] if accuracies else 0,
                    'latest': accuracies[-1] if accuracies else 0,
                    'best': max(accuracies) if accuracies else 0,
                    'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
                },
                'loss': {
                    'initial': losses[0] if losses else 0,
                    'latest': losses[-1] if losses else 0,
                    'best': min(losses) if losses else 0
                }
            },
            'convergence_status': {
                'converged': self.training_complete,
                'recent_changes': list(self.convergence_history)[-3:] if self.convergence_history else []
            },
            'participation_stats': {
                'total_client_participations': sum(len(r['participants']) for r in self.round_history),
                'avg_clients_per_round': np.mean([len(r['participants']) for r in self.round_history]) if self.round_history else 0
            }
        }
        
        return summary
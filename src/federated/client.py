import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import time

from .communication import CommunicationManager, CommunicationConfig, Message
from ..models.client_model import ClientModel, ClientModelConfig
from ..bias_mitigation.fairness_metrics import FairnessMetricsCalculator
from ..privacy.differential_privacy import DifferentialPrivacyManager
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FederatedClientConfig:
    """Configuration for federated learning client."""
    client_id: str
    server_endpoint: str
    communication_config: CommunicationConfig
    model_config: ClientModelConfig
    privacy_config: Optional[Dict[str, Any]] = None
    fairness_config: Optional[Dict[str, Any]] = None
    auto_participate: bool = True
    min_data_samples: int = 10
    max_rounds: int = 100
    local_evaluation: bool = True


class FederatedClient:
    """High-level federated learning client with full FL lifecycle management."""
    
    def __init__(self, config: FederatedClientConfig):
        """
        Initialize federated learning client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.client_id = config.client_id
        
        # Initialize components
        self.comm_manager = CommunicationManager(
            config.client_id, 
            config.communication_config
        )
        
        self.model = None  # Will be initialized when model architecture is received
        self.privacy_manager = None
        self.fairness_calculator = None
        
        # Training state
        self.current_round = 0
        self.is_participating = False
        self.training_data = None
        self.validation_data = None
        self.demographic_data = None
        
        # Performance tracking
        self.performance_history = []
        self.communication_metrics = {}
        
        # Setup privacy if enabled
        if config.privacy_config:
            self.privacy_manager = DifferentialPrivacyManager(
                **config.privacy_config
            )
        
        # Setup fairness evaluation if enabled
        if config.fairness_config:
            self.fairness_calculator = FairnessMetricsCalculator(
                **config.fairness_config
            )
        
        # Register message handlers
        self._register_message_handlers()
        
        logger.info(f"Federated client {self.client_id} initialized")
    
    def _register_message_handlers(self) -> None:
        """Register handlers for different message types."""
        handlers = {
            'model_architecture': self._handle_model_architecture,
            'global_weights': self._handle_global_weights,
            'training_request': self._handle_training_request,
            'evaluation_request': self._handle_evaluation_request,
            'aggregation_complete': self._handle_aggregation_complete,
            'round_start': self._handle_round_start,
            'round_end': self._handle_round_end,
            'server_shutdown': self._handle_server_shutdown
        }
        
        for message_type, handler in handlers.items():
            self.comm_manager.register_message_handler(message_type, handler)
    
    async def start(self, host: str = "0.0.0.0", port: int = 0) -> None:
        """
        Start the federated client.
        
        Args:
            host: Host to bind to
            port: Port to listen on (0 for auto-assign)
        """
        # Start communication manager
        await self.comm_manager.start(host, port)
        
        # Register with server
        await self._register_with_server()
        
        # Start client lifecycle
        if self.config.auto_participate:
            asyncio.create_task(self._client_lifecycle())
        
        logger.info(f"Federated client {self.client_id} started")
    
    async def stop(self) -> None:
        """Stop the federated client."""
        # Notify server of disconnection
        await self._notify_server_disconnect()
        
        # Stop communication
        await self.comm_manager.stop()
        
        logger.info(f"Federated client {self.client_id} stopped")
    
    def set_training_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        demographic_data: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Set client training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            demographic_data: Demographic information for fairness
        """
        self.training_data = (X_train, y_train)
        if X_val is not None and y_val is not None:
            self.validation_data = (X_val, y_val)
        
        self.demographic_data = demographic_data
        
        logger.info(f"Training data set: {len(X_train)} samples")
        
        # Check minimum data requirement
        if len(X_train) < self.config.min_data_samples:
            logger.warning(f"Training data size ({len(X_train)}) below minimum ({self.config.min_data_samples})")
    
    async def _register_with_server(self) -> None:
        """Register this client with the federated server."""
        registration_payload = {
            'client_id': self.client_id,
            'capabilities': {
                'privacy_enabled': self.privacy_manager is not None,
                'fairness_enabled': self.fairness_calculator is not None,
                'data_samples': len(self.training_data[0]) if self.training_data else 0
            },
            'client_config': {
                'auto_participate': self.config.auto_participate,
                'local_evaluation': self.config.local_evaluation
            }
        }
        
        # Register server as peer
        self.comm_manager.register_peer('server', self.config.server_endpoint)
        
        success = await self.comm_manager.send_message(
            'server',
            'client_registration',
            registration_payload
        )
        
        if success:
            logger.info("Successfully registered with server")
        else:
            logger.error("Failed to register with server")
            raise RuntimeError("Server registration failed")
    
    async def _notify_server_disconnect(self) -> None:
        """Notify server of client disconnection."""
        disconnect_payload = {
            'client_id': self.client_id,
            'reason': 'normal_shutdown'
        }
        
        await self.comm_manager.send_message(
            'server',
            'client_disconnect',
            disconnect_payload
        )
    
    async def _client_lifecycle(self) -> None:
        """Main client lifecycle loop."""
        while self.current_round < self.config.max_rounds:
            try:
                # Wait for server instructions
                await asyncio.sleep(1)  # Prevent tight loop
                
                # Check if we should participate in current round
                if self.is_participating and self.training_data:
                    await self._participate_in_round()
                
            except Exception as e:
                logger.error(f"Error in client lifecycle: {e}")
                await asyncio.sleep(5)  # Backoff on error
    
    async def _participate_in_round(self) -> None:
        """Participate in the current training round."""
        if not self.model:
            logger.warning("No model available for training")
            return
        
        logger.info(f"Participating in round {self.current_round}")
        
        try:
            # Perform local training
            training_result = await self._perform_local_training()
            
            # Send results to server
            await self._send_training_results(training_result)
            
            # Local evaluation if enabled
            if self.config.local_evaluation:
                await self._perform_local_evaluation()
            
        except Exception as e:
            logger.error(f"Error in round participation: {e}")
            # Notify server of training failure
            await self.comm_manager.send_message(
                'server',
                'training_error',
                {'client_id': self.client_id, 'error': str(e), 'round': self.current_round}
            )
    
    async def _perform_local_training(self) -> Dict[str, Any]:
        """Perform local model training."""
        if not self.model or not self.training_data:
            raise ValueError("Model or training data not available")
        
        X_train, y_train = self.training_data
        validation_data = self.validation_data
        
        logger.info(f"Starting local training with {len(X_train)} samples")
        
        # Run training in executor to avoid blocking
        loop = asyncio.get_event_loop()
        training_result = await loop.run_in_executor(
            None,
            self._train_model_sync,
            X_train, y_train, validation_data
        )
        
        # Apply privacy if enabled
        if self.privacy_manager:
            training_result = self._apply_privacy_to_updates(training_result)
        
        logger.info(f"Local training completed. Loss: {training_result.get('final_loss', 'N/A')}")
        return training_result
    
    def _train_model_sync(self, X_train, y_train, validation_data):
        """Synchronous model training (runs in executor)."""
        return self.model.train_local_model(
            X_train,
            y_train,
            validation_data=validation_data,
            demographic_data=self.demographic_data
        )
    
    def _apply_privacy_to_updates(self, training_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy mechanisms to training results."""
        if 'weight_updates' in training_result:
            # Privatize weight updates
            weight_updates = training_result['weight_updates']
            privatized_updates = {}
            
            for layer_name, updates in weight_updates.items():
                # Apply differential privacy
                privatized_updates[layer_name] = self.privacy_manager.privatize_data(
                    updates,
                    sensitivity=1.0,  # Assuming gradient clipping
                    epsilon_fraction=0.1
                )
            
            training_result['weight_updates'] = privatized_updates
            training_result['privacy_applied'] = True
        
        return training_result
    
    async def _send_training_results(self, training_result: Dict[str, Any]) -> None:
        """Send training results to server."""
        results_payload = {
            'client_id': self.client_id,
            'round_number': self.current_round,
            'training_result': training_result,
            'data_samples': len(self.training_data[0]) if self.training_data else 0,
            'timestamp': time.time()
        }
        
        success = await self.comm_manager.send_message(
            'server',
            'training_results',
            results_payload,
            round_number=self.current_round
        )
        
        if success:
            logger.info(f"Training results sent for round {self.current_round}")
        else:
            logger.error(f"Failed to send training results for round {self.current_round}")
    
    async def _perform_local_evaluation(self) -> None:
        """Perform local model evaluation."""
        if not self.model or not self.validation_data:
            return
        
        X_val, y_val = self.validation_data
        
        # Run evaluation in executor
        loop = asyncio.get_event_loop()
        eval_result = await loop.run_in_executor(
            None,
            self._evaluate_model_sync,
            X_val, y_val
        )
        
        # Calculate fairness metrics if enabled
        if self.fairness_calculator and self.demographic_data:
            y_pred = self.model.base_model.model.predict(X_val)
            fairness_result = self.fairness_calculator.calculate_all_metrics(
                y_val, y_pred, None, self.demographic_data
            )
            eval_result['fairness_metrics'] = fairness_result
        
        # Store evaluation results
        self.performance_history.append({
            'round': self.current_round,
            'evaluation': eval_result,
            'timestamp': time.time()
        })
        
        logger.info(f"Local evaluation - Accuracy: {eval_result.get('accuracy', 'N/A'):.4f}")
    
    def _evaluate_model_sync(self, X_val, y_val):
        """Synchronous model evaluation (runs in executor)."""
        return self.model.evaluate_model(X_val, y_val, self.demographic_data)
    
    # Message handlers
    async def _handle_model_architecture(self, message: Message) -> None:
        """Handle model architecture message from server."""
        try:
            model_config = message.payload['model_config']
            
            # Initialize client model
            self.model = ClientModel(
                self.config.model_config,
                model_type=model_config.get('model_type', 'cnn'),
                framework=model_config.get('framework', 'tensorflow')
            )
            
            logger.info("Model architecture received and initialized")
            
            # Acknowledge receipt
            await self.comm_manager.send_message(
                'server',
                'architecture_ack',
                {'client_id': self.client_id, 'status': 'ready'}
            )
            
        except Exception as e:
            logger.error(f"Error handling model architecture: {e}")
    
    async def _handle_global_weights(self, message: Message) -> None:
        """Handle global model weights from server."""
        try:
            if not self.model:
                logger.warning("Received global weights but no model initialized")
                return
            
            # Extract and set weights
            global_weights = message.payload['weights']
            self.model.set_weights(global_weights)
            
            logger.info(f"Global weights updated for round {message.round_number}")
            
        except Exception as e:
            logger.error(f"Error handling global weights: {e}")
    
    async def _handle_training_request(self, message: Message) -> None:
        """Handle training request from server."""
        try:
            self.current_round = message.round_number
            self.is_participating = True
            
            # Check if we have sufficient data
            if not self.training_data or len(self.training_data[0]) < self.config.min_data_samples:
                # Decline participation
                await self.comm_manager.send_message(
                    'server',
                    'training_declined',
                    {
                        'client_id': self.client_id,
                        'reason': 'insufficient_data',
                        'round_number': self.current_round
                    }
                )
                self.is_participating = False
                return
            
            # Acknowledge training request
            await self.comm_manager.send_message(
                'server',
                'training_accepted',
                {
                    'client_id': self.client_id,
                    'round_number': self.current_round,
                    'data_samples': len(self.training_data[0])
                }
            )
            
            logger.info(f"Training request accepted for round {self.current_round}")
            
        except Exception as e:
            logger.error(f"Error handling training request: {e}")
    
    async def _handle_evaluation_request(self, message: Message) -> None:
        """Handle evaluation request from server."""
        try:
            if not self.model or not self.validation_data:
                return
            
            # Perform evaluation
            await self._perform_local_evaluation()
            
            # Send evaluation results
            latest_eval = self.performance_history[-1] if self.performance_history else None
            if latest_eval:
                await self.comm_manager.send_message(
                    'server',
                    'evaluation_results',
                    {
                        'client_id': self.client_id,
                        'evaluation_result': latest_eval['evaluation'],
                        'round_number': message.round_number
                    }
                )
            
        except Exception as e:
            logger.error(f"Error handling evaluation request: {e}")
    
    async def _handle_aggregation_complete(self, message: Message) -> None:
        """Handle aggregation complete notification."""
        aggregation_info = message.payload
        logger.info(f"Aggregation complete for round {message.round_number}")
        logger.debug(f"Aggregation info: {aggregation_info}")
        
        # Reset participation flag
        self.is_participating = False
    
    async def _handle_round_start(self, message: Message) -> None:
        """Handle round start notification."""
        self.current_round = message.round_number
        logger.info(f"Round {self.current_round} started")
    
    async def _handle_round_end(self, message: Message) -> None:
        """Handle round end notification."""
        round_info = message.payload
        logger.info(f"Round {message.round_number} ended")
        
        # Store round statistics
        self.communication_metrics[f'round_{message.round_number}'] = round_info
    
    async def _handle_server_shutdown(self, message: Message) -> None:
        """Handle server shutdown notification."""
        logger.info("Server shutdown notification received")
        # Gracefully stop the client
        await self.stop()
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get current client status and metrics."""
        status = {
            'client_id': self.client_id,
            'current_round': self.current_round,
            'is_participating': self.is_participating,
            'model_initialized': self.model is not None,
            'data_available': self.training_data is not None,
            'data_samples': len(self.training_data[0]) if self.training_data else 0,
            'privacy_enabled': self.privacy_manager is not None,
            'fairness_enabled': self.fairness_calculator is not None,
            'performance_history_length': len(self.performance_history),
            'communication_stats': self.comm_manager.get_communication_statistics()
        }
        
        # Add latest performance if available
        if self.performance_history:
            latest_performance = self.performance_history[-1]
            status['latest_performance'] = latest_performance
        
        # Add privacy status if enabled
        if self.privacy_manager:
            status['privacy_status'] = self.privacy_manager.get_privacy_report()
        
        return status
    
    async def manually_participate(self) -> bool:
        """Manually trigger participation in current round."""
        if not self.is_participating:
            logger.warning("Not currently participating in any round")
            return False
        
        if not self.training_data:
            logger.error("No training data available")
            return False
        
        try:
            await self._participate_in_round()
            return True
        except Exception as e:
            logger.error(f"Manual participation failed: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of client performance across rounds."""
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        # Calculate performance statistics
        accuracies = [p['evaluation'].get('accuracy', 0) for p in self.performance_history]
        losses = [p['evaluation'].get('loss', float('inf')) for p in self.performance_history]
        
        summary = {
            'total_rounds': len(self.performance_history),
            'accuracy_stats': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'latest': accuracies[-1] if accuracies else 0
            },
            'loss_stats': {
                'mean': np.mean(losses),
                'std': np.std(losses),
                'min': np.min(losses),
                'max': np.max(losses),
                'latest': losses[-1] if losses else 0
            },
            'trend': 'improving' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'stable'
        }
        
        # Add fairness summary if available
        fairness_scores = []
        for perf in self.performance_history:
            if 'fairness_metrics' in perf['evaluation']:
                overall_fairness = perf['evaluation']['fairness_metrics'].get('overall_fairness')
                if overall_fairness:
                    fairness_scores.append(overall_fairness.overall_score)
        
        if fairness_scores:
            summary['fairness_stats'] = {
                'mean': np.mean(fairness_scores),
                'std': np.std(fairness_scores),
                'min': np.min(fairness_scores),
                'max': np.max(fairness_scores),
                'latest': fairness_scores[-1]
            }
        
        return summary
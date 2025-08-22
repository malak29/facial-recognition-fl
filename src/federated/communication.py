import logging
import asyncio
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
import aiohttp
import websockets
import ssl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
from cryptography.fernet import Fernet
import threading
from queue import Queue, Empty
import pickle
import base64
import gzip

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Federated learning message structure."""
    message_id: str
    message_type: str  # model_update, aggregation_result, control, heartbeat
    sender_id: str
    recipient_id: str
    payload: Dict[str, Any]
    timestamp: float
    round_number: int = 0
    encrypted: bool = False
    compressed: bool = False
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(**data)
    
    def calculate_checksum(self) -> str:
        """Calculate message checksum for integrity."""
        payload_str = json.dumps(self.payload, sort_keys=True)
        message_data = f"{self.message_id}{self.message_type}{self.sender_id}{payload_str}"
        return hashlib.sha256(message_data.encode()).hexdigest()
    
    def verify_checksum(self) -> bool:
        """Verify message integrity."""
        if self.checksum is None:
            return False
        return self.checksum == self.calculate_checksum()


@dataclass
class CommunicationConfig:
    """Communication configuration."""
    protocol: str = "http"  # http, websocket, grpc
    encryption_enabled: bool = True
    compression_enabled: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3
    heartbeat_interval: int = 60
    message_queue_size: int = 1000
    batch_messages: bool = True
    batch_size: int = 10
    authentication_enabled: bool = True


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols."""
    
    @abstractmethod
    async def send_message(self, message: Message, endpoint: str) -> bool:
        """Send message to endpoint."""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Message]:
        """Receive message from the protocol."""
        pass
    
    @abstractmethod
    async def start_listening(self, host: str, port: int) -> None:
        """Start listening for incoming messages."""
        pass
    
    @abstractmethod
    async def stop_listening(self) -> None:
        """Stop listening for messages."""
        pass


class HTTPProtocol(CommunicationProtocol):
    """HTTP-based communication protocol."""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.session = None
        self.server = None
        self.message_handlers = {}
        self.ssl_context = None
        
        if config.encryption_enabled:
            self._setup_ssl()
    
    def _setup_ssl(self):
        """Setup SSL context for secure communication."""
        self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        # In production, you would configure proper certificates
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    async def send_message(self, message: Message, endpoint: str) -> bool:
        """Send HTTP message."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Prepare message
        message_data = self._prepare_message(message)
        
        try:
            async with self.session.post(
                endpoint,
                json=message_data,
                ssl=self.ssl_context
            ) as response:
                if response.status == 200:
                    logger.debug(f"Message {message.message_id} sent successfully")
                    return True
                else:
                    logger.error(f"Failed to send message: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def receive_message(self) -> Optional[Message]:
        """HTTP doesn't have built-in receive - handled by server."""
        # This would be implemented as part of the HTTP server handler
        return None
    
    async def start_listening(self, host: str, port: int) -> None:
        """Start HTTP server for receiving messages."""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_post('/message', self._handle_http_message)
        app.router.add_get('/health', self._health_check)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port, ssl_context=self.ssl_context)
        await site.start()
        
        logger.info(f"HTTP server listening on {host}:{port}")
    
    async def stop_listening(self) -> None:
        """Stop HTTP server."""
        if self.session:
            await self.session.close()
        logger.info("HTTP server stopped")
    
    async def _handle_http_message(self, request) -> web.Response:
        """Handle incoming HTTP message."""
        try:
            data = await request.json()
            message = self._parse_message(data)
            
            if message and message.verify_checksum():
                # Process message through registered handlers
                await self._process_message(message)
                return web.Response(status=200, text="Message received")
            else:
                return web.Response(status=400, text="Invalid message")
                
        except Exception as e:
            logger.error(f"Error handling HTTP message: {e}")
            return web.Response(status=500, text="Internal server error")
    
    async def _health_check(self, request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy", "timestamp": time.time()})
    
    def _prepare_message(self, message: Message) -> Dict[str, Any]:
        """Prepare message for transmission."""
        # Set checksum
        message.checksum = message.calculate_checksum()
        
        # Convert to dict
        message_data = message.to_dict()
        
        # Compress if enabled
        if self.config.compression_enabled:
            message_data = self._compress_message(message_data)
        
        # Encrypt if enabled
        if self.config.encryption_enabled:
            message_data = self._encrypt_message(message_data)
        
        return message_data
    
    def _parse_message(self, data: Dict[str, Any]) -> Optional[Message]:
        """Parse received message data."""
        try:
            # Decrypt if needed
            if self.config.encryption_enabled and data.get('encrypted'):
                data = self._decrypt_message(data)
            
            # Decompress if needed
            if self.config.compression_enabled and data.get('compressed'):
                data = self._decompress_message(data)
            
            # Create message object
            message = Message.from_dict(data)
            return message
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None
    
    def _compress_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress message data."""
        try:
            json_str = json.dumps(data)
            compressed = gzip.compress(json_str.encode())
            encoded = base64.b64encode(compressed).decode()
            
            return {
                'compressed_data': encoded,
                'compressed': True,
                'original_size': len(json_str),
                'compressed_size': len(encoded)
            }
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data
    
    def _decompress_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress message data."""
        try:
            if 'compressed_data' in data:
                compressed = base64.b64decode(data['compressed_data'])
                decompressed = gzip.decompress(compressed)
                return json.loads(decompressed.decode())
            return data
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data
    
    def _encrypt_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt message data."""
        # Simplified encryption - in production use proper key management
        key = Fernet.generate_key()
        f = Fernet(key)
        
        try:
            json_str = json.dumps(data)
            encrypted = f.encrypt(json_str.encode())
            encoded = base64.b64encode(encrypted).decode()
            
            return {
                'encrypted_data': encoded,
                'encryption_key': key.decode(),  # Don't do this in production!
                'encrypted': True
            }
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def _decrypt_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt message data."""
        try:
            if 'encrypted_data' in data and 'encryption_key' in data:
                f = Fernet(data['encryption_key'].encode())
                encrypted = base64.b64decode(data['encrypted_data'])
                decrypted = f.decrypt(encrypted)
                return json.loads(decrypted.decode())
            return data
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return data
    
    async def _process_message(self, message: Message) -> None:
        """Process received message through handlers."""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register message handler for specific message type."""
        self.message_handlers[message_type] = handler


class WebSocketProtocol(CommunicationProtocol):
    """WebSocket-based communication protocol."""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.connections = {}
        self.message_queue = Queue(maxsize=config.message_queue_size)
        self.server = None
        self.client_websockets = {}
        
    async def send_message(self, message: Message, endpoint: str) -> bool:
        """Send WebSocket message."""
        try:
            if endpoint not in self.client_websockets:
                # Create new WebSocket connection
                websocket = await websockets.connect(endpoint)
                self.client_websockets[endpoint] = websocket
            
            websocket = self.client_websockets[endpoint]
            message_data = self._prepare_message(message)
            
            await websocket.send(json.dumps(message_data))
            logger.debug(f"WebSocket message {message.message_id} sent")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            # Remove failed connection
            if endpoint in self.client_websockets:
                del self.client_websockets[endpoint]
            return False
    
    async def receive_message(self) -> Optional[Message]:
        """Receive message from queue."""
        try:
            message_data = self.message_queue.get_nowait()
            return self._parse_message(message_data)
        except Empty:
            return None
    
    async def start_listening(self, host: str, port: int) -> None:
        """Start WebSocket server."""
        async def handler(websocket, path):
            try:
                client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
                self.connections[client_id] = websocket
                logger.info(f"WebSocket client connected: {client_id}")
                
                async for message_raw in websocket:
                    try:
                        message_data = json.loads(message_raw)
                        if not self.message_queue.full():
                            self.message_queue.put(message_data)
                        else:
                            logger.warning("Message queue full, dropping message")
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON received")
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"WebSocket client disconnected: {client_id}")
            except Exception as e:
                logger.error(f"WebSocket handler error: {e}")
            finally:
                if client_id in self.connections:
                    del self.connections[client_id]
        
        self.server = await websockets.serve(handler, host, port)
        logger.info(f"WebSocket server listening on {host}:{port}")
    
    async def stop_listening(self) -> None:
        """Stop WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close client connections
        for websocket in self.client_websockets.values():
            await websocket.close()
        self.client_websockets.clear()
        
        logger.info("WebSocket server stopped")
    
    def _prepare_message(self, message: Message) -> Dict[str, Any]:
        """Prepare message for WebSocket transmission."""
        message.checksum = message.calculate_checksum()
        return message.to_dict()
    
    def _parse_message(self, data: Dict[str, Any]) -> Optional[Message]:
        """Parse WebSocket message."""
        try:
            return Message.from_dict(data)
        except Exception as e:
            logger.error(f"Error parsing WebSocket message: {e}")
            return None


class CommunicationManager:
    """High-level communication manager for federated learning."""
    
    def __init__(self, node_id: str, config: CommunicationConfig):
        """
        Initialize communication manager.
        
        Args:
            node_id: Unique identifier for this node
            config: Communication configuration
        """
        self.node_id = node_id
        self.config = config
        self.protocol = self._create_protocol()
        self.message_handlers = {}
        self.peer_endpoints = {}
        self.heartbeat_task = None
        self.message_processor_task = None
        self.statistics = {
            'messages_sent': 0,
            'messages_received': 0,
            'send_failures': 0,
            'connection_errors': 0,
            'last_heartbeat': None
        }
        
        logger.info(f"Communication manager initialized for node {node_id}")
    
    def _create_protocol(self) -> CommunicationProtocol:
        """Create communication protocol based on configuration."""
        if self.config.protocol == "http":
            return HTTPProtocol(self.config)
        elif self.config.protocol == "websocket":
            return WebSocketProtocol(self.config)
        else:
            raise ValueError(f"Unsupported protocol: {self.config.protocol}")
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start communication manager."""
        # Start protocol listening
        await self.protocol.start_listening(host, port)
        
        # Register default handlers
        self._register_default_handlers()
        
        # Start background tasks
        if self.config.heartbeat_interval > 0:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        self.message_processor_task = asyncio.create_task(self._message_processor_loop())
        
        logger.info(f"Communication manager started on {host}:{port}")
    
    async def stop(self) -> None:
        """Stop communication manager."""
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        if self.message_processor_task:
            self.message_processor_task.cancel()
        
        # Stop protocol
        await self.protocol.stop_listening()
        
        logger.info("Communication manager stopped")
    
    def register_peer(self, peer_id: str, endpoint: str) -> None:
        """Register a peer node."""
        self.peer_endpoints[peer_id] = endpoint
        logger.info(f"Registered peer {peer_id} at {endpoint}")
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register handler for specific message type."""
        self.message_handlers[message_type] = handler
        if hasattr(self.protocol, 'register_message_handler'):
            self.protocol.register_message_handler(message_type, handler)
    
    async def send_message(
        self,
        recipient_id: str,
        message_type: str,
        payload: Dict[str, Any],
        round_number: int = 0
    ) -> bool:
        """
        Send message to a peer.
        
        Args:
            recipient_id: ID of recipient node
            message_type: Type of message
            payload: Message payload
            round_number: Training round number
        
        Returns:
            True if message sent successfully
        """
        if recipient_id not in self.peer_endpoints:
            logger.error(f"Unknown recipient: {recipient_id}")
            return False
        
        # Create message
        message = Message(
            message_id=self._generate_message_id(),
            message_type=message_type,
            sender_id=self.node_id,
            recipient_id=recipient_id,
            payload=payload,
            timestamp=time.time(),
            round_number=round_number
        )
        
        endpoint = self.peer_endpoints[recipient_id]
        success = await self._send_with_retry(message, endpoint)
        
        if success:
            self.statistics['messages_sent'] += 1
            logger.debug(f"Message sent to {recipient_id}: {message_type}")
        else:
            self.statistics['send_failures'] += 1
            logger.error(f"Failed to send message to {recipient_id}")
        
        return success
    
    async def broadcast_message(
        self,
        message_type: str,
        payload: Dict[str, Any],
        round_number: int = 0,
        exclude: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Broadcast message to all peers.
        
        Args:
            message_type: Type of message
            payload: Message payload
            round_number: Training round number
            exclude: Peer IDs to exclude from broadcast
        
        Returns:
            Dictionary mapping peer IDs to success status
        """
        exclude = exclude or []
        results = {}
        
        tasks = []
        peer_ids = []
        
        for peer_id in self.peer_endpoints:
            if peer_id not in exclude:
                task = self.send_message(peer_id, message_type, payload, round_number)
                tasks.append(task)
                peer_ids.append(peer_id)
        
        # Execute all sends concurrently
        send_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for peer_id, result in zip(peer_ids, send_results):
            results[peer_id] = result if not isinstance(result, Exception) else False
        
        successful_sends = sum(1 for success in results.values() if success)
        logger.info(f"Broadcast complete: {successful_sends}/{len(results)} successful")
        
        return results
    
    async def _send_with_retry(self, message: Message, endpoint: str) -> bool:
        """Send message with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                success = await self.protocol.send_message(message, endpoint)
                if success:
                    return True
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Send attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return False
    
    async def _message_processor_loop(self) -> None:
        """Process incoming messages continuously."""
        while True:
            try:
                message = await self.protocol.receive_message()
                if message:
                    await self._process_incoming_message(message)
                else:
                    await asyncio.sleep(0.1)  # Small delay if no messages
                    
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_incoming_message(self, message: Message) -> None:
        """Process a received message."""
        try:
            # Verify message integrity
            if not message.verify_checksum():
                logger.warning(f"Message integrity check failed: {message.message_id}")
                return
            
            # Update statistics
            self.statistics['messages_received'] += 1
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to all peers."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                heartbeat_payload = {
                    'timestamp': time.time(),
                    'statistics': self.statistics.copy()
                }
                
                await self.broadcast_message('heartbeat', heartbeat_payload)
                self.statistics['last_heartbeat'] = time.time()
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.register_message_handler('heartbeat', self._handle_heartbeat)
        self.register_message_handler('ping', self._handle_ping)
    
    async def _handle_heartbeat(self, message: Message) -> None:
        """Handle heartbeat message."""
        logger.debug(f"Heartbeat received from {message.sender_id}")
        
        # Could update peer health status here
        peer_stats = message.payload.get('statistics', {})
        logger.debug(f"Peer {message.sender_id} stats: {peer_stats}")
    
    async def _handle_ping(self, message: Message) -> None:
        """Handle ping message with pong response."""
        pong_payload = {
            'original_timestamp': message.payload.get('timestamp'),
            'response_timestamp': time.time()
        }
        
        await self.send_message(
            message.sender_id,
            'pong',
            pong_payload,
            message.round_number
        )
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        timestamp = str(int(time.time() * 1000000))  # Microsecond precision
        return f"{self.node_id}_{timestamp}"
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            'node_id': self.node_id,
            'protocol': self.config.protocol,
            'registered_peers': len(self.peer_endpoints),
            'peer_endpoints': self.peer_endpoints.copy(),
            'statistics': self.statistics.copy(),
            'configuration': {
                'encryption_enabled': self.config.encryption_enabled,
                'compression_enabled': self.config.compression_enabled,
                'timeout_seconds': self.config.timeout_seconds,
                'max_retries': self.config.max_retries,
                'heartbeat_interval': self.config.heartbeat_interval
            }
        }
    
    async def test_connectivity(self, peer_id: str) -> Tuple[bool, float]:
        """
        Test connectivity to a peer.
        
        Args:
            peer_id: ID of peer to test
        
        Returns:
            Tuple of (success, round_trip_time)
        """
        if peer_id not in self.peer_endpoints:
            return False, 0.0
        
        start_time = time.time()
        
        ping_payload = {
            'timestamp': start_time,
            'test_message': 'connectivity_test'
        }
        
        success = await self.send_message(peer_id, 'ping', ping_payload)
        
        if success:
            # In a real implementation, you would wait for the pong response
            # and calculate actual round-trip time
            end_time = time.time()
            rtt = end_time - start_time
            return True, rtt
        
        return False, 0.0
    
    async def wait_for_message(
        self,
        message_type: str,
        sender_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Wait for a specific message type.
        
        Args:
            message_type: Type of message to wait for
            sender_id: Optional specific sender to wait for
            timeout: Timeout in seconds
        
        Returns:
            Received message or None if timeout
        """
        start_time = time.time()
        timeout = timeout or self.config.timeout_seconds
        
        while True:
            message = await self.protocol.receive_message()
            
            if message and message.message_type == message_type:
                if sender_id is None or message.sender_id == sender_id:
                    return message
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for message type: {message_type}")
                return None
            
            await asyncio.sleep(0.1)
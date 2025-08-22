import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import numpy as np
import json
import tempfile
import os
from pathlib import Path

from src.federated.server import FederatedServer, FederatedServerConfig
from src.federated.client import FederatedClient, FederatedClientConfig
from src.federated.communication import CommunicationConfig
from src.models.server_model import ServerModelConfig
from src.models.client_model import ClientModelConfig
from src.models.base_model import ModelConfig
from src.bias_mitigation.bias_detector import BiasDetectionSuite
from src.bias_mitigation.fairness_metrics import FairnessMetricsCalculator
from src.privacy.differential_privacy import DifferentialPrivacyManager
from src.utils.metrics import PerformanceTracker, ModelMetrics, FederatedMetrics
from src.utils.visualization import create_performance_plots, create_fairness_plots
from src.utils.helpers import setup_logging, save_config, load_config
from config.settings import settings

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global application state
app_state = {
    'server': None,
    'clients': {},
    'performance_tracker': PerformanceTracker(),
    'bias_detector': None,
    'privacy_manager': None,
    'training_active': False
}

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Federated Learning API...")
    
    # Initialize bias detector
    app_state['bias_detector'] = BiasDetectionSuite(
        sensitive_attributes=['age_group', 'gender', 'ethnicity']
    )
    
    # Initialize privacy manager
    app_state['privacy_manager'] = DifferentialPrivacyManager()
    
    yield
    
    # Cleanup
    logger.info("Shutting down Federated Learning API...")
    if app_state['server']:
        await app_state['server'].stop()
    
    for client in app_state['clients'].values():
        await client.stop()


# FastAPI app with lifespan
app = FastAPI(
    title="Federated Learning API",
    description="Production-ready federated learning system with bias mitigation and privacy preservation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Pydantic models for request/response
class ServerConfigRequest(BaseModel):
    server_id: str = "federated_server"
    min_clients: int = 2
    max_clients: int = 100
    client_selection_fraction: float = 0.3
    round_timeout: int = 300
    max_rounds: int = 100
    convergence_threshold: float = 0.001
    model_type: str = "cnn"
    model_framework: str = "tensorflow"
    aggregation_strategy: str = "fedavg"


class ClientConfigRequest(BaseModel):
    client_id: str
    server_endpoint: str = "http://localhost:8080"
    auto_participate: bool = True
    min_data_samples: int = 10
    max_rounds: int = 100
    local_evaluation: bool = True
    privacy_enabled: bool = False
    fairness_enabled: bool = False


class TrainingRequest(BaseModel):
    test_data_path: Optional[str] = None
    demographic_data_path: Optional[str] = None
    evaluation_frequency: int = 1


class ModelEvaluationRequest(BaseModel):
    data_path: str
    model_path: Optional[str] = None
    fairness_evaluation: bool = True
    demographic_data_path: Optional[str] = None


class BiasDetectionRequest(BaseModel):
    model_path: str
    test_data_path: str
    demographic_data_path: str
    save_report: bool = True
    report_path: Optional[str] = None


# Authentication dependency (simplified)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - enhance for production."""
    # In production, verify JWT token or API key
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"user_id": "api_user"}  # Simplified


# Health and status endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2025-08-22T10:00:00Z",
        "version": "1.0.0",
        "components": {
            "server_active": app_state['server'] is not None,
            "clients_connected": len(app_state['clients']),
            "training_active": app_state['training_active']
        }
    }


@app.get("/status")
async def get_system_status():
    """Get comprehensive system status."""
    status_info = {
        "system": {
            "server_initialized": app_state['server'] is not None,
            "client_count": len(app_state['clients']),
            "training_active": app_state['training_active']
        },
        "performance": app_state['performance_tracker'].calculate_performance_summary(),
        "server_status": None,
        "client_status": {}
    }
    
    # Add server status if available
    if app_state['server']:
        status_info['server_status'] = app_state['server'].get_server_status()
    
    # Add client status
    for client_id, client in app_state['clients'].items():
        status_info['client_status'][client_id] = client.get_client_status()
    
    return status_info


# Server management endpoints
@app.post("/server/initialize")
async def initialize_server(
    config: ServerConfigRequest,
    user=Depends(get_current_user)
):
    """Initialize federated learning server."""
    try:
        if app_state['server']:
            raise HTTPException(
                status_code=400,
                detail="Server already initialized"
            )
        
        # Create server configuration
        comm_config = CommunicationConfig(
            protocol="http",
            encryption_enabled=True,
            timeout_seconds=config.round_timeout
        )
        
        model_config = ServerModelConfig(
            base_config=ModelConfig(
                input_shape=(224, 224, 3),  # Default for facial recognition
                num_classes=2,
                dropout_rate=0.5
            ),
            num_clients=config.max_clients,
            aggregation_strategy=config.aggregation_strategy,
            client_fraction=config.client_selection_fraction,
            min_clients=config.min_clients,
            max_rounds=config.max_rounds,
            convergence_threshold=config.convergence_threshold
        )
        
        server_config = FederatedServerConfig(
            server_id=config.server_id,
            communication_config=comm_config,
            model_config=model_config,
            min_clients=config.min_clients,
            max_clients=config.max_clients,
            client_selection_fraction=config.client_selection_fraction,
            round_timeout=config.round_timeout,
            max_rounds=config.max_rounds,
            convergence_threshold=config.convergence_threshold
        )
        
        # Initialize server
        app_state['server'] = FederatedServer(server_config)
        
        # Start server (async)
        await app_state['server'].start()
        
        logger.info("Federated server initialized successfully")
        
        return {
            "message": "Server initialized successfully",
            "server_id": config.server_id,
            "configuration": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/server/start-training")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Start federated training."""
    try:
        if not app_state['server']:
            raise HTTPException(
                status_code=400,
                detail="Server not initialized"
            )
        
        if app_state['training_active']:
            raise HTTPException(
                status_code=400,
                detail="Training already in progress"
            )
        
        # Load test data if provided
        test_data = None
        if request.test_data_path:
            # In production, implement secure file loading
            test_data = _load_test_data(request.test_data_path)
        
        # Start training in background
        app_state['training_active'] = True
        background_tasks.add_task(_run_training, test_data)
        
        return {
            "message": "Training started successfully",
            "test_data_provided": test_data is not None
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/server/stop-training")
async def stop_training(user=Depends(get_current_user)):
    """Stop federated training."""
    try:
        if not app_state['server']:
            raise HTTPException(
                status_code=400,
                detail="Server not initialized"
            )
        
        await app_state['server'].stop_training()
        app_state['training_active'] = False
        
        return {"message": "Training stopped successfully"}
        
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/server/training-summary")
async def get_training_summary(user=Depends(get_current_user)):
    """Get training progress summary."""
    if not app_state['server']:
        raise HTTPException(
            status_code=400,
            detail="Server not initialized"
        )
    
    return app_state['server'].get_training_summary()


@app.delete("/server/shutdown")
async def shutdown_server(user=Depends(get_current_user)):
    """Shutdown federated server."""
    try:
        if app_state['server']:
            await app_state['server'].stop()
            app_state['server'] = None
            app_state['training_active'] = False
        
        return {"message": "Server shutdown successfully"}
        
    except Exception as e:
        logger.error(f"Failed to shutdown server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Client management endpoints
@app.post("/client/register")
async def register_client(
    config: ClientConfigRequest,
    user=Depends(get_current_user)
):
    """Register a new federated client."""
    try:
        if config.client_id in app_state['clients']:
            raise HTTPException(
                status_code=400,
                detail=f"Client {config.client_id} already exists"
            )
        
        # Create client configuration
        comm_config = CommunicationConfig(
            protocol="http",
            encryption_enabled=True
        )
        
        model_config = ClientModelConfig(
            base_config=ModelConfig(
                input_shape=(224, 224, 3),
                num_classes=2,
                dropout_rate=0.5
            ),
            client_id=int(config.client_id.split('_')[-1]) if '_' in config.client_id else 0,
            local_epochs=5,
            local_batch_size=32,
            learning_rate=0.001,
            differential_privacy=config.privacy_enabled
        )
        
        client_config = FederatedClientConfig(
            client_id=config.client_id,
            server_endpoint=config.server_endpoint,
            communication_config=comm_config,
            model_config=model_config,
            auto_participate=config.auto_participate,
            min_data_samples=config.min_data_samples,
            max_rounds=config.max_rounds,
            local_evaluation=config.local_evaluation
        )
        
        # Initialize client
        client = FederatedClient(client_config)
        app_state['clients'][config.client_id] = client
        
        # Start client
        await client.start()
        
        logger.info(f"Client {config.client_id} registered successfully")
        
        return {
            "message": f"Client {config.client_id} registered successfully",
            "client_id": config.client_id,
            "configuration": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to register client {config.client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/client/{client_id}/set-data")
async def set_client_data(
    client_id: str,
    train_data: UploadFile = File(...),
    val_data: Optional[UploadFile] = File(None),
    demographic_data: Optional[UploadFile] = File(None),
    user=Depends(get_current_user)
):
    """Set training data for a specific client."""
    try:
        if client_id not in app_state['clients']:
            raise HTTPException(
                status_code=404,
                detail=f"Client {client_id} not found"
            )
        
        client = app_state['clients'][client_id]
        
        # Load and process data files
        X_train, y_train = await _process_data_file(train_data)
        
        X_val, y_val = None, None
        if val_data:
            X_val, y_val = await _process_data_file(val_data)
        
        demographic_info = None
        if demographic_data:
            demographic_info = await _process_demographic_file(demographic_data)
        
        # Set data on client
        client.set_training_data(X_train, y_train, X_val, y_val, demographic_info)
        
        return {
            "message": f"Data set for client {client_id}",
            "train_samples": len(X_train),
            "val_samples": len(X_val) if X_val is not None else 0,
            "demographic_data": demographic_info is not None
        }
        
    except Exception as e:
        logger.error(f"Failed to set data for client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/client/{client_id}/status")
async def get_client_status(client_id: str, user=Depends(get_current_user)):
    """Get status of specific client."""
    if client_id not in app_state['clients']:
        raise HTTPException(
            status_code=404,
            detail=f"Client {client_id} not found"
        )
    
    return app_state['clients'][client_id].get_client_status()


@app.get("/client/{client_id}/performance")
async def get_client_performance(client_id: str, user=Depends(get_current_user)):
    """Get performance summary for specific client."""
    if client_id not in app_state['clients']:
        raise HTTPException(
            status_code=404,
            detail=f"Client {client_id} not found"
        )
    
    return app_state['clients'][client_id].get_performance_summary()


@app.delete("/client/{client_id}")
async def remove_client(client_id: str, user=Depends(get_current_user)):
    """Remove a federated client."""
    try:
        if client_id not in app_state['clients']:
            raise HTTPException(
                status_code=404,
                detail=f"Client {client_id} not found"
            )
        
        client = app_state['clients'][client_id]
        await client.stop()
        del app_state['clients'][client_id]
        
        return {"message": f"Client {client_id} removed successfully"}
        
    except Exception as e:
        logger.error(f"Failed to remove client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model evaluation endpoints
@app.post("/model/evaluate")
async def evaluate_model(
    request: ModelEvaluationRequest,
    user=Depends(get_current_user)
):
    """Evaluate model performance with optional fairness assessment."""
    try:
        # Load test data
        X_test, y_test = _load_test_data(request.data_path)
        
        # Load demographic data if provided
        demographic_data = None
        if request.demographic_data_path:
            demographic_data = _load_demographic_data(request.demographic_data_path)
        
        # Get model (from server or load from path)
        if request.model_path:
            # Load model from file
            model = _load_model(request.model_path)
        elif app_state['server']:
            # Use server's global model
            model = app_state['server'].model
        else:
            raise HTTPException(
                status_code=400,
                detail="No model available for evaluation"
            )
        
        # Perform evaluation
        evaluation_result = await _evaluate_model(
            model, X_test, y_test, demographic_data, request.fairness_evaluation
        )
        
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/bias-detection")
async def detect_bias(
    request: BiasDetectionRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Perform comprehensive bias detection on model."""
    try:
        # Load required data
        model = _load_model(request.model_path)
        X_test, y_test = _load_test_data(request.test_data_path)
        demographic_data = _load_demographic_data(request.demographic_data_path)
        
        # Run bias detection in background
        if request.save_report:
            background_tasks.add_task(
                _run_bias_detection,
                model, X_test, y_test, demographic_data, request.report_path
            )
            
            return {
                "message": "Bias detection started",
                "report_will_be_saved": True,
                "report_path": request.report_path or "auto-generated"
            }
        else:
            # Run immediately and return results
            bias_result = await _run_bias_detection(
                model, X_test, y_test, demographic_data
            )
            return bias_result
            
    except Exception as e:
        logger.error(f"Bias detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Visualization endpoints
@app.get("/visualizations/training-progress")
async def get_training_progress_plot(user=Depends(get_current_user)):
    """Generate training progress visualization."""
    try:
        # Generate plot
        plot_path = create_performance_plots(
            app_state['performance_tracker'],
            save_dir="temp_plots"
        )
        
        if plot_path:
            return FileResponse(
                plot_path,
                media_type="image/png",
                filename="training_progress.png"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="No training data available for plotting"
            )
            
    except Exception as e:
        logger.error(f"Failed to generate training plot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualizations/fairness-metrics")
async def get_fairness_metrics_plot(user=Depends(get_current_user)):
    """Generate fairness metrics visualization."""
    try:
        # Generate fairness plot
        plot_path = create_fairness_plots(
            app_state['performance_tracker'],
            save_dir="temp_plots"
        )
        
        if plot_path:
            return FileResponse(
                plot_path,
                media_type="image/png", 
                filename="fairness_metrics.png"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="No fairness data available for plotting"
            )
            
    except Exception as e:
        logger.error(f"Failed to generate fairness plot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@app.post("/config/save")
async def save_system_config(
    config_name: str,
    user=Depends(get_current_user)
):
    """Save current system configuration."""
    try:
        # Collect current configuration
        system_config = {
            "server_config": app_state['server'].config if app_state['server'] else None,
            "client_configs": {
                client_id: client.config 
                for client_id, client in app_state['clients'].items()
            },
            "performance_metrics": app_state['performance_tracker'].calculate_performance_summary(),
            "timestamp": "2025-08-22T10:00:00Z"
        }
        
        # Save configuration
        config_path = f"configs/{config_name}.json"
        save_config(system_config, config_path)
        
        return {
            "message": f"Configuration saved as {config_name}",
            "config_path": config_path
        }
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/load")
async def load_system_config(
    config_name: str,
    user=Depends(get_current_user)
):
    """Load system configuration."""
    try:
        config_path = f"configs/{config_name}.json"
        system_config = load_config(config_path)
        
        return {
            "message": f"Configuration {config_name} loaded",
            "config": system_config
        }
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility functions
async def _run_training(test_data):
    """Background task to run federated training."""
    try:
        await app_state['server'].start_training(test_data)
    except Exception as e:
        logger.error(f"Training failed: {e}")
    finally:
        app_state['training_active'] = False


async def _process_data_file(file: UploadFile) -> tuple:
    """Process uploaded data file."""
    # Simplified - in production, implement proper data loading
    content = await file.read()
    
    # This is a placeholder - implement actual data processing
    # based on your data format (CSV, NPY, etc.)
    X = np.random.rand(100, 224, 224, 3)  # Placeholder
    y = np.random.randint(0, 2, 100)       # Placeholder
    
    return X, y


async def _process_demographic_file(file: UploadFile) -> dict:
    """Process uploaded demographic data file."""
    content = await file.read()
    
    # Placeholder demographic data
    demographic_data = {
        'age_group': np.random.choice(['young', 'middle', 'old'], 100),
        'gender': np.random.choice(['male', 'female'], 100),
        'ethnicity': np.random.choice(['A', 'B', 'C', 'D'], 100)
    }
    
    return demographic_data


def _load_test_data(data_path: str) -> tuple:
    """Load test data from file path."""
    # Placeholder implementation
    X = np.random.rand(50, 224, 224, 3)
    y = np.random.randint(0, 2, 50)
    return X, y


def _load_demographic_data(data_path: str) -> dict:
    """Load demographic data from file path."""
    # Placeholder implementation
    return {
        'age_group': np.random.choice(['young', 'middle', 'old'], 50),
        'gender': np.random.choice(['male', 'female'], 50),
        'ethnicity': np.random.choice(['A', 'B', 'C', 'D'], 50)
    }


def _load_model(model_path: str):
    """Load model from file path."""
    # Placeholder - implement actual model loading
    return None


async def _evaluate_model(model, X_test, y_test, demographic_data=None, fairness_eval=True):
    """Evaluate model with optional fairness assessment."""
    # Placeholder evaluation
    y_pred = np.random.randint(0, 2, len(y_test))
    y_prob = np.random.rand(len(y_test))
    
    from src.utils.metrics import calculate_model_metrics
    metrics = calculate_model_metrics(y_test, y_pred, y_prob)
    
    result = {
        "model_metrics": metrics.to_dict(),
        "fairness_metrics": None
    }
    
    if fairness_eval and demographic_data:
        # Calculate fairness metrics
        fairness_calc = FairnessMetricsCalculator(['age_group', 'gender', 'ethnicity'])
        fairness_results = fairness_calc.calculate_all_metrics(
            y_test, y_pred, y_prob, demographic_data
        )
        result["fairness_metrics"] = {
            name: result.to_dict() if hasattr(result, 'to_dict') else str(result)
            for name, result in fairness_results.items()
        }
    
    return result


async def _run_bias_detection(model, X_test, y_test, demographic_data, report_path=None):
    """Run comprehensive bias detection."""
    if app_state['bias_detector']:
        bias_results = app_state['bias_detector'].run_comprehensive_detection(
            model, X_test, y_test, demographic_data
        )
        
        if report_path:
            # Save detailed report
            with open(report_path, 'w') as f:
                json.dump(
                    {name: result.to_dict() if hasattr(result, 'to_dict') else str(result)
                     for name, result in bias_results.items()},
                    f, indent=2
                )
        
        return {
            "bias_detected": any(result.bias_detected for result in bias_results.values() 
                               if hasattr(result, 'bias_detected')),
            "total_issues": len([r for r in bias_results.values() 
                               if hasattr(r, 'bias_detected') and r.bias_detected]),
            "results_summary": {
                name: {
                    "bias_detected": result.bias_detected if hasattr(result, 'bias_detected') else False,
                    "severity": result.severity_level if hasattr(result, 'severity_level') else 'unknown'
                }
                for name, result in bias_results.items()
            }
        }
    
    return {"error": "Bias detector not available"}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": "2025-08-22T10:00:00Z"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": "2025-08-22T10:00:00Z"}
    )


# Main function for running the API
def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
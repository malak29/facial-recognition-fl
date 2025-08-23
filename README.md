# 🤖 Federated Learning System for Healthcare
### *Privacy-Preserving Facial Recognition with Bias Mitigation*

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)  
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [License](#license)

## 🎯 Overview

This **production-ready federated learning system** enables hospitals and healthcare institutions to collaboratively train facial recognition models while:

- 🔒 **Preserving Privacy**: Data never leaves individual institutions
- ⚖️ **Mitigating Bias**: Advanced fairness algorithms ensure equitable performance across demographics
- 🏥 **Healthcare Focused**: Designed specifically for medical environments with HIPAA considerations
- 🚀 **Production Ready**: Complete CI/CD, monitoring, and deployment infrastructure

### What Problem Does This Solve?

Traditional machine learning requires centralizing sensitive patient data, creating privacy risks and regulatory challenges. This system enables:

- **Collaborative Learning**: Multiple hospitals can improve model accuracy together
- **Data Privacy**: Patient data stays within each institution's secure environment  
- **Bias Reduction**: Ensures fair performance across age, gender, and ethnic groups
- **Regulatory Compliance**: Meets healthcare privacy requirements (HIPAA, GDPR)

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Healthcare Institution A"
        CA[Client A<br/>Local Training]
        DA[Patient Data A<br/>🔒 Private]
        MA[Local Model A]
        
        CA --> DA
        CA --> MA
    end
    
    subgraph "Healthcare Institution B"  
        CB[Client B<br/>Local Training]
        DB[Patient Data B<br/>🔒 Private]
        MB[Local Model B]
        
        CB --> DB
        CB --> MB
    end
    
    subgraph "Healthcare Institution C"
        CC[Client C<br/>Local Training]
        DC[Patient Data C<br/>🔒 Private]
        MC[Local Model C]
        
        CC --> DC
        CC --> MC
    end
    
    subgraph "Federated Learning Server"
        FS[Federated Server<br/>Orchestration]
        AG[Model Aggregator<br/>FedAvg/FairFed]
        GM[Global Model<br/>Shared Weights]
        BM[Bias Mitigation<br/>Fairness Engine]
        PM[Privacy Manager<br/>Differential Privacy]
        
        FS --> AG
        AG --> GM
        AG --> BM
        AG --> PM
    end
    
    subgraph "Monitoring & DevOps"
        MO[Prometheus<br/>Metrics]
        GR[Grafana<br/>Dashboards] 
        AL[Alertmanager<br/>Notifications]
        JA[Jaeger<br/>Tracing]
        
        MO --> GR
        MO --> AL
    end
    
    subgraph "API Layer"
        API[FastAPI<br/>REST Endpoints]
        AUTH[Authentication<br/>JWT/OAuth]
        DOC[Auto Documentation<br/>OpenAPI]
        
        API --> AUTH
        API --> DOC
    end
    
    CA -.->|Encrypted Updates| FS
    CB -.->|Encrypted Updates| FS  
    CC -.->|Encrypted Updates| FS
    FS -.->|Global Weights| CA
    FS -.->|Global Weights| CB
    FS -.->|Global Weights| CC
    
    FS --> MO
    API --> FS
    
    classDef client fill:#e1f5fe
    classDef server fill:#f3e5f5
    classDef monitoring fill:#e8f5e8
    classDef api fill:#fff3e0
    classDef data fill:#ffebee
    
    class CA,CB,CC client
    class FS,AG,GM,BM,PM server
    class MO,GR,AL,JA monitoring
    class API,AUTH,DOC api
    class DA,DB,DC data
```

### System Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Federated Server** | Orchestrates training rounds, aggregates models | Python, FastAPI |
| **Federated Clients** | Train models locally on private data | Python, TensorFlow/PyTorch |
| **Bias Mitigation** | Ensures fairness across demographic groups | Custom algorithms |
| **Privacy Engine** | Differential privacy and secure aggregation | TensorFlow Privacy |
| **Communication** | Secure client-server messaging | HTTP/WebSocket, SSL/TLS |
| **Monitoring** | System health and performance tracking | Prometheus, Grafana |
| **API Gateway** | REST API for system management | FastAPI, OAuth |

## ✨ Key Features

### 🔐 Privacy & Security
- **Differential Privacy**: Mathematical privacy guarantees with epsilon-delta framework
- **Secure Aggregation**: Encrypted model updates with homomorphic encryption
- **Data Isolation**: Patient data never leaves institutional boundaries
- **SSL/TLS**: End-to-end encryption for all communications

### ⚖️ Fairness & Bias Mitigation
- **Demographic Parity**: Ensures equal treatment across groups
- **Equalized Odds**: Balances true/false positive rates
- **Intersectional Analysis**: Detects bias across multiple attributes
- **Fairness-Aware Aggregation**: FairFed algorithm for bias reduction

### 🚀 Production Ready
- **Auto-scaling**: Kubernetes HPA for dynamic client scaling
- **Health Monitoring**: Comprehensive metrics and alerting
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **Multi-environment**: Development, staging, and production configs

### 🧠 Advanced ML
- **Multiple Architectures**: CNN, ResNet, EfficientNet support
- **Aggregation Strategies**: FedAvg, FedProx, SCAFFOLD algorithms
- **Non-IID Handling**: Robust performance with heterogeneous data
- **Model Compression**: Efficient communication and storage

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- 8GB+ RAM recommended

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourorg/federated-learning.git
cd federated-learning
```

### 2️⃣ Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Quick Demo with Docker
```bash
# Start all services
docker-compose up -d

# Check system health
curl http://localhost:8080/health

# Access Grafana dashboard
open http://localhost:3000
# Default login: admin/admin
```

### 4️⃣ Verify Installation
```bash
# Run tests
python -m pytest tests/unit/ -v

# Check API endpoints
curl http://localhost:8080/status
```

## 📦 Installation

### Development Environment
```bash
# 1. Clone and setup
git clone https://github.com/yourorg/federated-learning.git
cd federated-learning

# 2. Install system dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake pkg-config \
    libopencv-dev libhdf5-dev libjpeg-dev libpng-dev

# 3. Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Initialize database
python scripts/init_database.py

# 6. Start development server
python -m api.app --reload
```

### Production Deployment

#### Option A: Docker Compose
```bash
# 1. Production configuration
cp docker/docker-compose.yml docker/docker-compose.prod.yml
# Edit production settings

# 2. Deploy with production config
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 3. Verify deployment
./scripts/verify-deployment.sh
```

#### Option B: Kubernetes
```bash
# 1. Deploy infrastructure
cd infrastructure/terraform
terraform init && terraform apply

# 2. Deploy application
./scripts/deploy.sh --environment production --type kubernetes

# 3. Setup monitoring
./scripts/monitoring-setup.sh --environment production
```

#### Option C: Helm (Recommended)
```bash
# 1. Add Helm repository
helm repo add federated-learning https://charts.yourdomain.com

# 2. Install with custom values
helm install fl-system federated-learning/federated-learning \
    --namespace federated-learning \
    --create-namespace \
    --values production-values.yaml

# 3. Verify installation
helm status fl-system
```

## 💡 Usage Examples

### Basic API Usage

#### 1. Initialize Server
```python
import requests

# Initialize federated server
response = requests.post("http://localhost:8080/server/initialize", 
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "server_id": "hospital_network_server",
        "min_clients": 3,
        "max_clients": 10,
        "aggregation_strategy": "fairfed",
        "max_rounds": 50
    }
)
```

#### 2. Register Clients
```python
# Register hospital client
response = requests.post("http://localhost:8080/client/register",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "client_id": "hospital_a_client",
        "server_endpoint": "http://localhost:8081",
        "privacy_enabled": True,
        "fairness_enabled": True
    }
)
```

#### 3. Start Training
```python
# Begin federated training
response = requests.post("http://localhost:8080/server/start-training",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "test_data_path": "/data/global_test_set.npz",
        "evaluation_frequency": 1
    }
)
```

#### 4. Monitor Progress
```python
# Get training status
status = requests.get("http://localhost:8080/status",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

# Get fairness metrics
fairness = requests.get("http://localhost:8080/model/fairness-report",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

### Python SDK Usage

```python
from src.federated.server import FederatedServer, FederatedServerConfig
from src.federated.client import FederatedClient, FederatedClientConfig
from src.models.base_model import ModelConfig

# 1. Setup server
server_config = FederatedServerConfig(
    min_clients=2,
    aggregation_strategy="fairfed",
    privacy_enabled=True
)

server = FederatedServer(server_config)
await server.start()

# 2. Setup client
client_config = FederatedClientConfig(
    client_id="hospital_a",
    server_endpoint="http://server:8081"
)

client = FederatedClient(client_config)
client.set_training_data(X_train, y_train, demographics=demo_data)
await client.start()

# 3. Run training
await server.start_training()
```

### Command Line Usage

```bash
# Initialize and start training
python scripts/train_federated.py \
    --config configs/hospital_network.yaml \
    --clients 5 \
    --rounds 100 \
    --privacy-budget 1.0

# Evaluate model fairness
python scripts/evaluate_fairness.py \
    --model checkpoints/global_model_round_50.h5 \
    --test-data data/test_set.npz \
    --demographics data/demographics.json

# Run bias detection
python scripts/detect_bias.py \
    --model checkpoints/global_model_round_50.h5 \
    --data data/evaluation_set.npz \
    --report-path reports/bias_analysis.html
```

## 📖 API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/status` | GET | Comprehensive system status |
| `/server/initialize` | POST | Initialize federated server |
| `/server/start-training` | POST | Begin federated training |
| `/client/register` | POST | Register new client |
| `/model/evaluate` | POST | Evaluate model performance |
| `/model/bias-detection` | POST | Run bias detection analysis |

### Authentication
All API endpoints require JWT authentication:
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:8080/status
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## 🚀 Deployment

### Quick Development Setup
```bash
# Start entire stack with one command
docker-compose up -d

# Services will be available at:
# - API: http://localhost:8080
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
```

### Production Deployment Options

#### 🐳 Docker Swarm
```bash
# Deploy to Docker Swarm
docker stack deploy -c docker-compose.yml -c docker-compose.prod.yml fl-system
```

#### ☸️ Kubernetes
```bash
# Deploy to Kubernetes cluster
./scripts/deploy.sh --environment production --type kubernetes

# Monitor deployment
kubectl get pods -n federated-learning
```

#### 📊 Helm Chart
```bash
# Install via Helm
helm install federated-learning ./helm/federated-learning \
    --namespace federated-learning \
    --create-namespace \
    --values values-production.yaml
```

#### ☁️ Cloud Deployment (AWS)
```bash
# Deploy infrastructure with Terraform
cd infrastructure/terraform
terraform init && terraform apply

# Deploy application
./scripts/deploy.sh --environment production --type helm
```

## 📊 Monitoring & Observability

### Dashboards & Metrics
- **Training Progress**: Model accuracy, loss, convergence tracking
- **Fairness Metrics**: Demographic parity, equalized odds, bias detection
- **Client Health**: Participation rates, performance, connectivity
- **System Resources**: CPU, memory, network usage
- **Privacy Metrics**: Budget utilization, noise levels

### Alerting
Automated alerts for:
- Training failures or stalls
- Fairness violations (bias > threshold)
- Privacy budget exhaustion
- System resource limits
- Client connectivity issues

### Access Monitoring
```bash
# Grafana Dashboard
kubectl port-forward service/grafana 3000:3000 -n monitoring

# Prometheus Metrics  
kubectl port-forward service/prometheus 9090:9090 -n monitoring

# Jaeger Tracing
kubectl port-forward service/jaeger 16686:16686 -n observability
```

## 🔧 Configuration

### Environment Variables
```bash
# Core Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/fl_db
REDIS_URL=redis://localhost:6379/0

# Federated Learning
MAX_CLIENTS=100
MIN_CLIENTS=2
AGGREGATION_STRATEGY=fairfed
PRIVACY_BUDGET=1.0

# Security
JWT_SECRET_KEY=your-secret-key
API_RATE_LIMIT=100

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=secure-password
SENTRY_DSN=https://your-sentry-dsn
```

### Configuration Files
- `config/settings.py`: Core application settings
- `configs/hospital_network.yaml`: Multi-hospital deployment
- `configs/single_institution.yaml`: Single institution setup
- `helm/values-production.yaml`: Production Helm values

## 🧪 Testing

### Run Test Suite
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# End-to-end tests
python -m pytest tests/e2e/ -v

# Performance benchmarks
python tests/benchmarks/performance_tests.py

# All tests with coverage
python -m pytest --cov=src --cov-report=html
```

### Load Testing
```bash
# Stress test with multiple clients
python tests/benchmarks/load_test.py --clients 50 --rounds 10

# Performance regression tests
python scripts/performance_regression.py --baseline baseline.json
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


#!/bin/bash
# Automated deployment script for Federated Learning System

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/deployment.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
DEPLOYMENT_TYPE="docker-compose"
NAMESPACE="federated-learning"
DRY_RUN=false
SKIP_TESTS=false
SKIP_BUILD=false
FORCE_DEPLOY=false
CONFIG_FILE=""

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $*${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*${NC}" | tee -a "$LOG_FILE"
}

# Help function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Federated Learning System

OPTIONS:
    -e, --environment ENV       Target environment (development|staging|production) [default: development]
    -t, --type TYPE            Deployment type (docker-compose|kubernetes|helm) [default: docker-compose]
    -n, --namespace NS         Kubernetes namespace [default: federated-learning]
    -c, --config FILE          Configuration file path
    -d, --dry-run              Perform dry run without actual deployment
    --skip-tests               Skip running tests before deployment
    --skip-build               Skip building Docker images
    --force                    Force deployment even with warnings
    -h, --help                 Show this help message

EXAMPLES:
    $0 --environment development --type docker-compose
    $0 --environment staging --type kubernetes --namespace fl-staging
    $0 --environment production --type helm --config prod-config.yaml
    $0 --dry-run --environment production

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
}

# Validation functions
validate_environment() {
    case $ENVIRONMENT in
        development|staging|production)
            log "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT. Must be development, staging, or production"
            ;;
    esac
}

validate_deployment_type() {
    case $DEPLOYMENT_TYPE in
        docker-compose|kubernetes|helm)
            log "Using $DEPLOYMENT_TYPE deployment type"
            ;;
        *)
            error "Invalid deployment type: $DEPLOYMENT_TYPE"
            ;;
    esac
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("python3" "pip" "git")
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            required_tools+=("docker" "docker-compose")
            ;;
        kubernetes)
            required_tools+=("kubectl")
            ;;
        helm)
            required_tools+=("kubectl" "helm")
            ;;
    esac
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check Docker daemon
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        if ! docker info &> /dev/null; then
            error "Docker daemon is not running"
        fi
    fi
    
    # Check Kubernetes connection
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" || "$DEPLOYMENT_TYPE" == "helm" ]]; then
        if ! kubectl cluster-info &> /dev/null; then
            error "Cannot connect to Kubernetes cluster"
        fi
    fi
    
    log "Prerequisites check passed"
}

# Pre-deployment checks
run_pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check if namespace exists (for Kubernetes)
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" || "$DEPLOYMENT_TYPE" == "helm" ]]; then
        if kubectl get namespace "$NAMESPACE" &> /dev/null; then
            warn "Namespace $NAMESPACE already exists"
        else
            log "Creating namespace $NAMESPACE"
            if [[ "$DRY_RUN" == "false" ]]; then
                kubectl create namespace "$NAMESPACE"
            fi
        fi
    fi
    
    # Check existing deployments
    local existing_deployments=""
    case $DEPLOYMENT_TYPE in
        docker-compose)
            existing_deployments=$(docker-compose ps --services 2>/dev/null || true)
            ;;
        kubernetes)
            existing_deployments=$(kubectl get deployments -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l || echo "0")
            ;;
        helm)
            existing_deployments=$(helm list -n "$NAMESPACE" --short 2>/dev/null || true)
            ;;
    esac
    
    if [[ -n "$existing_deployments" && "$existing_deployments" != "0" ]]; then
        if [[ "$FORCE_DEPLOY" == "false" ]]; then
            warn "Existing deployments found. Use --force to proceed"
            info "Existing deployments: $existing_deployments"
            read -p "Continue with deployment? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                error "Deployment cancelled by user"
            fi
        fi
    fi
}

# Test execution
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping tests as requested"
        return 0
    fi
    
    log "Running tests before deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Install test dependencies if not present
    if ! python3 -c "import pytest" 2>/dev/null; then
        log "Installing test dependencies..."
        pip3 install pytest pytest-cov pytest-asyncio pytest-mock
    fi
    
    # Run unit tests
    log "Running unit tests..."
    python3 -m pytest tests/unit/ -v --tb=short
    
    # Run integration tests for non-production
    if [[ "$ENVIRONMENT" != "production" ]]; then
        log "Running integration tests..."
        python3 -m pytest tests/integration/ -v --tb=short
    fi
    
    log "All tests passed"
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        warn "Skipping image build as requested"
        return 0
    fi
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" || "$DEPLOYMENT_TYPE" == "helm" ]]; then
        log "Building Docker images..."
        
        cd "$PROJECT_ROOT"
        
        # Build server image
        log "Building server image..."
        if [[ "$DRY_RUN" == "false" ]]; then
            docker build -f docker/Dockerfile.server -t fl-system/server:latest .
            docker tag fl-system/server:latest fl-system/server:$(git rev-parse --short HEAD)
        fi
        
        # Build client image
        log "Building client image..."
        if [[ "$DRY_RUN" == "false" ]]; then
            docker build -f docker/Dockerfile.client -t fl-system/client:latest .
            docker tag fl-system/client:latest fl-system/client:$(git rev-parse --short HEAD)
        fi
        
        log "Docker images built successfully"
    fi
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Check if override file exists for environment
    local compose_files="-f docker/docker-compose.yml"
    
    if [[ -f "docker/docker-compose.$ENVIRONMENT.yml" ]]; then
        compose_files="$compose_files -f docker/docker-compose.$ENVIRONMENT.yml"
        log "Using environment-specific override: docker-compose.$ENVIRONMENT.yml"
    fi
    
    # Create required secrets directory and files
    local secrets_dir="docker/secrets"
    if [[ ! -d "$secrets_dir" ]]; then
        log "Creating secrets directory..."
        mkdir -p "$secrets_dir"
        
        # Generate default passwords if they don't exist
        [[ ! -f "$secrets_dir/postgres_password.txt" ]] && openssl rand -base64 32 > "$secrets_dir/postgres_password.txt"
        [[ ! -f "$secrets_dir/grafana_admin_password.txt" ]] && openssl rand -base64 32 > "$secrets_dir/grafana_admin_password.txt"
        [[ ! -f "$secrets_dir/minio_password.txt" ]] && openssl rand -base64 32 > "$secrets_dir/minio_password.txt"
        [[ ! -f "$secrets_dir/api_secret_key.txt" ]] && openssl rand -base64 64 > "$secrets_dir/api_secret_key.txt"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Would execute: docker-compose $compose_files up -d"
        docker-compose $compose_files config --quiet
        log "Docker Compose configuration is valid"
    else
        # Pull latest images
        docker-compose $compose_files pull
        
        # Deploy services
        docker-compose $compose_files up -d
        
        # Wait for services to be healthy
        log "Waiting for services to be healthy..."
        sleep 30
        
        # Check service health
        local max_attempts=30
        local attempt=1
        
        while [[ $attempt -le $max_attempts ]]; do
            if curl -f http://localhost:8080/health &> /dev/null; then
                log "Services are healthy"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                error "Services failed to become healthy after $max_attempts attempts"
            fi
            
            info "Waiting for services... (attempt $attempt/$max_attempts)"
            sleep 10
            ((attempt++))
        done
    fi
    
    log "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Apply ConfigMaps and Secrets first
    log "Applying configurations..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f kubernetes/manifests/ -n "$NAMESPACE"
        log "DRY RUN: Kubernetes manifests are valid"
    else
        # Create namespace if it doesn't exist
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
        
        # Apply all manifests
        kubectl apply -f kubernetes/manifests/ -n "$NAMESPACE"
        
        # Wait for deployments
        log "Waiting for deployments to be ready..."
        kubectl rollout status deployment/fl-server -n "$NAMESPACE" --timeout=300s
        kubectl rollout status deployment/fl-client -n "$NAMESPACE" --timeout=300s
        
        # Check pod status
        kubectl get pods -n "$NAMESPACE"
        
        # Verify services
        local server_ip=$(kubectl get service fl-server-external -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [[ -n "$server_ip" ]]; then
            log "Server external IP: $server_ip"
            
            # Health check
            local max_attempts=20
            local attempt=1
            
            while [[ $attempt -le $max_attempts ]]; do
                if curl -f "http://$server_ip/health" &> /dev/null; then
                    log "Service health check passed"
                    break
                fi
                
                if [[ $attempt -eq $max_attempts ]]; then
                    warn "Health check failed, but deployment may still be starting"
                fi
                
                info "Waiting for service... (attempt $attempt/$max_attempts)"
                sleep 15
                ((attempt++))
            done
        fi
    fi
    
    log "Kubernetes deployment completed"
}

# Deploy with Helm
deploy_helm() {
    log "Deploying with Helm..."
    
    cd "$PROJECT_ROOT"
    
    local helm_values="helm/federated-learning/values.yaml"
    
    # Use environment-specific values if available
    if [[ -f "helm/federated-learning/values-$ENVIRONMENT.yaml" ]]; then
        helm_values="helm/federated-learning/values-$ENVIRONMENT.yaml"
        log "Using environment-specific Helm values: values-$ENVIRONMENT.yaml"
    fi
    
    # Custom config file override
    if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
        helm_values="$CONFIG_FILE"
        log "Using custom configuration file: $CONFIG_FILE"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        helm template federated-learning helm/federated-learning/ \
            --namespace "$NAMESPACE" \
            --values "$helm_values" \
            --set environment="$ENVIRONMENT" \
            --set image.tag=$(git rev-parse --short HEAD)
        log "DRY RUN: Helm chart is valid"
    else
        # Add/update Helm repositories
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo add grafana https://grafana.github.io/helm-charts
        helm repo update
        
        # Deploy with Helm
        helm upgrade --install federated-learning helm/federated-learning/ \
            --namespace "$NAMESPACE" \
            --create-namespace \
            --values "$helm_values" \
            --set environment="$ENVIRONMENT" \
            --set image.tag=$(git rev-parse --short HEAD) \
            --wait \
            --timeout=600s
        
        # Verify deployment
        helm status federated-learning -n "$NAMESPACE"
        kubectl get pods -n "$NAMESPACE"
    fi
    
    log "Helm deployment completed"
}

# Infrastructure deployment with Terraform
deploy_infrastructure() {
    if [[ "$DEPLOYMENT_TYPE" != "kubernetes" && "$DEPLOYMENT_TYPE" != "helm" ]]; then
        return 0
    fi
    
    log "Deploying infrastructure with Terraform..."
    
    cd "$PROJECT_ROOT/infrastructure/terraform"
    
    # Initialize Terraform
    terraform init
    
    # Validate configuration
    terraform validate
    
    # Plan infrastructure changes
    log "Planning infrastructure changes..."
    if [[ "$DRY_RUN" == "true" ]]; then
        terraform plan -var="environment=$ENVIRONMENT" -out=tfplan
        log "DRY RUN: Terraform plan completed"
        return 0
    fi
    
    terraform plan -var="environment=$ENVIRONMENT" -out=tfplan
    
    # Ask for confirmation in production
    if [[ "$ENVIRONMENT" == "production" && "$FORCE_DEPLOY" == "false" ]]; then
        read -p "Apply infrastructure changes? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Infrastructure deployment cancelled"
        fi
    fi
    
    # Apply infrastructure
    log "Applying infrastructure changes..."
    terraform apply tfplan
    
    # Update kubeconfig
    aws eks update-kubeconfig --region $(terraform output -raw aws_region) --name $(terraform output -raw cluster_name)
    
    log "Infrastructure deployment completed"
}

# Post-deployment verification
verify_deployment() {
    log "Verifying deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            # Check Docker services
            docker-compose ps
            
            # Test API endpoints
            if curl -f http://localhost:8080/health &> /dev/null; then
                log "✓ API health check passed"
            else
                error "✗ API health check failed"
            fi
            
            if curl -f http://localhost:8080/status &> /dev/null; then
                log "✓ API status check passed"
            else
                warn "✗ API status check failed"
            fi
            ;;
            
        kubernetes|helm)
            # Check pod status
            kubectl get pods -n "$NAMESPACE"
            
            # Check service endpoints
            local endpoints=$(kubectl get endpoints -n "$NAMESPACE" --no-headers | wc -l)
            if [[ $endpoints -gt 0 ]]; then
                log "✓ Service endpoints available"
            else
                error "✗ No service endpoints found"
            fi
            
            # Check ingress (if available)
            if kubectl get ingress fl-ingress -n "$NAMESPACE" &> /dev/null; then
                local ingress_host=$(kubectl get ingress fl-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')
                if [[ -n "$ingress_host" ]]; then
                    log "✓ Ingress configured: $ingress_host"
                fi
            fi
            ;;
    esac
    
    log "Deployment verification completed"
}

# Rollback function
rollback_deployment() {
    log "Rolling back deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            docker-compose down
            ;;
        kubernetes)
            kubectl delete -f kubernetes/manifests/ -n "$NAMESPACE" --ignore-not-found
            ;;
        helm)
            helm uninstall federated-learning -n "$NAMESPACE" || true
            ;;
    esac
    
    error "Deployment failed and was rolled back"
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # Remove temporary files
    rm -f tfplan
    
    # Clean up Docker build cache (if not production)
    if [[ "$ENVIRONMENT" != "production" && "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        docker system prune -f
    fi
}

# Signal handlers
trap cleanup EXIT
trap 'error "Deployment interrupted by user"' INT TERM

# Main deployment function
main() {
    log "Starting Federated Learning System deployment"
    log "Environment: $ENVIRONMENT"
    log "Deployment Type: $DEPLOYMENT_TYPE"
    log "Namespace: $NAMESPACE"
    log "Dry Run: $DRY_RUN"
    
    # Validation
    validate_environment
    validate_deployment_type
    check_prerequisites
    
    # Pre-deployment
    run_pre_deployment_checks
    
    # Tests
    if [[ "$ENVIRONMENT" != "production" ]]; then
        run_tests
    fi
    
    # Infrastructure (for cloud deployments)
    if [[ "$ENVIRONMENT" == "staging" || "$ENVIRONMENT" == "production" ]]; then
        deploy_infrastructure
    fi
    
    # Build images
    build_images
    
    # Deploy application
    case $DEPLOYMENT_TYPE in
        docker-compose)
            deploy_docker_compose
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        helm)
            deploy_helm
            ;;
    esac
    
    # Verification
    if [[ "$DRY_RUN" == "false" ]]; then
        verify_deployment
    fi
    
    log "Deployment completed successfully!"
    
    # Display access information
    case $DEPLOYMENT_TYPE in
        docker-compose)
            info "API available at: http://localhost:8080"
            info "Grafana available at: http://localhost:3000"
            info "Prometheus available at: http://localhost:9090"
            ;;
        kubernetes|helm)
            info "Use 'kubectl get services -n $NAMESPACE' to see service endpoints"
            info "Use 'kubectl port-forward -n $NAMESPACE service/fl-server 8080:8080' for local access"
            ;;
    esac
}

# Parse arguments and run main function
parse_args "$@"
main
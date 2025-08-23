#!/bin/bash
# Monitoring and Observability Setup for Federated Learning System

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
MONITORING_NAMESPACE="monitoring"
ENVIRONMENT="development"
GRAFANA_ADMIN_PASSWORD=""
ENABLE_ALERTING=true
ENABLE_TRACING=true
SLACK_WEBHOOK_URL=""

log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $*${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*${NC}"; exit 1; }
info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*${NC}"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup monitoring and observability for Federated Learning System

OPTIONS:
    -e, --environment ENV       Target environment [default: development]
    -n, --namespace NS          Monitoring namespace [default: monitoring]
    -p, --grafana-password PWD  Grafana admin password
    --disable-alerting          Disable alerting setup
    --disable-tracing           Disable distributed tracing
    --slack-webhook URL         Slack webhook for alerts
    -h, --help                  Show this help message

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                MONITORING_NAMESPACE="$2"
                shift 2
                ;;
            -p|--grafana-password)
                GRAFANA_ADMIN_PASSWORD="$2"
                shift 2
                ;;
            --disable-alerting)
                ENABLE_ALERTING=false
                shift
                ;;
            --disable-tracing)
                ENABLE_TRACING=false
                shift
                ;;
            --slack-webhook)
                SLACK_WEBHOOK_URL="$2"
                shift 2
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

check_prerequisites() {
    log "Checking prerequisites..."
    
    local required_tools=("kubectl" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    log "Prerequisites check passed"
}

setup_prometheus_stack() {
    log "Setting up Prometheus monitoring stack..."
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm repo update
    
    # Create namespace
    kubectl create namespace "$MONITORING_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Generate Grafana password if not provided
    if [[ -z "$GRAFANA_ADMIN_PASSWORD" ]]; then
        GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 12)
        warn "Generated Grafana admin password: $GRAFANA_ADMIN_PASSWORD"
    fi
    
    # Create Prometheus values file
    cat > /tmp/prometheus-values.yaml << EOF
prometheus:
  prometheusSpec:
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 20Gi
    
    retention: 15d
    retentionSize: 15GB
    
    additionalScrapeConfigs:
      - job_name: 'federated-learning-server'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names: ["federated-learning"]
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: fl-server
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            target_label: __address__
            regex: (.+)
            replacement: \${1}:8082
      
      - job_name: 'federated-learning-clients'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names: ["federated-learning"]
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: fl-client
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            target_label: __address__
            regex: (.+)
            replacement: \${1}:8083

alertmanager:
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 10Gi

grafana:
  adminPassword: "$GRAFANA_ADMIN_PASSWORD"
  
  persistence:
    enabled: true
    storageClassName: gp3
    size: 5Gi
  
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
      - name: 'federated-learning'
        orgId: 1
        folder: 'Federated Learning'
        type: file
        disableDeletion: false
        editable: true
        options:
          path: /var/lib/grafana/dashboards/federated-learning
  
  dashboards:
    federated-learning:
      fl-overview:
        gnetId: 1860
        revision: 27
        datasource: Prometheus
      
      fl-training:
        url: https://raw.githubusercontent.com/yourorg/fl-grafana-dashboards/main/training-dashboard.json
      
      fl-fairness:
        url: https://raw.githubusercontent.com/yourorg/fl-grafana-dashboards/main/fairness-dashboard.json

  grafana.ini:
    server:
      root_url: "https://grafana.$ENVIRONMENT.yourdomain.com"
    security:
      disable_initial_admin_creation: false
    auth:
      disable_login_form: false
    auth.anonymous:
      enabled: true
      org_role: Viewer
      hide_version: true

nodeExporter:
  enabled: true

kubeStateMetrics:
  enabled: true

defaultRules:
  create: true
  rules:
    alertmanager: true
    etcd: true
    general: true
    k8s: true
    kubeApiserver: true
    kubePrometheusNodeRecording: true
    kubernetesApps: true
    kubernetesResources: true
    kubernetesStorage: true
    kubernetesSystem: true
    node: true
    prometheus: true
    prometheusOperator: true
EOF

    # Install Prometheus stack
    helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace "$MONITORING_NAMESPACE" \
        --values /tmp/prometheus-values.yaml \
        --wait --timeout=600s
    
    log "Prometheus stack installed successfully"
}

setup_custom_dashboards() {
    log "Setting up custom Grafana dashboards..."
    
    # Create dashboard ConfigMaps
    kubectl apply -f - << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: fl-training-dashboard
  namespace: $MONITORING_NAMESPACE
  labels:
    grafana_dashboard: "true"
data:
  fl-training-dashboard.json: |
    {
      "dashboard": {
        "id": null,
        "title": "Federated Learning Training",
        "tags": ["federated-learning", "training"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Training Progress",
            "type": "graph",
            "targets": [
              {
                "expr": "fl_training_round",
                "legendFormat": "Current Round"
              },
              {
                "expr": "fl_global_accuracy",
                "legendFormat": "Global Accuracy"
              }
            ],
            "yAxes": [
              {"label": "Round Number", "min": 0},
              {"label": "Accuracy", "min": 0, "max": 1}
            ]
          },
          {
            "id": 2,
            "title": "Client Participation",
            "type": "stat",
            "targets": [
              {
                "expr": "fl_active_clients",
                "legendFormat": "Active Clients"
              }
            ]
          },
          {
            "id": 3,
            "title": "Model Performance",
            "type": "table",
            "targets": [
              {
                "expr": "fl_model_accuracy",
                "legendFormat": "{{client_id}}"
              }
            ]
          }
        ],
        "time": {
          "from": "now-1h",
          "to": "now"
        },
        "refresh": "10s"
      }
    }
EOF

    log "Custom dashboards configured"
}

setup_alerting_rules() {
    if [[ "$ENABLE_ALERTING" == "false" ]]; then
        log "Skipping alerting setup"
        return 0
    fi
    
    log "Setting up alerting rules..."
    
    # Create custom alerting rules
    kubectl apply -f - << EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: federated-learning-alerts
  namespace: $MONITORING_NAMESPACE
  labels:
    app: kube-prometheus-stack
    release: kube-prometheus-stack
spec:
  groups:
  - name: federated-learning.rules
    rules:
    # Server health alerts
    - alert: FLServerDown
      expr: up{job="federated-learning-server"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Federated Learning server is down"
        description: "FL server {{ \$labels.instance }} has been down for more than 1 minute"
    
    # Training progress alerts
    - alert: FLTrainingStalled
      expr: increase(fl_training_round[10m]) == 0 and fl_training_active == 1
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Federated Learning training appears stalled"
        description: "No training progress for 10 minutes"
    
    # Client participation alerts
    - alert: FLLowClientParticipation
      expr: fl_active_clients / fl_total_clients < 0.5
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Low client participation in federated learning"
        description: "Only {{ \$value }}% of clients are participating"
    
    # Performance alerts
    - alert: FLHighModelLoss
      expr: fl_global_loss > 1.0
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High model loss detected"
        description: "Global model loss is {{ \$value }}, which may indicate training issues"
    
    # Fairness alerts
    - alert: FLFairnessViolation
      expr: fl_fairness_score < 0.8
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "Model fairness violation detected"
        description: "Fairness score {{ \$value }} is below acceptable threshold"
    
    # Resource alerts
    - alert: FLHighMemoryUsage
      expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High memory usage on FL nodes"
        description: "Memory usage is {{ \$value }}% on {{ \$labels.instance }}"
EOF

    # Configure Alertmanager if Slack webhook provided
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        log "Configuring Slack notifications..."
        
        kubectl create secret generic alertmanager-slack-webhook \
            --from-literal=url="$SLACK_WEBHOOK_URL" \
            -n "$MONITORING_NAMESPACE" \
            --dry-run=client -o yaml | kubectl apply -f -
        
        # Create Alertmanager configuration
        kubectl apply -f - << EOF
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-kube-prometheus-stack-alertmanager
  namespace: $MONITORING_NAMESPACE
type: Opaque
stringData:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'localhost:587'
      smtp_from: 'alerts@yourdomain.com'
    
    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'default'
      routes:
      - match:
          severity: critical
        receiver: 'slack-critical'
      - match:
          severity: warning
        receiver: 'slack-warnings'
    
    receivers:
    - name: 'default'
      slack_configs:
      - api_url: '$SLACK_WEBHOOK_URL'
        channel: '#federated-learning-alerts'
        title: 'FL System Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    
    - name: 'slack-critical'
      slack_configs:
      - api_url: '$SLACK_WEBHOOK_URL'
        channel: '#federated-learning-critical'
        title: 'CRITICAL: FL System Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
        color: 'danger'
    
    - name: 'slack-warnings'
      slack_configs:
      - api_url: '$SLACK_WEBHOOK_URL'
        channel: '#federated-learning-alerts'
        title: 'Warning: FL System Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        color: 'warning'
EOF
    fi
    
    log "Alerting rules configured"
}

setup_jaeger_tracing() {
    if [[ "$ENABLE_TRACING" == "false" ]]; then
        log "Skipping tracing setup"
        return 0
    fi
    
    log "Setting up Jaeger distributed tracing..."
    
    # Install Jaeger operator
    kubectl create namespace observability --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Jaeger All-in-One for development, or production setup for prod
    if [[ "$ENVIRONMENT" == "development" ]]; then
        kubectl apply -f - << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: observability
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:1.48
        ports:
        - containerPort: 16686
        - containerPort: 14268
        - containerPort: 14250
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger-service
  namespace: observability
spec:
  selector:
    app: jaeger
  ports:
  - name: ui
    port: 16686
    targetPort: 16686
  - name: collector
    port: 14268
    targetPort: 14268
  - name: grpc
    port: 14250
    targetPort: 14250
EOF
    else
        # Production Jaeger setup with persistence
        helm upgrade --install jaeger jaegertracing/jaeger \
            --namespace observability \
            --set provisionDataStore.cassandra=false \
            --set provisionDataStore.elasticsearch=true \
            --set elasticsearch.replicas=3 \
            --set elasticsearch.minimumMasterNodes=2 \
            --wait --timeout=300s
    fi
    
    log "Jaeger tracing setup completed"
}

setup_logging() {
    log "Setting up centralized logging..."
    
    # Install Fluent Bit for log collection
    helm upgrade --install fluent-bit fluent/fluent-bit \
        --namespace "$MONITORING_NAMESPACE" \
        --set config.outputs="[OUTPUT]\n    Name es\n    Match *\n    Host elasticsearch-master\n    Port 9200\n    Index fl-logs" \
        --wait --timeout=300s
    
    log "Centralized logging configured"
}

setup_custom_metrics() {
    log "Setting up custom metrics collection..."
    
    # Create custom metrics ServiceMonitor
    kubectl apply -f - << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: federated-learning-metrics
  namespace: $MONITORING_NAMESPACE
  labels:
    app: kube-prometheus-stack
    release: kube-prometheus-stack
spec:
  selector:
    matchLabels:
      app: fl-server
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
  namespaceSelector:
    matchNames:
    - federated-learning
    ---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: federated-learning-client-metrics
  namespace: $MONITORING_NAMESPACE
  labels:
    app: kube-prometheus-stack
    release: kube-prometheus-stack
spec:
  selector:
    matchLabels:
      app: fl-client
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
  namespaceSelector:
    matchNames:
    - federated-learning
EOF

    log "Custom metrics collection configured"
}

setup_network_policies() {
    log "Setting up network policies for monitoring..."
    
    # Network policy for monitoring namespace
    kubectl apply -f - << EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: monitoring-network-policy
  namespace: $MONITORING_NAMESPACE
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: $MONITORING_NAMESPACE
    - namespaceSelector:
        matchLabels:
          name: federated-learning
  - from: []
    ports:
    - protocol: TCP
      port: 9090  # Prometheus
    - protocol: TCP
      port: 3000  # Grafana
    - protocol: TCP
      port: 9093  # Alertmanager
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
  - to:
    - namespaceSelector:
        matchLabels:
          name: federated-learning
    ports:
    - protocol: TCP
      port: 8082  # FL Server metrics
    - protocol: TCP
      port: 8083  # FL Client metrics
EOF

    log "Network policies configured"
}

create_monitoring_ingress() {
    log "Creating ingress for monitoring services..."
    
    kubectl apply -f - << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: monitoring-ingress
  namespace: $MONITORING_NAMESPACE
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: monitoring-basic-auth
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - grafana.$ENVIRONMENT.yourdomain.com
    - prometheus.$ENVIRONMENT.yourdomain.com
    - alertmanager.$ENVIRONMENT.yourdomain.com
    secretName: monitoring-tls-secret
  rules:
  - host: grafana.$ENVIRONMENT.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: kube-prometheus-stack-grafana
            port:
              number: 80
  - host: prometheus.$ENVIRONMENT.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: kube-prometheus-stack-prometheus
            port:
              number: 9090
  - host: alertmanager.$ENVIRONMENT.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: kube-prometheus-stack-alertmanager
            port:
              number: 9093
EOF

    # Create basic auth secret for monitoring access
    MONITORING_USERNAME="admin"
    MONITORING_PASSWORD=$(openssl rand -base64 12)
    
    kubectl create secret generic monitoring-basic-auth \
        --from-literal=auth=$(htpasswd -nb "$MONITORING_USERNAME" "$MONITORING_PASSWORD") \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    info "Monitoring access credentials:"
    info "Username: $MONITORING_USERNAME"
    info "Password: $MONITORING_PASSWORD"
    
    log "Monitoring ingress created"
}

setup_log_aggregation() {
    log "Setting up log aggregation with ELK stack..."
    
    # Install Elasticsearch
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace "$MONITORING_NAMESPACE" \
        --set replicas=3 \
        --set minimumMasterNodes=2 \
        --set resources.requests.cpu="1000m" \
        --set resources.requests.memory="2Gi" \
        --set resources.limits.cpu="2000m" \
        --set resources.limits.memory="4Gi" \
        --set volumeClaimTemplate.accessModes[0]="ReadWriteOnce" \
        --set volumeClaimTemplate.storageClassName="gp3" \
        --set volumeClaimTemplate.resources.requests.storage="30Gi" \
        --wait --timeout=600s
    
    # Install Kibana
    helm upgrade --install kibana elastic/kibana \
        --namespace "$MONITORING_NAMESPACE" \
        --set elasticsearchHosts="http://elasticsearch-master:9200" \
        --set resources.requests.cpu="500m" \
        --set resources.requests.memory="1Gi" \
        --set resources.limits.cpu="1000m" \
        --set resources.limits.memory="2Gi" \
        --wait --timeout=300s
    
    # Install Fluent Bit
    helm upgrade --install fluent-bit fluent/fluent-bit \
        --namespace "$MONITORING_NAMESPACE" \
        --set config.outputs="[OUTPUT]\n    Name es\n    Match *\n    Host elasticsearch-master\n    Port 9200\n    Index fl-logs\n    Type _doc" \
        --set config.filters="[FILTER]\n    Name kubernetes\n    Match kube.*\n    Kube_URL https://kubernetes.default.svc:443\n    Merge_Log On\n    K8S-Logging.Parser On\n    K8S-Logging.Exclude Off" \
        --wait --timeout=300s
    
    log "Log aggregation setup completed"
}

verify_monitoring_setup() {
    log "Verifying monitoring setup..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus -n "$MONITORING_NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana -n "$MONITORING_NAMESPACE" --timeout=300s
    
    # Check service endpoints
    local prometheus_svc=$(kubectl get svc kube-prometheus-stack-prometheus -n "$MONITORING_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    local grafana_svc=$(kubectl get svc kube-prometheus-stack-grafana -n "$MONITORING_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    if [[ -n "$prometheus_svc" && -n "$grafana_svc" ]]; then
        log "✓ Services are accessible"
        info "Prometheus: $prometheus_svc:9090"
        info "Grafana: $grafana_svc:3000"
    else
        error "✗ Services not accessible"
    fi
    
    # Check metrics collection
    log "Checking metrics collection..."
    kubectl exec -n "$MONITORING_NAMESPACE" deployment/prometheus-kube-prometheus-stack-prometheus -- \
        wget -qO- http://localhost:9090/api/v1/query?query=up | grep -q "success" && \
        log "✓ Prometheus metrics collection working" || \
        warn "✗ Prometheus metrics may not be working correctly"
    
    log "Monitoring verification completed"
}

generate_monitoring_docs() {
    log "Generating monitoring documentation..."
    
    local docs_dir="$PROJECT_ROOT/docs/monitoring"
    mkdir -p "$docs_dir"
    
    cat > "$docs_dir/monitoring-guide.md" << EOF
# Federated Learning System Monitoring Guide

## Overview
This guide covers the monitoring and observability setup for the Federated Learning System.

## Access Information

### Grafana Dashboard
- URL: https://grafana.$ENVIRONMENT.yourdomain.com
- Username: admin
- Password: $GRAFANA_ADMIN_PASSWORD

### Prometheus
- URL: https://prometheus.$ENVIRONMENT.yourdomain.com

### Alertmanager
- URL: https://alertmanager.$ENVIRONMENT.yourdomain.com

## Key Metrics to Monitor

### Training Metrics
- \`fl_training_round\`: Current training round number
- \`fl_global_accuracy\`: Global model accuracy
- \`fl_global_loss\`: Global model loss
- \`fl_convergence_rate\`: Model convergence rate

### Client Metrics
- \`fl_active_clients\`: Number of active clients
- \`fl_client_participation_rate\`: Client participation percentage
- \`fl_client_data_sizes\`: Distribution of client data sizes

### Performance Metrics
- \`fl_aggregation_time\`: Time taken for model aggregation
- \`fl_communication_overhead\`: Communication overhead per round
- \`fl_model_size_mb\`: Model size in megabytes

### Fairness Metrics
- \`fl_fairness_score\`: Overall fairness score
- \`fl_demographic_parity\`: Demographic parity score
- \`fl_equalized_odds\`: Equalized odds score

### Privacy Metrics
- \`fl_privacy_budget_used\`: Privacy budget utilization
- \`fl_noise_level\`: Current noise level for differential privacy

## Alert Thresholds

### Critical Alerts
- Server down for > 1 minute
- Fairness score < 0.8
- Privacy budget exhausted

### Warning Alerts
- Training stalled for > 10 minutes
- Client participation < 50%
- High resource usage (> 90%)

## Troubleshooting

### Common Issues
1. **High Memory Usage**: Scale up nodes or optimize model size
2. **Low Client Participation**: Check client connectivity and data availability
3. **Training Stalled**: Check for resource constraints or client failures
4. **Fairness Violations**: Review data distribution and bias mitigation strategies

### Useful Commands
\`\`\`bash
# Check pod status
kubectl get pods -n federated-learning

# View logs
kubectl logs deployment/fl-server -n federated-learning

# Check metrics
kubectl port-forward -n monitoring service/kube-prometheus-stack-prometheus 9090:9090

# Access Grafana locally
kubectl port-forward -n monitoring service/kube-prometheus-stack-grafana 3000:80
\`\`\`

## Dashboard Customization

Custom dashboards can be added by creating ConfigMaps with the label \`grafana_dashboard: "true"\`.

See the example dashboard configurations in the monitoring setup.
EOF

    log "Monitoring documentation generated: $docs_dir/monitoring-guide.md"
}

display_access_information() {
    log "Deployment completed! Access information:"
    
    echo
    info "=== MONITORING ACCESS ==="
    
    case $ENVIRONMENT in
        development)
            info "Grafana: kubectl port-forward -n $MONITORING_NAMESPACE service/kube-prometheus-stack-grafana 3000:80"
            info "Prometheus: kubectl port-forward -n $MONITORING_NAMESPACE service/kube-prometheus-stack-prometheus 9090:9090"
            info "Alertmanager: kubectl port-forward -n $MONITORING_NAMESPACE service/kube-prometheus-stack-alertmanager 9093:9093"
            ;;
        staging|production)
            info "Grafana: https://grafana.$ENVIRONMENT.yourdomain.com"
            info "Prometheus: https://prometheus.$ENVIRONMENT.yourdomain.com"
            info "Alertmanager: https://alertmanager.$ENVIRONMENT.yourdomain.com"
            ;;
    esac
    
    echo
    info "=== CREDENTIALS ==="
    info "Grafana Admin Password: $GRAFANA_ADMIN_PASSWORD"
    
    if [[ "$ENABLE_TRACING" == "true" ]]; then
        echo
        info "=== TRACING ==="
        info "Jaeger UI: kubectl port-forward -n observability service/jaeger-service 16686:16686"
    fi
    
    echo
    info "=== NEXT STEPS ==="
    info "1. Import custom dashboards in Grafana"
    info "2. Configure alert channels in Alertmanager"
    info "3. Review and adjust alert thresholds"
    info "4. Set up log retention policies"
    info "5. Configure backup for monitoring data"
    
    echo
    warn "IMPORTANT: Save the Grafana admin password in a secure location!"
}

cleanup_monitoring() {
    log "Cleaning up monitoring resources..."
    
    # Remove temporary files
    rm -f /tmp/prometheus-values.yaml
    
    log "Cleanup completed"
}

main() {
    log "Starting Federated Learning System monitoring setup"
    log "Environment: $ENVIRONMENT"
    log "Namespace: $MONITORING_NAMESPACE"
    log "Alerting: $ENABLE_ALERTING"
    log "Tracing: $ENABLE_TRACING"
    
    # Prerequisites
    check_prerequisites
    
    # Core monitoring setup
    setup_prometheus_stack
    setup_custom_dashboards
    setup_custom_metrics
    
    # Optional components
    setup_alerting_rules
    setup_jaeger_tracing
    setup_logging
    
    # Security and networking
    setup_network_policies
    create_monitoring_ingress
    
    # Verification
    verify_monitoring_setup
    
    # Documentation
    generate_monitoring_docs
    
    # Display access information
    display_access_information
    
    log "Monitoring setup completed successfully!"
}

# Trap for cleanup
trap cleanup_monitoring EXIT

# Parse arguments and run main function
parse_args "$@"
main
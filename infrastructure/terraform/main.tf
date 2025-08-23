# Terraform Infrastructure for Federated Learning System

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
  
  backend "s3" {
    bucket         = "fl-terraform-state"
    key            = "federated-learning/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "fl-terraform-locks"
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "federated-learning"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "fl-team"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Data sources
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

# Local values
locals {
  cluster_name = "${var.project_name}-${var.environment}"
  
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)
  
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]
  
  database_subnets = ["10.0.7.0/24", "10.0.8.0/24", "10.0.9.0/24"]
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Terraform   = "true"
  }
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${local.cluster_name}-vpc"
  cidr = local.vpc_cidr
  
  azs             = local.azs
  private_subnets = local.private_subnets
  public_subnets  = local.public_subnets
  database_subnets = local.database_subnets
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  # Required for EKS
  public_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"             = "1"
  }
  
  tags = local.common_tags
}

# Security Groups
resource "aws_security_group" "additional" {
  name_prefix = "${local.cluster_name}-additional"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    cidr_blocks = [local.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-additional-sg"
  })
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.16"
  
  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version
  
  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  
  # OIDC Identity provider
  cluster_identity_providers = {
    sts = {
      client_id = "sts.amazonaws.com"
    }
  }
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # Server node group with high-memory instances
    server_nodes = {
      name = "fl-server-nodes"
      
      instance_types = ["m5.2xlarge"]
      min_size       = 1
      max_size       = 5
      desired_size   = 2
      
      capacity_type = "ON_DEMAND"
      
      labels = {
        role = "server"
        workload = "federated-learning-server"
      }
      
      taints = {
        server = {
          key    = "fl-server"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
      
      tags = merge(local.common_tags, {
        Name = "${local.cluster_name}-server-nodes"
      })
    }
    
    # Client node group with CPU-optimized instances
    client_nodes = {
      name = "fl-client-nodes"
      
      instance_types = ["c5.xlarge", "c5.2xlarge"]
      min_size       = 2
      max_size       = 20
      desired_size   = 3
      
      capacity_type = "SPOT"  # Use spot instances for cost optimization
      
      labels = {
        role = "client"
        workload = "federated-learning-client"
      }
      
      tags = merge(local.common_tags, {
        Name = "${local.cluster_name}-client-nodes"
      })
    }
    
    # GPU node group for ML workloads (optional)
    gpu_nodes = {
      name = "fl-gpu-nodes"
      
      instance_types = ["g4dn.xlarge"]
      min_size       = 0
      max_size       = 10
      desired_size   = 0  # Start with 0, scale up as needed
      
      capacity_type = "SPOT"
      
      ami_type = "AL2_x86_64_GPU"
      
      labels = {
        role = "gpu-worker"
        workload = "federated-learning-gpu"
        nvidia.com/gpu = "true"
      }
      
      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
      
      tags = merge(local.common_tags, {
        Name = "${local.cluster_name}-gpu-nodes"
      })
    }
  }
  
  # aws-auth configmap
  manage_aws_auth_configmap = true
  
  aws_auth_roles = [
    {
      rolearn  = module.eks_admins_iam_role.iam_role_arn
      username = "admin"
      groups   = ["system:masters"]
    },
  ]
  
  aws_auth_users = var.map_users
  
  tags = local.common_tags
}

# EKS Add-ons
resource "aws_eks_addon" "addons" {
  for_each = {
    coredns = {
      version = "v1.10.1-eksbuild.2"
    }
    kube-proxy = {
      version = "v1.27.5-eksbuild.1"
    }
    vpc-cni = {
      version = "v1.13.4-eksbuild.1"
    }
    aws-ebs-csi-driver = {
      version = "v1.21.0-eksbuild.1"
    }
  }
  
  cluster_name             = module.eks.cluster_name
  addon_name               = each.key
  addon_version            = each.value.version
  resolve_conflicts        = "OVERWRITE"
  service_account_role_arn = each.key == "aws-ebs-csi-driver" ? aws_iam_role.ebs_csi_driver.arn : null
  
  depends_on = [module.eks]
}

# IAM Role for EBS CSI Driver
resource "aws_iam_role" "ebs_csi_driver" {
  name = "${local.cluster_name}-ebs-csi-driver"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ebs_csi_driver_policy" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/Amazon_EBS_CSI_DriverPolicy"
  role       = aws_iam_role.ebs_csi_driver.name
}

# RDS Instance for application database
resource "aws_db_subnet_group" "database" {
  name       = "${local.cluster_name}-db-subnet-group"
  subnet_ids = module.vpc.database_subnets
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-db-subnet-group"
  })
}

resource "aws_security_group" "rds" {
  name_prefix = "${local.cluster_name}-rds"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-rds-sg"
  })
}

resource "random_password" "db_password" {
  length  = 16
  special = true
}

resource "aws_secretsmanager_secret" "db_password" {
  name = "${local.cluster_name}-db-password"
  
  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db_password.result
}

resource "aws_db_instance" "postgres" {
  identifier = "${local.cluster_name}-postgres"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp3"
  storage_encrypted     = true
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.db_instance_class
  
  db_name  = "federated_learning"
  username = "fl_admin"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.database.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${local.cluster_name}-postgres-final-snapshot"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-postgres"
  })
}

# IAM Role for RDS monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${local.cluster_name}-rds-monitoring"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ElastiCache Redis Cluster
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.cluster_name}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis-subnet-group"
  })
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.cluster_name}-redis"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis-sg"
  })
}

resource "aws_elasticache_replication_group" "redis" {
  description          = "Redis cluster for federated learning"
  replication_group_id = "${var.project_name}-${var.environment}-redis"
  
  port               = 6379
  parameter_group_name = aws_elasticache_parameter_group.redis.name
  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]
  
  num_cache_clusters = 2
  node_type          = var.redis_node_type
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth.result
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  tags = local.common_tags
}

resource "aws_elasticache_parameter_group" "redis" {
  name   = "${local.cluster_name}-redis-params"
  family = "redis7.x"
  
  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }
}

resource "random_password" "redis_auth" {
  length  = 32
  special = true
}

# S3 Bucket for model artifacts and data
resource "aws_s3_bucket" "fl_artifacts" {
  bucket = "${var.project_name}-${var.environment}-artifacts"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_server_side_encryption_configuration" "fl_artifacts" {
  bucket = aws_s3_bucket.fl_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_versioning" "fl_artifacts" {
  bucket = aws_s3_bucket.fl_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "fl_artifacts" {
  bucket = aws_s3_bucket.fl_artifacts.id
  
  rule {
    id     = "model_artifacts_lifecycle"
    status = "Enabled"
    
    expiration {
      days = 90
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "fl_server" {
  name              = "/aws/eks/${local.cluster_name}/fl-server"
  retention_in_days = 14
  
  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "fl_client" {
  name              = "/aws/eks/${local.cluster_name}/fl-client"
  retention_in_days = 7
  
  tags = local.common_tags
}

# IAM Roles and Policies
module "eks_admins_iam_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.30"
  
  role_name = "${local.cluster_name}-eks-admins"
  
  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }
  
  tags = local.common_tags
}

# AWS Load Balancer Controller IAM Role
module "load_balancer_controller_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.30"
  
  role_name = "${local.cluster_name}-aws-load-balancer-controller"
  
  attach_load_balancer_controller_policy = true
  
  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }
  
  tags = local.common_tags
}

# Helm Release for AWS Load Balancer Controller
resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.6.1"
  
  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }
  
  set {
    name  = "serviceAccount.create"
    value = "false"
  }
  
  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }
  
  set {
    name  = "region"
    value = var.aws_region
  }
  
  set {
    name  = "vpcId"
    value = module.vpc.vpc_id
  }
  
  depends_on = [
    kubernetes_service_account.aws_load_balancer_controller
  ]
}

resource "kubernetes_service_account" "aws_load_balancer_controller" {
  metadata {
    name      = "aws-load-balancer-controller"
    namespace = "kube-system"
    annotations = {
      "eks.amazonaws.com/role-arn" = module.load_balancer_controller_irsa_role.iam_role_arn
    }
    labels = {
      "app.kubernetes.io/component" = "controller"
      "app.kubernetes.io/name"      = "aws-load-balancer-controller"
    }
  }
}

# Cluster Autoscaler
resource "helm_release" "cluster_autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  namespace  = "kube-system"
  version    = "9.29.0"
  
  set {
    name  = "autoDiscovery.clusterName"
    value = module.eks.cluster_name
  }
  
  set {
    name  = "awsRegion"
    value = var.aws_region
  }
  
  set {
    name  = "rbac.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.cluster_autoscaler_irsa_role.iam_role_arn
  }
  
  depends_on = [module.eks]
}

module "cluster_autoscaler_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.30"
  
  role_name                        = "${local.cluster_name}-cluster-autoscaler"
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [module.eks.cluster_name]
  
  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }
  
  tags = local.common_tags
}

# Prometheus and Grafana via Helm
resource "helm_release" "kube_prometheus_stack" {
  name             = "kube-prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true
  version          = "51.3.0"
  
  values = [
    file("${path.module}/helm-values/prometheus-values.yaml")
  ]
  
  set {
    name  = "grafana.adminPassword"
    value = random_password.grafana_admin.result
  }
  
  depends_on = [module.eks]
}

resource "random_password" "grafana_admin" {
  length  = 16
  special = true
}

# EFS for shared storage
resource "aws_efs_file_system" "fl_shared_storage" {
  creation_token = "${local.cluster_name}-shared-storage"
  
  performance_mode = "generalPurpose"
  throughput_mode  = "provisioned"
  provisioned_throughput_in_mibps = 500
  
  encrypted = true
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-shared-storage"
  })
}

resource "aws_efs_mount_target" "fl_shared_storage" {
  count = length(module.vpc.private_subnets)
  
  file_system_id  = aws_efs_file_system.fl_shared_storage.id
  subnet_id       = module.vpc.private_subnets[count.index]
  security_groups = [aws_security_group.efs.id]
}

resource "aws_security_group" "efs" {
  name_prefix = "${local.cluster_name}-efs"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-efs-sg"
  })
}

# Output important information
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_name" {
  description = "The name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  sensitive   = true
}

output "s3_bucket" {
  description = "S3 bucket for artifacts"
  value       = aws_s3_bucket.fl_artifacts.bucket
}

output "efs_file_system_id" {
  description = "EFS file system ID for shared storage"
  value       = aws_efs_file_system.fl_shared_storage.id
}
# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

locals {
  azs            = slice(data.aws_availability_zones.available.names, 0, length(var.private_subnet_cidrs))
  create_vpc     = var.vpc_id == ""
  vpc_id         = local.create_vpc ? module.vpc[0].vpc_id : var.vpc_id
  private_subnets = local.create_vpc ? module.vpc[0].private_subnets : data.aws_subnets.existing_private[0].ids
  public_subnets  = local.create_vpc ? module.vpc[0].public_subnets : data.aws_subnets.existing_public[0].ids
}

# ---------------------------------------------------------------------------
# Existing-VPC subnet lookups (only when vpc_id is provided)
# ---------------------------------------------------------------------------

data "aws_subnets" "existing_private" {
  count = local.create_vpc ? 0 : 1

  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }

  tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

data "aws_subnets" "existing_public" {
  count = local.create_vpc ? 0 : 1

  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }

  tags = {
    "kubernetes.io/role/elb" = "1"
  }
}

# ---------------------------------------------------------------------------
# VPC (created only when var.vpc_id is empty)
# ---------------------------------------------------------------------------

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  count   = local.create_vpc ? 1 : 0

  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr
  azs  = local.azs

  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb"                      = "1"
    "kubernetes.io/cluster/${var.cluster_name}"    = "owned"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"             = "1"
    "kubernetes.io/cluster/${var.cluster_name}"    = "owned"
  }

  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}

# ---------------------------------------------------------------------------
# EKS Cluster
# ---------------------------------------------------------------------------

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id     = local.vpc_id
  subnet_ids = local.private_subnets

  cluster_endpoint_public_access = true

  cluster_addons = {
    coredns                = { most_recent = true }
    kube-proxy             = { most_recent = true }
    vpc-cni                = { most_recent = true }
    aws-ebs-csi-driver     = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
  }

  eks_managed_node_groups = {
    hebbs = {
      instance_types = [var.instance_type]
      desired_size   = var.node_count
      min_size       = var.node_min_size
      max_size       = var.node_max_size

      labels = {
        "hebbs.dev/role" = "storage"
      }

      iam_role_additional_policies = {
        AmazonEBSCSIDriverPolicy = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
      }
    }
  }

  tags = {
    Name = var.cluster_name
  }
}

# ---------------------------------------------------------------------------
# IRSA for EBS CSI Driver
# ---------------------------------------------------------------------------

module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.cluster_name}-ebs-csi-driver"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = {
    Name = "${var.cluster_name}-ebs-csi-irsa"
  }
}

# ---------------------------------------------------------------------------
# StorageClass: gp3 tuned for RocksDB
# ---------------------------------------------------------------------------

resource "kubernetes_storage_class_v1" "gp3_hebbs" {
  metadata {
    name = var.hebbs_storage_class
    annotations = {
      "storageclass.kubernetes.io/is-default-class" = "false"
    }
  }

  storage_provisioner    = "ebs.csi.aws.com"
  volume_binding_mode    = "WaitForFirstConsumer"
  allow_volume_expansion = true
  reclaim_policy         = "Retain"

  parameters = {
    type       = "gp3"
    fsType     = "ext4"
    iops       = tostring(var.ebs_iops)
    throughput = tostring(var.ebs_throughput)
    encrypted  = "true"
  }

  depends_on = [module.eks]
}

# ---------------------------------------------------------------------------
# IRSA for AWS Load Balancer Controller
# ---------------------------------------------------------------------------

module "lb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                              = "${var.cluster_name}-aws-lb-controller"
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }

  tags = {
    Name = "${var.cluster_name}-lb-controller-irsa"
  }
}

# ---------------------------------------------------------------------------
# AWS Load Balancer Controller (Helm)
# ---------------------------------------------------------------------------

resource "helm_release" "aws_lb_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.7.2"

  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "serviceAccount.create"
    value = "true"
  }

  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }

  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.lb_controller_irsa.iam_role_arn
  }

  set {
    name  = "region"
    value = var.region
  }

  set {
    name  = "vpcId"
    value = local.vpc_id
  }

  depends_on = [module.eks]
}

# ---------------------------------------------------------------------------
# HEBBS (Helm)
# ---------------------------------------------------------------------------

resource "helm_release" "hebbs" {
  name      = "hebbs"
  chart     = "${path.module}/../../helm/hebbs"
  namespace = "hebbs"

  create_namespace = true

  set {
    name  = "image.tag"
    value = var.hebbs_version
  }

  set {
    name  = "storage.className"
    value = var.hebbs_storage_class
  }

  set {
    name  = "storage.size"
    value = var.hebbs_storage_size
  }

  set {
    name  = "ingress.enabled"
    value = "true"
  }

  set {
    name  = "ingress.className"
    value = "alb"
  }

  set {
    name  = "ingress.annotations.alb\\.ingress\\.kubernetes\\.io/scheme"
    value = "internet-facing"
  }

  set {
    name  = "ingress.annotations.alb\\.ingress\\.kubernetes\\.io/target-type"
    value = "ip"
  }

  set {
    name  = "ingress.annotations.alb\\.ingress\\.kubernetes\\.io/backend-protocol-version"
    value = "GRPC"
  }

  set {
    name  = "ingress.annotations.alb\\.ingress\\.kubernetes\\.io/listen-ports"
    value = "[{\"HTTPS\":443}\\,{\"HTTP\":80}]"
  }

  depends_on = [
    kubernetes_storage_class_v1.gp3_hebbs,
    helm_release.aws_lb_controller,
  ]
}

# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

variable "cluster_name" {
  description = "Name of the EKS cluster. Also used as a prefix for dependent resources (VPC, node group, IAM roles)."
  type        = string
  default     = "hebbs"
}

variable "region" {
  description = "AWS region where all resources are created."
  type        = string
  default     = "us-west-2"
}

variable "kubernetes_version" {
  description = "Kubernetes control-plane version for the EKS cluster. Must match an EKS-supported release."
  type        = string
  default     = "1.29"
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

variable "vpc_id" {
  description = <<-EOT
    ID of an existing VPC. When empty (default), the module creates a new VPC
    using var.vpc_cidr. Supply this when you need HEBBS to live inside a shared
    or pre-provisioned network.
  EOT
  type        = string
  default     = ""
}

variable "vpc_cidr" {
  description = "CIDR block for the new VPC. Ignored when var.vpc_id is set."
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = <<-EOT
    CIDR blocks for private subnets (one per AZ). Worker nodes and pods run here.
    Defaults cover two AZs. Add a third element for three-AZ deployments.
  EOT
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "public_subnet_cidrs" {
  description = <<-EOT
    CIDR blocks for public subnets (one per AZ). NAT gateways and the ALB live
    here. Must match the length of var.private_subnet_cidrs.
  EOT
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}

# ---------------------------------------------------------------------------
# Node Group
# ---------------------------------------------------------------------------

variable "instance_type" {
  description = <<-EOT
    EC2 instance type for the managed node group.
    Sizing guidance:
      - m6i.large  (2 vCPU / 8 GiB)  — dev / low-traffic
      - m6i.xlarge (4 vCPU / 16 GiB)  — production baseline
      - r6i.xlarge (4 vCPU / 32 GiB)  — large memory stores (>5M memories)
    HEBBS is CPU + I/O bound; memory-optimised instances help only at scale.
  EOT
  type        = string
  default     = "m6i.large"
}

variable "node_count" {
  description = "Desired number of nodes in the managed node group."
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum number of nodes the auto-scaler may scale down to."
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes the auto-scaler may scale up to."
  type        = number
  default     = 4
}

# ---------------------------------------------------------------------------
# HEBBS Application
# ---------------------------------------------------------------------------

variable "hebbs_version" {
  description = "HEBBS container image tag / chart appVersion to deploy."
  type        = string
  default     = "0.1.0"
}

variable "hebbs_storage_size" {
  description = "Size of the PersistentVolumeClaim for RocksDB data. 20 Gi supports ~5M memories."
  type        = string
  default     = "20Gi"
}

variable "hebbs_storage_class" {
  description = "Name of the Kubernetes StorageClass created for HEBBS EBS volumes."
  type        = string
  default     = "gp3-hebbs"
}

# ---------------------------------------------------------------------------
# EBS Storage Tuning
# ---------------------------------------------------------------------------

variable "ebs_iops" {
  description = <<-EOT
    Provisioned IOPS for the gp3 EBS volumes backing RocksDB.
    3000 IOPS is the gp3 baseline (free). Increase for write-heavy workloads;
    maximum is 16000 for gp3.
  EOT
  type        = number
  default     = 3000
}

variable "ebs_throughput" {
  description = <<-EOT
    Provisioned throughput (MiB/s) for gp3 EBS volumes.
    125 MiB/s is the gp3 baseline (free). Increase up to 1000 MiB/s for
    compaction-heavy workloads.
  EOT
  type        = number
  default     = 125
}

# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

variable "tags" {
  description = "Map of tags applied to every resource via the AWS provider default_tags."
  type        = map(string)
  default = {
    Project   = "hebbs"
    ManagedBy = "terraform"
  }
}

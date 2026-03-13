# HEBBS on AWS EKS — Terraform Module

Deploy a production-ready HEBBS cognitive memory server on Amazon EKS with gp3-backed RocksDB storage and ALB ingress for gRPC.

## Prerequisites

| Tool | Minimum version |
|------|-----------------|
| Terraform | >= 1.5.0 |
| AWS CLI | v2 |
| kubectl | >= 1.29 |
| Helm | >= 3.12 |

You also need an AWS account with permissions to create VPCs, EKS clusters, IAM roles, and EBS volumes.

```bash
aws configure   # set credentials + default region
```

## Quick Start

```bash
cd hebbs-deploy/terraform/aws

terraform init
terraform plan -out=plan.tfplan
terraform apply plan.tfplan
```

After apply completes (~15 minutes), configure kubectl:

```bash
$(terraform output -raw kubeconfig_command)
```

Verify the cluster:

```bash
kubectl get nodes
kubectl -n hebbs get pods
```

## Using an Existing VPC

Pass the VPC ID to skip VPC creation. The existing VPC must have subnets tagged for EKS:

```bash
terraform apply -var="vpc_id=vpc-0abc123def456"
```

Required subnet tags:
- Private subnets: `kubernetes.io/role/internal-elb = 1`
- Public subnets: `kubernetes.io/role/elb = 1`

## Customisation

Override defaults with `-var` flags or a `terraform.tfvars` file:

```hcl
cluster_name    = "hebbs-prod"
region          = "eu-west-1"
instance_type   = "m6i.xlarge"
node_count      = 3
node_max_size   = 6
hebbs_storage_size = "50Gi"
ebs_iops        = 6000
ebs_throughput  = 250
```

## Accessing HEBBS

The ALB endpoint is provisioned by the AWS Load Balancer Controller. Retrieve it with:

```bash
kubectl -n hebbs get ingress -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}'
```

gRPC clients connect on port 443; HTTP health checks on port 80.

## Cleanup

```bash
terraform destroy
```

This removes all AWS resources including the EKS cluster, node group, VPC, and IAM roles. PersistentVolumes with `Retain` policy are preserved as unattached EBS volumes — delete them manually if no longer needed.

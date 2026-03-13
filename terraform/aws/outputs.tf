output "cluster_endpoint" {
  description = "EKS cluster API server endpoint."
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "Name of the EKS cluster."
  value       = module.eks.cluster_name
}

output "kubeconfig_command" {
  description = "Run this command to configure kubectl for the HEBBS cluster."
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "hebbs_grpc_endpoint" {
  description = "gRPC endpoint for the HEBBS server (available after ALB provisioning completes)."
  value       = "grpc://${try(helm_release.hebbs.metadata[0].name, "hebbs")}.${var.region}.elb.amazonaws.com:443"
}

output "hebbs_http_endpoint" {
  description = "HTTP endpoint for the HEBBS server health and admin APIs."
  value       = "http://${try(helm_release.hebbs.metadata[0].name, "hebbs")}.${var.region}.elb.amazonaws.com:80"
}

output "storage_class_name" {
  description = "Name of the Kubernetes StorageClass created for RocksDB volumes."
  value       = kubernetes_storage_class_v1.gp3_hebbs.metadata[0].name
}

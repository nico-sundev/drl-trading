# Outputs from ephemeral ECS deployment

output "service_url" {
  description = "URL of the deployed service (ALB DNS name)"
  value       = "http://${aws_lb.main.dns_name}"
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Route53 zone ID of the ALB"
  value       = aws_lb.main.zone_id
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.main.name
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = data.aws_ecs_cluster.main.cluster_name
}

output "log_group_name" {
  description = "CloudWatch Log Group name for service logs"
  value       = aws_cloudwatch_log_group.main.name
}

output "security_group_id" {
  description = "Security group ID for ECS tasks"
  value       = aws_security_group.ecs_tasks.id
}

output "env_id" {
  description = "Environment identifier"
  value       = var.env_id
}

output "deployed_image" {
  description = "Docker image deployed"
  value       = "${data.aws_ecr_repository.service.repository_url}:${var.image_tag}"
}

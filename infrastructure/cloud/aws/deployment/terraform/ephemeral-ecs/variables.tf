# Input variables for ephemeral ECS deployment

variable "service_name" {
  description = "Name of the service to deploy (e.g., drl-trading-preprocess)"
  type        = string
}

variable "env_id" {
  description = "Unique identifier for this ephemeral environment (e.g., e2e-abc123)"
  type        = string

  validation {
    condition     = can(regex("^e2e-[a-z0-9]+$", var.env_id))
    error_message = "env_id must match pattern: e2e-<alphanumeric>"
  }
}

variable "git_commit_sha" {
  description = "Git commit SHA that triggered this deployment"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag to deploy (usually git commit SHA)"
  type        = string
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID where resources will be created"
  type        = string
}

variable "ecs_cluster_name" {
  description = "Name of the existing ECS cluster"
  type        = string
  default     = "drl-trading-cluster"
}

variable "container_port" {
  description = "Port exposed by the container"
  type        = number
  default     = 8080
}

variable "health_check_path" {
  description = "HTTP path for health checks"
  type        = string
  default     = "/health"
}

variable "task_cpu" {
  description = "CPU units for the ECS task (1024 = 1 vCPU)"
  type        = string
  default     = "512"
}

variable "task_memory" {
  description = "Memory (MB) for the ECS task"
  type        = string
  default     = "1024"
}

# Terraform configuration for ephemeral E2E testing ECS environment
# Creates temporary ECS service, task definition, ALB, and security groups
# Designed to be created/destroyed per CI pipeline run

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Use S3 backend with dynamic key based on environment ID
  backend "s3" {
    bucket         = "drl-trading-terraform-state"
    key            = "ephemeral-ecs/${var.env_id}/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "drl-trading-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "drl-trading"
      Environment = "ephemeral-e2e"
      ManagedBy   = "terraform"
      EnvId       = var.env_id
      GitCommit   = var.git_commit_sha
      CreatedBy   = "gitlab-ci"
      AutoCleanup = "true"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}

data "aws_ecr_repository" "service" {
  name = var.service_name
}

data "aws_ecs_cluster" "main" {
  cluster_name = var.ecs_cluster_name
}

data "aws_vpc" "main" {
  id = var.vpc_id
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }

  tags = {
    Tier = "private"
  }
}

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }

  tags = {
    Tier = "public"
  }
}

# Security Groups
resource "aws_security_group" "alb" {
  name_prefix = "${var.service_name}-e2e-alb-"
  description = "Security group for ephemeral E2E ALB (${var.env_id})"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${var.service_name}-e2e-alb-${var.env_id}"
  }
}

resource "aws_security_group" "ecs_tasks" {
  name_prefix = "${var.service_name}-e2e-tasks-"
  description = "Security group for ephemeral E2E ECS tasks (${var.env_id})"
  vpc_id      = var.vpc_id

  ingress {
    description     = "HTTP from ALB"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${var.service_name}-e2e-tasks-${var.env_id}"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = substr("${var.service_name}-e2e-${var.env_id}", 0, 32)
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.public.ids

  enable_deletion_protection = false

  tags = {
    Name = "${var.service_name}-e2e-${var.env_id}"
  }
}

resource "aws_lb_target_group" "main" {
  name        = substr("${var.service_name}-e2e-${var.env_id}", 0, 32)
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = var.health_check_path
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }

  deregistration_delay = 30

  tags = {
    Name = "${var.service_name}-e2e-${var.env_id}"
  }
}

resource "aws_lb_listener" "main" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.main.arn
  }
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_task_execution" {
  name_prefix = "${var.service_name}-e2e-exec-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.service_name}-e2e-execution-${var.env_id}"
  }
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Role for ECS Task (application permissions)
resource "aws_iam_role" "ecs_task" {
  name_prefix = "${var.service_name}-e2e-task-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.service_name}-e2e-task-${var.env_id}"
  }
}

# Add necessary permissions for the application
resource "aws_iam_role_policy" "ecs_task" {
  name_prefix = "${var.service_name}-e2e-task-"
  role        = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::drl-trading-data-*",
          "arn:aws:s3:::drl-trading-data-*/*"
        ]
      }
    ]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "main" {
  name              = "/ecs/ephemeral/${var.service_name}/${var.env_id}"
  retention_in_days = 1  # Short retention for ephemeral logs

  tags = {
    Name = "${var.service_name}-e2e-${var.env_id}"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "main" {
  family                   = "${var.service_name}-e2e-${var.env_id}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = var.service_name
      image     = "${data.aws_ecr_repository.service.repository_url}:${var.image_tag}"
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "STAGE"
          value = "e2e"
        },
        {
          name  = "ENV_ID"
          value = var.env_id
        },
        {
          name  = "LOG_LEVEL"
          value = "DEBUG"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.main.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}${var.health_check_path} || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Name = "${var.service_name}-e2e-${var.env_id}"
  }
}

# ECS Service
resource "aws_ecs_service" "main" {
  name            = "${var.service_name}-e2e-${var.env_id}"
  cluster         = data.aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.private.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.main.arn
    container_name   = var.service_name
    container_port   = var.container_port
  }

  # Wait for ALB to be ready before creating service
  depends_on = [aws_lb_listener.main]

  # Faster deployment for ephemeral environments
  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 50
  }

  # Enable circuit breaker for faster failure detection
  deployment_circuit_breaker {
    enable   = true
    rollback = false  # Don't rollback, just fail fast
  }

  tags = {
    Name = "${var.service_name}-e2e-${var.env_id}"
  }
}

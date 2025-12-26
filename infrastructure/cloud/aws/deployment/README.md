# Ephemeral ECS E2E Testing Infrastructure

This directory contains Infrastructure as Code (Terraform) for deploying temporary ECS environments for end-to-end testing in CI/CD pipelines.

## Overview

The ephemeral infrastructure automatically:
- Deploys a complete ECS service with ALB, security groups, and task definitions
- Uses unique identifiers per CI pipeline run to avoid conflicts
- Automatically cleans up after tests complete
- Provides comprehensive logging and monitoring

## Components

### Terraform Configuration

**Location:** `terraform/ephemeral-ecs/`

- `main.tf` - Core infrastructure (ECS, ALB, security groups, IAM)
- `variables.tf` - Input variables and validation
- `outputs.tf` - Service URL and resource identifiers

**Resources Created:**
- Application Load Balancer (public-facing)
- Target Group with health checks
- Security Groups (ALB + ECS tasks)
- ECS Task Definition (Fargate)
- ECS Service
- IAM Roles (task execution + task)
- CloudWatch Log Group (1-day retention)

### Deployment Scripts

**deploy-ephemeral-env.sh**
- Provisions complete infrastructure using Terraform
- Waits for service to become healthy
- Validates health endpoint responds
- Returns service URL for testing

**cleanup-ephemeral-env.sh**
- Destroys all Terraform-managed resources
- Handles orphaned resources via AWS tags
- Cleans up remote state from S3
- Safe to run multiple times (idempotent)

## Usage in CI/CD

The GitLab CI pipeline automatically:

1. **Build Stage:** Creates Docker image on master branch
2. **E2E Stage:**
   - Deploys ephemeral environment
   - Runs E2E tests against live service
   - Cleans up automatically (even on failure)

Example pipeline job:
```yaml
e2e:preprocess:
  stage: e2e
  script:
    - ./infrastructure/cloud/aws/deployment/deploy-ephemeral-env.sh \
        drl-trading-preprocess \
        ${CI_COMMIT_SHORT_SHA} \
        e2e-${CI_COMMIT_SHORT_SHA}
    - uv run pytest tests/e2e/ --env-url="${SERVICE_URL}"
  after_script:
    - ./infrastructure/cloud/aws/deployment/cleanup-ephemeral-env.sh \
        drl-trading-preprocess \
        e2e-${CI_COMMIT_SHORT_SHA}
```

## Manual Usage

### Deploy Environment

```bash
export AWS_ACCOUNT_ID="123456789012"
export AWS_REGION="us-east-1"
export VPC_ID="vpc-abc123"

./deploy-ephemeral-env.sh \
  drl-trading-preprocess \
  abc123 \
  e2e-test-1
```

### Run Tests

```bash
SERVICE_URL=$(cd terraform/ephemeral-ecs && terraform output -raw service_url)
pytest tests/e2e/ --env-url="${SERVICE_URL}"
```

### Cleanup

```bash
./cleanup-ephemeral-env.sh drl-trading-preprocess e2e-test-1
```

## Prerequisites

### AWS Infrastructure
- **ECS Cluster:** `drl-trading-cluster` (must exist)
- **VPC:** Public and private subnets with NAT gateway
- **ECR Repository:** Service images already pushed
- **S3 Bucket:** `drl-trading-terraform-state` (for state storage)
- **DynamoDB Table:** `drl-trading-terraform-locks` (for state locking)

### Required Environment Variables
- `AWS_ACCOUNT_ID` - AWS account number
- `AWS_REGION` - AWS region (default: us-east-1)
- `VPC_ID` - VPC identifier (auto-discovers default if not set)
- `AWS_ACCESS_KEY_ID` - AWS credentials (via IAM role or keys)
- `AWS_SECRET_ACCESS_KEY` - AWS credentials

### IAM Permissions Required
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:*",
        "ec2:*",
        "elasticloadbalancing:*",
        "iam:*Role*",
        "iam:*Policy*",
        "logs:*",
        "ecr:*",
        "s3:*",
        "dynamodb:*"
      ],
      "Resource": "*"
    }
  ]
}
```

## Cost Management

**Cost Optimization Features:**
- Minimal Fargate task size (512 CPU / 1024 MB)
- 1-day log retention
- Automatic cleanup after 2 hours
- Auto-stop environment in GitLab

**Estimated Cost per E2E Run:**
- ECS Task: ~$0.05/hour
- ALB: ~$0.03/hour
- CloudWatch Logs: ~$0.01
- **Total: ~$0.10-0.15 per run** (assuming 15-30 minute test duration)

## Troubleshooting

### Service won't become healthy

Check CloudWatch logs:
```bash
ENV_ID="e2e-abc123"
SERVICE_NAME="drl-trading-preprocess"
aws logs tail "/ecs/ephemeral/${SERVICE_NAME}/${ENV_ID}" --follow
```

Check ECS service events:
```bash
aws ecs describe-services \
  --cluster drl-trading-cluster \
  --services "${SERVICE_NAME}-e2e-${ENV_ID}"
```

### Cleanup fails

Run manual orphaned resource cleanup:
```bash
# The cleanup script automatically attempts this
./cleanup-ephemeral-env.sh drl-trading-preprocess e2e-abc123
```

### State locked

Clear DynamoDB lock (use with caution):
```bash
aws dynamodb delete-item \
  --table-name drl-trading-terraform-locks \
  --key "{\"LockID\":{\"S\":\"drl-trading-terraform-state/ephemeral-ecs/e2e-abc123/terraform.tfstate\"}}"
```

## Security Considerations

1. **Resource Tagging:** All resources tagged with `EnvId`, `AutoCleanup`, `CreatedBy` for tracking
2. **IAM Isolation:** Each environment gets unique IAM roles
3. **Network Isolation:** Security groups restrict access to ALB only
4. **Credential Protection:** Protected variables in GitLab prevent MR access
5. **Short-lived:** Max 2-hour lifetime reduces attack surface

## Future Enhancements

- [ ] Support for multiple services (microservices)
- [ ] Custom VPC/subnet selection per service
- [ ] HTTPS support with ACM certificates
- [ ] Database provisioning (RDS snapshot restore)
- [ ] Cost tracking and alerts
- [ ] Terraform workspace isolation
- [ ] Parallel E2E environment support

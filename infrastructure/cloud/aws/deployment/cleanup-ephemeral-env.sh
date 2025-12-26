#!/bin/bash
# Cleanup ephemeral ECS environment after E2E testing
# Destroys all Terraform-managed resources

set -euo pipefail

# Script arguments
SERVICE_NAME="${1:?Usage: $0 <service-name> <env-id>}"
ENV_ID="${2:?Usage: $0 <service-name> <env-id>}"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/terraform/ephemeral-ecs"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "==========================================="
echo "Cleaning Up Ephemeral ECS Environment"
echo "==========================================="
echo "Service:    ${SERVICE_NAME}"
echo "Env ID:     ${ENV_ID}"
echo "Region:     ${AWS_REGION}"
echo "==========================================="

cd "${TERRAFORM_DIR}"

# Check if terraform state exists
if [ ! -f "terraform.tfstate" ] && [ ! -f "backend.tf" ]; then
  echo "INFO: No terraform state found locally, checking remote backend..."

  # Recreate backend configuration
  cat > backend.tf <<EOF
terraform {
  backend "s3" {
    bucket         = "drl-trading-terraform-state"
    key            = "ephemeral-ecs/${ENV_ID}/terraform.tfstate"
    region         = "${AWS_REGION}"
    encrypt        = true
    dynamodb_table = "drl-trading-terraform-locks"
  }
}
EOF

  terraform init -reconfigure || {
    echo "WARNING: Failed to initialize terraform with remote state"
    echo "State may have already been destroyed or never created"
    exit 0
  }
fi

# Check if state is empty
if ! terraform state list &>/dev/null; then
  echo "INFO: No resources found in terraform state"
  echo "Environment may have already been cleaned up"

  # Try to clean up any orphaned resources by tags
  echo ""
  echo "Checking for orphaned AWS resources..."
  cleanup_orphaned_resources

  exit 0
fi

# Destroy infrastructure
echo ""
echo "Destroying Terraform-managed resources..."
terraform destroy -auto-approve || {
  echo "ERROR: Terraform destroy failed"
  echo "Attempting manual cleanup of resources..."
  cleanup_orphaned_resources
  exit 1
}

# Clean up terraform state from S3
echo ""
echo "Cleaning up remote terraform state..."
aws s3 rm "s3://drl-trading-terraform-state/ephemeral-ecs/${ENV_ID}/terraform.tfstate" \
  --region "${AWS_REGION}" 2>/dev/null || echo "INFO: Remote state already removed"

# Clean up local files
echo ""
echo "Cleaning up local terraform files..."
rm -f terraform.tfvars
rm -f tfplan
rm -f backend.tf
rm -rf .terraform/

echo ""
echo "==========================================="
echo "✓ Cleanup Complete"
echo "==========================================="
echo "Environment ID: ${ENV_ID}"
echo "All resources have been destroyed"
echo "==========================================="

exit 0

# Function to clean up orphaned resources by tags
cleanup_orphaned_resources() {
  echo "Searching for resources tagged with EnvId=${ENV_ID}..."

  # Clean up ECS services
  echo ""
  echo "Checking for ECS services..."
  ECS_SERVICES=$(aws ecs list-services \
    --cluster drl-trading-cluster \
    --region "${AWS_REGION}" \
    --query "serviceArns[?contains(@, '${ENV_ID}')]" \
    --output text 2>/dev/null || echo "")

  if [ -n "${ECS_SERVICES}" ]; then
    for SERVICE_ARN in ${ECS_SERVICES}; do
      echo "Deleting ECS service: ${SERVICE_ARN}"
      aws ecs update-service \
        --cluster drl-trading-cluster \
        --service "${SERVICE_ARN}" \
        --desired-count 0 \
        --region "${AWS_REGION}" 2>/dev/null || true

      aws ecs delete-service \
        --cluster drl-trading-cluster \
        --service "${SERVICE_ARN}" \
        --force \
        --region "${AWS_REGION}" 2>/dev/null || true
    done
  fi

  # Clean up load balancers
  echo ""
  echo "Checking for load balancers..."
  ALB_ARNS=$(aws elbv2 describe-load-balancers \
    --region "${AWS_REGION}" \
    --query "LoadBalancers[?contains(LoadBalancerName, '${ENV_ID}')].LoadBalancerArn" \
    --output text 2>/dev/null || echo "")

  if [ -n "${ALB_ARNS}" ]; then
    for ALB_ARN in ${ALB_ARNS}; do
      echo "Deleting load balancer: ${ALB_ARN}"
      aws elbv2 delete-load-balancer \
        --load-balancer-arn "${ALB_ARN}" \
        --region "${AWS_REGION}" 2>/dev/null || true
    done
  fi

  # Clean up target groups
  echo ""
  echo "Checking for target groups..."
  TG_ARNS=$(aws elbv2 describe-target-groups \
    --region "${AWS_REGION}" \
    --query "TargetGroups[?contains(TargetGroupName, '${ENV_ID}')].TargetGroupArn" \
    --output text 2>/dev/null || echo "")

  if [ -n "${TG_ARNS}" ]; then
    sleep 10  # Wait for ALB deletion
    for TG_ARN in ${TG_ARNS}; do
      echo "Deleting target group: ${TG_ARN}"
      aws elbv2 delete-target-group \
        --target-group-arn "${TG_ARN}" \
        --region "${AWS_REGION}" 2>/dev/null || true
    done
  fi

  # Clean up security groups
  echo ""
  echo "Checking for security groups..."
  SG_IDS=$(aws ec2 describe-security-groups \
    --region "${AWS_REGION}" \
    --filters "Name=tag:EnvId,Values=${ENV_ID}" \
    --query "SecurityGroups[].GroupId" \
    --output text 2>/dev/null || echo "")

  if [ -n "${SG_IDS}" ]; then
    sleep 30  # Wait for dependent resources
    for SG_ID in ${SG_IDS}; do
      echo "Deleting security group: ${SG_ID}"
      aws ec2 delete-security-group \
        --group-id "${SG_ID}" \
        --region "${AWS_REGION}" 2>/dev/null || true
    done
  fi

  # Clean up CloudWatch log groups
  echo ""
  echo "Checking for CloudWatch log groups..."
  LOG_GROUPS=$(aws logs describe-log-groups \
    --region "${AWS_REGION}" \
    --log-group-name-prefix "/ecs/ephemeral/${SERVICE_NAME}/${ENV_ID}" \
    --query "logGroups[].logGroupName" \
    --output text 2>/dev/null || echo "")

  if [ -n "${LOG_GROUPS}" ]; then
    for LOG_GROUP in ${LOG_GROUPS}; do
      echo "Deleting log group: ${LOG_GROUP}"
      aws logs delete-log-group \
        --log-group-name "${LOG_GROUP}" \
        --region "${AWS_REGION}" 2>/dev/null || true
    done
  fi

  echo ""
  echo "Orphaned resource cleanup completed"
}

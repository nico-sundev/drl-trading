#!/bin/bash
# Deploy ephemeral ECS environment for E2E testing
# Uses Terraform to create temporary infrastructure

set -euo pipefail

# Script arguments
SERVICE_NAME="${1:?Usage: $0 <service-name> <image-tag> <env-id>}"
IMAGE_TAG="${2:?Usage: $0 <service-name> <image-tag> <env-id>}"
ENV_ID="${3:?Usage: $0 <service-name> <image-tag> <env-id>}"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/terraform/ephemeral-ecs"
AWS_REGION="${AWS_REGION:-us-east-1}"
VPC_ID="${VPC_ID:-}"
MAX_WAIT_TIME=300  # 5 minutes

echo "==========================================="
echo "Deploying Ephemeral ECS Environment"
echo "==========================================="
echo "Service:    ${SERVICE_NAME}"
echo "Image Tag:  ${IMAGE_TAG}"
echo "Env ID:     ${ENV_ID}"
echo "Region:     ${AWS_REGION}"
echo "==========================================="

# Validate required environment variables
if [ -z "${AWS_ACCOUNT_ID:-}" ]; then
  echo "ERROR: AWS_ACCOUNT_ID environment variable is not set"
  exit 1
fi

if [ -z "${VPC_ID}" ]; then
  echo "INFO: VPC_ID not set, attempting to discover default VPC"
  VPC_ID=$(aws ec2 describe-vpcs \
    --region "${AWS_REGION}" \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" \
    --output text 2>/dev/null || echo "")

  if [ -z "${VPC_ID}" ] || [ "${VPC_ID}" == "None" ]; then
    echo "ERROR: Could not find default VPC. Please set VPC_ID environment variable"
    exit 1
  fi

  echo "INFO: Using default VPC: ${VPC_ID}"
fi

# Initialize Terraform
echo ""
echo "Initializing Terraform..."
cd "${TERRAFORM_DIR}"

# Use dynamic S3 backend key based on env_id
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

terraform init -upgrade

# Create terraform variables file
echo ""
echo "Creating Terraform variables..."
cat > terraform.tfvars <<EOF
service_name     = "${SERVICE_NAME}"
env_id           = "${ENV_ID}"
git_commit_sha   = "${IMAGE_TAG}"
image_tag        = "${IMAGE_TAG}"
aws_region       = "${AWS_REGION}"
vpc_id           = "${VPC_ID}"
ecs_cluster_name = "drl-trading-cluster"
container_port   = 8080
health_check_path = "/health"
task_cpu         = "512"
task_memory      = "1024"
EOF

# Plan and apply
echo ""
echo "Planning Terraform deployment..."
terraform plan -out=tfplan

echo ""
echo "Applying Terraform deployment..."
terraform apply -auto-approve tfplan

# Extract outputs
echo ""
echo "Extracting deployment information..."
SERVICE_URL=$(terraform output -raw service_url)
ALB_DNS=$(terraform output -raw alb_dns_name)
LOG_GROUP=$(terraform output -raw log_group_name)
ECS_SERVICE=$(terraform output -raw ecs_service_name)
ECS_CLUSTER=$(terraform output -raw ecs_cluster_name)

echo ""
echo "==========================================="
echo "Deployment Complete"
echo "==========================================="
echo "Service URL:  ${SERVICE_URL}"
echo "ALB DNS:      ${ALB_DNS}"
echo "ECS Service:  ${ECS_SERVICE}"
echo "ECS Cluster:  ${ECS_CLUSTER}"
echo "Log Group:    ${LOG_GROUP}"
echo "==========================================="

# Wait for service to become healthy
echo ""
echo "Waiting for service to become healthy..."
START_TIME=$(date +%s)

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  if [ ${ELAPSED} -ge ${MAX_WAIT_TIME} ]; then
    echo "ERROR: Timeout waiting for service to become healthy (${MAX_WAIT_TIME}s)"
    echo "Check CloudWatch logs: ${LOG_GROUP}"
    exit 1
  fi

  # Check ECS service status
  RUNNING_COUNT=$(aws ecs describe-services \
    --cluster "${ECS_CLUSTER}" \
    --services "${ECS_SERVICE}" \
    --region "${AWS_REGION}" \
    --query "services[0].runningCount" \
    --output text 2>/dev/null || echo "0")

  # Check target health
  TARGET_GROUP_ARN=$(aws elbv2 describe-target-groups \
    --region "${AWS_REGION}" \
    --names "${SERVICE_NAME}-e2e-${ENV_ID}" \
    --query "TargetGroups[0].TargetGroupArn" \
    --output text 2>/dev/null || echo "")

  if [ -n "${TARGET_GROUP_ARN}" ] && [ "${TARGET_GROUP_ARN}" != "None" ]; then
    HEALTHY_COUNT=$(aws elbv2 describe-target-health \
      --target-group-arn "${TARGET_GROUP_ARN}" \
      --region "${AWS_REGION}" \
      --query "length(TargetHealthDescriptions[?TargetHealth.State=='healthy'])" \
      --output text 2>/dev/null || echo "0")
  else
    HEALTHY_COUNT=0
  fi

  echo "[${ELAPSED}s] Running tasks: ${RUNNING_COUNT}, Healthy targets: ${HEALTHY_COUNT}"

  if [ "${RUNNING_COUNT}" -ge 1 ] && [ "${HEALTHY_COUNT}" -ge 1 ]; then
    echo "✓ Service is healthy!"
    break
  fi

  sleep 10
done

# Verify service is responding
echo ""
echo "Verifying service endpoint..."
MAX_RETRIES=5
RETRY_COUNT=0

while [ ${RETRY_COUNT} -lt ${MAX_RETRIES} ]; do
  if curl -f -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" | grep -q "200"; then
    echo "✓ Service is responding to health checks"
    break
  fi

  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "Attempt ${RETRY_COUNT}/${MAX_RETRIES} failed, retrying..."
  sleep 5
done

if [ ${RETRY_COUNT} -ge ${MAX_RETRIES} ]; then
  echo "WARNING: Service health check failed after ${MAX_RETRIES} attempts"
  echo "Continuing anyway - E2E tests may fail"
fi

echo ""
echo "==========================================="
echo "✓ Ephemeral environment ready for testing"
echo "==========================================="
echo "Environment ID: ${ENV_ID}"
echo "Service URL:    ${SERVICE_URL}"
echo ""
echo "To view logs:"
echo "  aws logs tail ${LOG_GROUP} --follow --region ${AWS_REGION}"
echo ""
echo "To cleanup:"
echo "  ./cleanup-ephemeral-env.sh ${SERVICE_NAME} ${ENV_ID}"
echo "==========================================="

exit 0

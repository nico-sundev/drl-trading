#!/bin/bash

# Script to create ECR repositories for all DRL Trading microservices
# Run this after setting up the IAM user with create-ci-ecr-push-user.sh

set -e  # Exit on error

# Get AWS region from configuration
REGION=$(aws configure get region)
echo "Creating ECR repositories in region: $REGION"

# Array of all microservices that need ECR repositories
SERVICES=(
    "drl-trading-preprocess"
    "drl-trading-training"
    "drl-trading-ingest"
    "drl-trading-inference"
    "drl-trading-execution"
)

# Create repository for each service
for SERVICE in "${SERVICES[@]}"; do
    echo "Creating repository: $SERVICE"
    aws ecr create-repository \
        --repository-name "$SERVICE" \
        --region "$REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 \
        2>/dev/null || echo "Repository $SERVICE already exists, skipping..."
done

echo ""
echo "ECR repositories created successfully!"
echo ""
echo "Repository URIs:"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
for SERVICE in "${SERVICES[@]}"; do
    echo "  ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${SERVICE}"
done

echo ""
echo "Next steps:"
echo "1. Push Docker images from GitLab CI/CD pipeline"
echo "2. Images will be automatically scanned for vulnerabilities on push"

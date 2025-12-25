#!/bin/bash
set -e

# Pull CI image from AWS ECR for local testing
# Usage: ./infrastructure/cloud/docker/pull-ci-image.sh [tag]
#
# Environment variables:
#   AWS_REGION: AWS region (default: us-east-1)
#   AWS_ACCOUNT_ID: Your AWS account ID (required)
#   ECR_REPOSITORY: ECR repository name (default: drl-trading-ci)

TAG="${1:-latest}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-drl-trading-ci}"

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "❌ ERROR: AWS_ACCOUNT_ID environment variable is required"
    exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:${TAG}"

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "⬇️  Pulling image..."
docker pull "$IMAGE_URI"

echo "✅ Image pulled successfully!"
echo ""
echo "🧪 To test the image:"
echo "   docker run --rm $IMAGE_URI"
echo ""
echo "🚀 To run tests with this image:"
echo "   docker run --rm -v \$(pwd):/workspace $IMAGE_URI run pytest tests/"

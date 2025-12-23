#!/bin/bash
set -e

# Build and push CI image to AWS ECR
# Usage: ./scripts/docker/build-and-push-ci-image.sh [tag]
#
# Environment variables:
#   AWS_REGION: AWS region (default: us-east-1)
#   AWS_ACCOUNT_ID: Your AWS account ID (required)
#   ECR_REPOSITORY: ECR repository name (default: drl-trading-ci)

# Configuration
TAG="${1:-latest}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-drl-trading-ci}"
DOCKER_BUILD_CONTEXT="$(dirname "$0")/../../.docker/ci"

# Validate AWS_ACCOUNT_ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "❌ ERROR: AWS_ACCOUNT_ID environment variable is required"
    echo "   Example: export AWS_ACCOUNT_ID=123456789012"
    exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:${TAG}"

echo "🔧 Configuration:"
echo "   AWS Region:      $AWS_REGION"
echo "   AWS Account:     $AWS_ACCOUNT_ID"
echo "   ECR Repository:  $ECR_REPOSITORY"
echo "   Image Tag:       $TAG"
echo "   Full Image URI:  $IMAGE_URI"
echo ""

# Step 1: Ensure ECR repository exists
echo "📦 Ensuring ECR repository exists..."
if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" &>/dev/null; then
    echo "   Creating repository: $ECR_REPOSITORY"
    aws ecr create-repository \
        --repository-name "$ECR_REPOSITORY" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo "   ✅ Repository created"
else
    echo "   ✅ Repository already exists"
fi

# Step 2: Login to ECR
echo ""
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REGISTRY"
echo "   ✅ Login successful"

# Step 3: Build the image
echo ""
echo "🏗️  Building Docker image..."
docker build \
    -t "$ECR_REPOSITORY:$TAG" \
    -t "$IMAGE_URI" \
    "$DOCKER_BUILD_CONTEXT"
echo "   ✅ Build successful"

# Step 4: Push to ECR
echo ""
echo "⬆️  Pushing image to ECR..."
docker push "$IMAGE_URI"
echo "   ✅ Push successful"

# Step 5: Tag as latest if not already
if [ "$TAG" != "latest" ]; then
    LATEST_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:latest"
    echo ""
    echo "🏷️  Tagging as latest..."
    docker tag "$IMAGE_URI" "$LATEST_URI"
    docker push "$LATEST_URI"
    echo "   ✅ Latest tag pushed"
fi

echo ""
echo "✨ Success! Image available at:"
echo "   $IMAGE_URI"
echo ""
echo "📝 To use in GitLab CI, set this variable:"
echo "   CI_IMAGE: $IMAGE_URI"
echo ""
echo "🧪 To test locally:"
echo "   docker run --rm $IMAGE_URI"

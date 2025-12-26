#!/bin/bash
# Quick CI Image Operations - Copy-paste ready commands

echo "═══════════════════════════════════════════════════════"
echo "  CI Image Quick Reference"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check if AWS_ACCOUNT_ID is set
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "⚠️  AWS_ACCOUNT_ID not set!"
    echo "   Run: export AWS_ACCOUNT_ID=123456789012"
    echo ""
fi

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-drl-trading-ci}"
IMAGE_TAG="${CI_IMAGE_TAG:-latest}"

if [ -n "$AWS_ACCOUNT_ID" ]; then
    IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

    echo "Current Configuration:"
    echo "  AWS Account:  $AWS_ACCOUNT_ID"
    echo "  AWS Region:   $AWS_REGION"
    echo "  Repository:   $ECR_REPOSITORY"
    echo "  Image Tag:    $IMAGE_TAG"
    echo "  Full URI:     $IMAGE_URI"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# 1. Configure environment"
echo "cp .env.example .env && nano .env && source .env"
echo ""
echo "# 2. Verify AWS access"
echo "aws sts get-caller-identity"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏗️  Build & Push"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# Build and push (recommended)"
echo "make -f Makefile.ci push-ci-image"
echo ""
echo "# Build with version tag"
echo "IMAGE_TAG=v1.0.0 make -f Makefile.ci push-ci-image"
echo ""
echo "# Using script directly"
echo "./infrastructure/cloud/docker/build-and-push-ci-image.sh"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⬇️  Pull & Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# Pull from ECR"
echo "make -f Makefile.ci pull-ci-image"
echo ""
echo "# Test image"
echo "make -f Makefile.ci test-ci-image"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Local Testing"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -n "$AWS_ACCOUNT_ID" ]; then
    echo "# Check UV version"
    echo "docker run --rm $IMAGE_URI --version"
    echo ""
    echo "# Check Docker version"
    echo "docker run --rm $IMAGE_URI run docker --version"
    echo ""
    echo "# Run tests in current directory"
    echo "docker run --rm -v \$(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock $IMAGE_URI run pytest tests/"
    echo ""
    echo "# Interactive shell"
    echo "make -f Makefile.ci shell-ci-image"
    echo ""
else
    echo "# (Set AWS_ACCOUNT_ID to see commands)"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Maintenance"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# Show configuration"
echo "make -f Makefile.ci ci-image-info"
echo ""
echo "# Clean local images"
echo "make -f Makefile.ci clean-ci-images"
echo ""
echo "# ECR login (if needed)"
echo "aws ecr get-login-password --region \$AWS_REGION | docker login --username AWS --password-stdin \$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_REGION.amazonaws.com"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 Documentation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Full guide: docs/CI_IMAGE_SETUP.md"
echo "  Dockerfile: .docker/ci/Dockerfile"
echo "  README:     .docker/ci/README.md"
echo ""
echo "═══════════════════════════════════════════════════════"

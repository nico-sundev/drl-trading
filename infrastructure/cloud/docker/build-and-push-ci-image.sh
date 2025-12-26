#!/bin/bash
set -e

# Build and push CI image to GitLab Container Registry
# Usage: ./scripts/docker/build-and-push-ci-image.sh [tag]
#
# Environment variables:
#   GITLAB_REGISTRY: GitLab registry URL (default: registry.gitlab.com)
#   GITLAB_PROJECT_PATH: Your GitLab project path (e.g., username/ai-trading)
#   GITLAB_TOKEN: Personal access token or CI token for authentication
#   IMAGE_NAME: Image name (default: drl-trading-ci)

# Configuration
TAG="${1:-latest}"
GITLAB_REGISTRY="${GITLAB_REGISTRY:-registry.gitlab.com}"
IMAGE_NAME="${IMAGE_NAME:-drl-trading-ci}"
DOCKER_BUILD_CONTEXT="$(dirname "$0")/ci"

# Validate required variables
if [ -z "$GITLAB_PROJECT_PATH" ]; then
    echo "❌ ERROR: GITLAB_PROJECT_PATH environment variable is required"
    echo "   Example: export GITLAB_PROJECT_PATH=yourusername/ai-trading"
    exit 1
fi

if [ -z "$GITLAB_TOKEN" ]; then
    echo "❌ ERROR: GITLAB_TOKEN environment variable is required"
    echo "   Create a personal access token with 'write_registry' scope at:"
    echo "   https://gitlab.com/-/profile/personal_access_tokens"
    echo "   Then: export GITLAB_TOKEN=glpat-xxxxxxxxxxxxx"
    exit 1
fi

IMAGE_URI="${GITLAB_REGISTRY}/${GITLAB_PROJECT_PATH}/${IMAGE_NAME}:${TAG}"

echo "🔧 Configuration:"
echo "   GitLab Registry:  $GITLAB_REGISTRY"
echo "   Project Path:     $GITLAB_PROJECT_PATH"
echo "   Image Name:       $IMAGE_NAME"
echo "   Image Tag:        $TAG"
echo "   Full Image URI:   $IMAGE_URI"
echo ""

# Step 1: Login to GitLab Container Registry
echo "🔐 Logging into GitLab Container Registry..."
echo "$GITLAB_TOKEN" | docker login "$GITLAB_REGISTRY" --username gitlab-ci-token --password-stdin
echo "   ✅ Login successful"

# Step 2: Build the image
echo ""
echo "🏗️  Building Docker image..."
docker build \
    -t "$IMAGE_NAME:$TAG" \
    -t "$IMAGE_URI" \
    "$DOCKER_BUILD_CONTEXT"
echo "   ✅ Build successful"

# Step 3: Push to GitLab Container Registry
echo ""
echo "⬆️  Pushing image to GitLab Container Registry..."
docker push "$IMAGE_URI"
echo "   ✅ Push successful"

# Step 4: Tag as latest if not already
if [ "$TAG" != "latest" ]; then
    LATEST_URI="${GITLAB_REGISTRY}/${GITLAB_PROJECT_PATH}/${IMAGE_NAME}:latest"
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
echo "📝 To use in GitLab CI, update your .gitlab-ci.yml:"
echo "   variables:"
echo "     CI_IMAGE: $GITLAB_REGISTRY/\$CI_PROJECT_PATH/$IMAGE_NAME:latest"
echo ""
echo "🧪 To test locally:"
echo "   docker run --rm $IMAGE_URI"

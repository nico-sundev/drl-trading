# CI/CD Docker Image Setup

Custom Docker image combining UV package manager with Docker CLI for running integration tests in CI/CD pipelines.

## Quick Start

**⚠️ Important:** The CI image must be built and pushed manually by developers. It is not automated in the GitLab pipeline to keep build concerns separate.

```bash
# 1. Configure AWS
cp .env.example .env
nano .env  # Add your AWS_ACCOUNT_ID
source .env

# 2. Build and push to ECR (MUST be done manually)
make -f Makefile.ci push-ci-image

# 3. Configure GitLab CI/CD variables (see below)

# 4. Done! Pipeline now runs integration tests
```

## Prerequisites

- Docker Desktop installed and running
- Fresh AWS account (or existing with ECR access)
- GitLab repository with CI/CD access

## AWS Setup (Fresh Account)

### 1. Get Your AWS Account ID

```bash
# Install AWS CLI (if not installed)
# Windows: https://awscli.amazonaws.com/AWSCLIV2.msi
# Mac: brew install awscli
# Linux: pip install awscli

# Configure AWS CLI
aws configure
# AWS Access Key ID: (paste from AWS console)
# AWS Secret Access Key: (paste from AWS console)
# Default region: us-east-1
# Default output format: json

# Verify setup
aws sts get-caller-identity
# Note the "Account" number - this is your AWS_ACCOUNT_ID
```

### 2. Create IAM User for CI/CD (Recommended)

**In AWS Console:**

1. Go to IAM → Users → Create user
2. Name: `gitlab-ci-ecr`
3. Attach policy: `AmazonEC2ContainerRegistryPowerUser`
4. Create access keys → Note down:
   - Access Key ID
   - Secret Access Key

**Using AWS CLI:**

```bash
# Create user
aws iam create-user --user-name gitlab-ci-ecr

# Attach ECR policy
aws iam attach-user-policy \
  --user-name gitlab-ci-ecr \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

# Create access keys
aws iam create-access-key --user-name gitlab-ci-ecr
# Save the AccessKeyId and SecretAccessKey
```

# Create .env from template

cp .env.example .env

# Edit .env with your values

nano .env  # or use your preferred editor

# Load environment

source .env

```

### Step 2: Verify AWS Access

```bash
# Check AWS CLI is configured
aws sts get-caller-identity

# Should show your account ID and user
```

### Step 3: ECR Repository (Auto-created)

The scripts automatically create the ECR repository if it doesn't exist. To create manually:

```bash
aws ecr create-repository \
  --repository-name drl-trading-ci \
  --region us-east-1 \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256
```

## Local Setup

### 1. Configure Environment

```bash
# Navigate to project root
cd /path/to/ai_trading

# Create and edit environment file
cp .env.example .env
nano .env
```

**Edit `.env` with your AWS details:**

```bash
export AWS_ACCOUNT_ID=123456789012  # From step above
export AWS_REGION=us-east-1
export ECR_REPOSITORY=drl-trading-ci
```

```bash
# Load environment
source .env
```

### 2. Build and Push to ECR

```bash
# This will:
# - Create ECR repository automatically
# - Build the Docker image
# - Push to your AWS ECR
make -f Makefile.ci push-ci-image
```

**Expected output:**

```
📦 Ensuring ECR repository exists...
   ✅ Repository created
🔐 Logging into ECR...
   ✅ Login successful
🏗️  Building Docker image...
   ✅ Build successful
⬆️  Pushing image to ECR...
   ✅ Push successful
✨ Success! Image available at:
   123456789012.dkr.ecr.us-east-1.amazonaws.com/drl-trading-ci:latest
```

### 3. Test Locally (Optional)

```bash
# Verify image works
make -f Makefile.ci test-ci-image

# Run tests with the image
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/drl-trading-ci:latest \
  run pytest tests/ -v
```

## GitLab CI Configuration

### Configure CI/CD Variables

Go to: **GitLab → Your Project → Settings → CI/CD → Variables**

Add these 4 variables:

| Variable | Value | Protected | Masked |
|----------|-------|-----------|--------|
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID | ✅ Yes | ✅ Yes |
| `AWS_REGION` | `us-east-1` | ❌ No | ❌ No |
| `AWS_ACCESS_KEY_ID` | From IAM user creation above | ✅ Yes | ✅ Yes |
| `AWS_SECRET_ACCESS_KEY` | From IAM user creation above | ✅ Yes | ✅ Yes |

### Verify Pipeline

```bash
# Commit and push
git add .
git commit -m "Add CI image configuration"
git push origin develop

# Watch pipeline in GitLab UI
```

**Expected behavior:**

- ✅ All test jobs run with custom CI image
- ✅ Integration tests execute (not skipped)
- ✅ Docker-in-Docker works
- ✅ Pipeline passes

## Common Issues

### "AWS_ACCOUNT_ID not set"

```bash
source .env  # Reload environment
```

### "Cannot connect to Docker daemon"

```bash
# Ensure Docker Desktop is running
docker info
```

### "No basic auth credentials"

```bash
# Re-authenticate with ECR
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

### Integration tests still skip

Check GitLab pipeline logs for:

```
Pulling docker image ...ecr.amazonaws.com/drl-trading-ci:latest
Service container docker:27-dind running
```

## Maintenance

### Update the Image

When you need to modify the image:

1. Edit `.docker/ci/Dockerfile`
2. **Manually rebuild and push**: `make -f Makefile.ci push-ci-image`
3. Verify in CI: Watch next pipeline run

### Recommended: Periodic Rebuilds

Periodically rebuild the image to get latest base image security patches:

```bash
# Pull latest base image and rebuild
make -f Makefile.ci push-ci-image
```

Consider doing this:

- Monthly for security patches
- When UV or Docker releases major updates
- Before important releases

## Useful Commands

```bash
# Show current configuration
make -f Makefile.ci ci-image-info

# Pull image from ECR
make -f Makefile.ci pull-ci-image

# Interactive shell in CI image
make -f Makefile.ci shell-ci-image

# Clean up local images
make -f Makefile.ci clean-ci-images

# Quick reference
bash scripts/docker/ci-image-quickref.sh
```

#!/bin/bash

# Script to create IAM user for GitLab CI/CD with restrictive ECR push permissions
# This follows least-privilege principle by only allowing ECR operations

set -e  # Exit on error

echo "Creating IAM group for CI/CD..."
# 1. Create IAM group
aws iam create-group --group-name ci-ecr-push

echo "Creating restrictive ECR policy..."
# 2. Create custom policy with only ECR push permissions (not full access)
aws iam create-policy \
  --policy-name ECRPushOnly \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "ecr:GetAuthorizationToken"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeRepositories",
          "ecr:ListImages"
        ],
        "Resource": "arn:aws:ecr:*:*:repository/drl-trading-*"
      }
    ]
  }'

echo "Getting AWS account ID..."
# 3. Get account ID for policy ARN
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account ID: $ACCOUNT_ID"

echo "Attaching policy to group..."
# 4. Attach the custom policy to the group
aws iam attach-group-policy \
  --group-name ci-ecr-push \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ECRPushOnly

echo "Creating IAM user..."
# 5. Create IAM user
aws iam create-user --user-name gitlab-ci-drl-trading

echo "Adding user to group..."
# 6. Add user to group
aws iam add-user-to-group \
  --user-name gitlab-ci-drl-trading \
  --group-name ci-ecr-push

echo "Creating access keys..."
# 7. Create access keys for this user
echo "IMPORTANT: Save these credentials immediately - they won't be shown again!"
aws iam create-access-key --user-name gitlab-ci-drl-trading

echo ""
echo "Setup complete! Next steps:"
echo "1. Save the AccessKeyId and SecretAccessKey from above"
echo "2. Add these to GitLab CI/CD variables (Settings → CI/CD → Variables):"
echo "   - AWS_ACCOUNT_ID = ${ACCOUNT_ID}"
echo "   - AWS_REGION = $(aws configure get region)"
echo "   - AWS_ACCESS_KEY_ID = <from output above>"
echo "   - AWS_SECRET_ACCESS_KEY = <from output above> (mark as masked)"
echo "3. Create ECR repositories using create-ecr-repositories.sh"

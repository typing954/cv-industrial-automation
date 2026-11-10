#!/usr/bin/env bash
# Build the inference image and push it to Amazon ECR.  Usage: build_push_ecr.sh [region] [model-weights] [repo]
#   e.g.  cloudstream/deploy/build_push_ecr.sh us-east-1 yolo11s.pt people-yolo-endpoint
# Prints the image URI to pass to sagemaker_deploy.py. Needs: aws cli (logged in), docker.
set -euo pipefail
REGION=${1:-us-east-1}; MODEL=${2:-yolo11s.pt}; REPO=${3:-people-yolo-endpoint}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
URI=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 || \
  aws ecr create-repository --region "$REGION" --repository-name "$REPO" >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
docker build --build-arg MODEL="$MODEL" -t "$URI" "$(dirname "$0")/../server"
docker push "$URI"
echo "IMAGE_URI=$URI"

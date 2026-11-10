#!/usr/bin/env bash
# Build the inference image and push it to Google Artifact Registry.  Usage: build_push_gar.sh <project> [region] [model-weights] [repo]
#   e.g.  cloudstream/deploy/build_push_gar.sh my-project us-central1 yolo11s.pt people-yolo
# Prints the image URI to pass to vertex_deploy.py. Needs: gcloud (logged in), docker.
set -euo pipefail
PROJECT=${1:?usage: build_push_gar.sh <project> [region] [model] [repo]}; REGION=${2:-us-central1}; MODEL=${3:-yolo11s.pt}; REPO=${4:-people-yolo}
URI=$REGION-docker.pkg.dev/$PROJECT/$REPO/people-yolo-endpoint:latest
gcloud artifacts repositories describe "$REPO" --project "$PROJECT" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --project "$PROJECT" --location "$REGION" --repository-format docker
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
docker build --build-arg MODEL="$MODEL" -t "$URI" "$(dirname "$0")/../server"
docker push "$URI"
echo "IMAGE_URI=$URI"

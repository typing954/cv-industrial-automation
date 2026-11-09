#!/usr/bin/env bash
# Spin up ONE GCP VM, benchmark, pull results, DELETE it. Costs money. Verify image family / quota first.
# Usage: cloud/gcp_trial.sh <machine-type> <gpu-type|none> <path/to/video.mp4> [benchmark args...]
#   e.g. n1-standard-8 nvidia-tesla-t4 | g2-standard-8 (L4, GPU bundled -> pass 'none') | c3-standard-8 none | n2-standard-8 none
set -euo pipefail
MT=$1; GPU=$2; VIDEO=$3; shift 3
ZONE=${GCP_ZONE:-us-central1-a}; NAME=pc-trial-$RANDOM
FAMILY=pytorch-latest-cpu; ACC=(); EXTRA=()
if [[ $GPU != none || $MT == g2-* ]]; then FAMILY=pytorch-latest-gpu; EXTRA=(--metadata=install-nvidia-driver=True --maintenance-policy=TERMINATE); fi
[[ $GPU != none ]] && ACC=(--accelerator=type=$GPU,count=1)
gcloud compute instances create $NAME --zone $ZONE --machine-type $MT "${ACC[@]}" "${EXTRA[@]}" \
  --image-family $FAMILY --image-project deeplearning-platform-release --boot-disk-size 100GB
trap 'gcloud compute instances delete '$NAME' --zone '$ZONE' --quiet' EXIT
until gcloud compute ssh $NAME --zone $ZONE --command true 2>/dev/null; do sleep 8; done
cd "$(dirname "$0")/.."
tar czf /tmp/pc.tgz --exclude=.venv --exclude=results --exclude=videos .
gcloud compute scp /tmp/pc.tgz "$VIDEO" $NAME:~ --zone $ZONE
[[ $GPU != none || $MT == g2-* ]] && sleep 60  # driver install on first boot
gcloud compute ssh $NAME --zone $ZONE --command "mkdir -p pc && tar xzf pc.tgz -C pc && cp $(basename "$VIDEO") pc/video.mp4 && cd pc && bash cloud/bootstrap.sh video.mp4 $*"
mkdir -p results && gcloud compute scp "$NAME:pc/results/*.json" results/ --zone $ZONE

#!/usr/bin/env bash
# Spin up ONE AWS instance, benchmark, pull results, TERMINATE. Costs money — read before running.
# Usage: cloud/aws_trial.sh <instance-type> <key-name> <path/to/video.mp4> [benchmark args...]
#   e.g. g4dn.xlarge (T4) | g5.xlarge (A10G) | g6.xlarge (L4) | c6i.2xlarge / c7i.2xlarge (CPU) | c7g.2xlarge (Graviton CPU)
set -euo pipefail
TYPE=$1; KEY=$2; VIDEO=$3; shift 3
REGION=${AWS_REGION:-us-east-1}; SG=${AWS_SG:?set AWS_SG to a security group allowing SSH from your IP}
if [[ $TYPE == g* ]]; then
  AMI=$(aws ssm get-parameter --region $REGION --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id --query Parameter.Value --output text)
else
  AMI=$(aws ssm get-parameter --region $REGION --name /aws/service/canonical/ubuntu/server/22.04/stable/current/$([[ $TYPE == *g.* ]] && echo arm64 || echo amd64)/hvm/ebs-gp2/ami-id --query Parameter.Value --output text)
fi
ID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type $TYPE --key-name $KEY --security-group-ids $SG \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' --query 'Instances[0].InstanceId' --output text)
trap 'echo terminating $ID; aws ec2 terminate-instances --region $REGION --instance-ids $ID >/dev/null' EXIT
aws ec2 wait instance-running --region $REGION --instance-ids $ID
IP=$(aws ec2 describe-instances --region $REGION --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
SSH="ssh -o StrictHostKeyChecking=accept-new -i ${AWS_KEY_FILE:-$HOME/.ssh/$KEY.pem} ubuntu@$IP"
until $SSH true 2>/dev/null; do sleep 5; done
cd "$(dirname "$0")/.."
tar czf - --exclude=.venv --exclude=results --exclude=videos . | $SSH 'mkdir -p pc && tar xzf - -C pc'
scp -i ${AWS_KEY_FILE:-$HOME/.ssh/$KEY.pem} "$VIDEO" ubuntu@$IP:pc/video.mp4
$SSH 'sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv libgl1 libglib2.0-0 >/dev/null; cd pc && bash cloud/bootstrap.sh video.mp4 '"$*"
mkdir -p results && scp -i ${AWS_KEY_FILE:-$HOME/.ssh/$KEY.pem} "ubuntu@$IP:pc/results/*.json" results/

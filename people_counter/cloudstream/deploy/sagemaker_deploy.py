"""Create / inspect / delete a SageMaker real-time endpoint for the inference image.

  python cloudstream/deploy/sagemaker_deploy.py deploy --name people-yolo --image <ECR uri> --role <execution role ARN> \
         [--instance-type ml.g4dn.xlarge] [--instance-count 1] [--region us-east-1]
  python cloudstream/deploy/sagemaker_deploy.py status --name people-yolo
  python cloudstream/deploy/sagemaker_deploy.py delete --name people-yolo

COST: a running endpoint is billed per instance-hour until you delete it. The role needs the AmazonSageMakerFullAccess
policy (or equivalent) and ECR pull access. Instance types: ml.g4dn.xlarge (T4), ml.g5.xlarge (A10G), ml.g6.xlarge (L4).
"""
import argparse
import sys
import time

import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["deploy", "status", "delete"])
    p.add_argument("--name", required=True)
    p.add_argument("--image")
    p.add_argument("--role")
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--region")
    p.add_argument("--env", action="append", default=[], help="container env, e.g. MODEL=yolo11m.pt (repeatable)")
    p.add_argument("--startup-timeout", type=int, default=900, help="seconds the container may take to become healthy")
    a = p.parse_args()
    sm = boto3.client("sagemaker", region_name=a.region)

    if a.cmd == "deploy":
        if not (a.image and a.role):
            sys.exit("--image and --role are required for deploy")
        env = dict(e.split("=", 1) for e in a.env)
        cfg = f"{a.name}-config"
        sm.create_model(ModelName=a.name, ExecutionRoleArn=a.role, PrimaryContainer={"Image": a.image, "Environment": env})
        sm.create_endpoint_config(EndpointConfigName=cfg, ProductionVariants=[{
            "VariantName": "AllTraffic", "ModelName": a.name, "InstanceType": a.instance_type,
            "InitialInstanceCount": a.instance_count, "ContainerStartupHealthCheckTimeoutInSeconds": a.startup_timeout}])
        sm.create_endpoint(EndpointName=a.name, EndpointConfigName=cfg)
        print(f"creating endpoint '{a.name}' on {a.instance_count}× {a.instance_type} (takes ~5-10 min; billed until deleted)…")
        sm.get_waiter("endpoint_in_service").wait(EndpointName=a.name, WaiterConfig={"Delay": 30, "MaxAttempts": 60})
        print("InService. Stream to it with:\n  python -m cloudstream stream videos/<file> --endpoint-type sagemaker "
              f"--endpoint-name {a.name}" + (f" --region {a.region}" if a.region else ""))
    elif a.cmd == "status":
        d = sm.describe_endpoint(EndpointName=a.name)
        print(d["EndpointStatus"], d.get("FailureReason", ""))
    else:
        for fn, kw in ((sm.delete_endpoint, {"EndpointName": a.name}), (sm.delete_endpoint_config, {"EndpointConfigName": f"{a.name}-config"}),
                       (sm.delete_model, {"ModelName": a.name})):
            try:
                fn(**kw)
                print("deleted", list(kw.values())[0])
            except Exception as e:  # noqa: BLE001
                print("skip:", e)
            time.sleep(1)


if __name__ == "__main__":
    main()

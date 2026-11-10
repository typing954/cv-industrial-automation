"""Create / delete a Vertex AI endpoint for the inference image.

  python cloudstream/deploy/vertex_deploy.py deploy --project P --image <Artifact Registry uri> --name people-yolo \
         [--location us-central1] [--machine-type n1-standard-8] [--accelerator NVIDIA_TESLA_T4] [--accelerator-count 1]
  python cloudstream/deploy/vertex_deploy.py delete --project P --endpoint-id 123456 [--location us-central1]

COST: a deployed endpoint is billed per node-hour until you undeploy/delete it. GPU options: n1-standard-8 + NVIDIA_TESLA_T4,
g2-standard-8 + NVIDIA_L4, a2-highgpu-1g + NVIDIA_TESLA_A100. Check GPU quota in your project/region first.
"""
import argparse
import sys

from google.cloud import aiplatform


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["deploy", "delete"])
    p.add_argument("--project", required=True)
    p.add_argument("--location", default="us-central1")
    p.add_argument("--image")
    p.add_argument("--name", default="people-yolo")
    p.add_argument("--machine-type", default="n1-standard-8")
    p.add_argument("--accelerator", default="NVIDIA_TESLA_T4")
    p.add_argument("--accelerator-count", type=int, default=1)
    p.add_argument("--min-replicas", type=int, default=1)
    p.add_argument("--max-replicas", type=int, default=1)
    p.add_argument("--endpoint-id")
    p.add_argument("--env", action="append", default=[], help="container env, e.g. MODEL=yolo11m.pt (repeatable)")
    a = p.parse_args()
    aiplatform.init(project=a.project, location=a.location)

    if a.cmd == "deploy":
        if not a.image:
            sys.exit("--image is required for deploy")
        model = aiplatform.Model.upload(
            display_name=a.name, serving_container_image_uri=a.image,
            serving_container_predict_route="/predict", serving_container_health_route="/health", serving_container_ports=[8080],
            serving_container_environment_variables=dict(e.split("=", 1) for e in a.env) or None)
        endpoint = aiplatform.Endpoint.create(display_name=a.name)
        print(f"deploying to endpoint {endpoint.name} (takes ~10-15 min; billed until deleted)…")
        model.deploy(endpoint=endpoint, machine_type=a.machine_type, accelerator_type=a.accelerator,
                     accelerator_count=a.accelerator_count, min_replica_count=a.min_replicas, max_replica_count=a.max_replicas)
        print(f"Deployed. Endpoint id: {endpoint.name}. Stream to it with:\n  python -m cloudstream stream videos/<file> "
              f"--endpoint-type vertex --project {a.project} --location {a.location} --endpoint-id {endpoint.name}")
    else:
        if not a.endpoint_id:
            sys.exit("--endpoint-id is required for delete")
        ep = aiplatform.Endpoint(a.endpoint_id)
        ep.undeploy_all()
        ep.delete()
        print("endpoint deleted (the uploaded Model still exists in the Model Registry; delete it in the console if unwanted)")


if __name__ == "__main__":
    main()

"""How a batch of frames reaches the endpoint. A transport turns request bytes into response bytes.
Each worker thread creates its own transport via the factory, so no client object is shared across threads."""
from __future__ import annotations

import base64
import json
from typing import Callable, Optional


class Transport:
    def invoke(self, body: bytes) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


class HttpTransport(Transport):
    """Any HTTP server speaking the protocol: local container, Vertex dedicated endpoint (with a bearer token), etc."""

    def __init__(self, url: str, headers: Optional[dict] = None, timeout: float = 30.0, route: str = "/invocations"):
        import requests
        self.s = requests.Session()
        self.url = url.rstrip("/") + route if route and not url.rstrip("/").endswith(route) else url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    def invoke(self, body: bytes) -> bytes:
        r = self.s.post(self.url, data=body, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.content


class SageMakerTransport(Transport):
    """Amazon SageMaker real-time endpoint via boto3 `sagemaker-runtime`. Credentials come from the usual AWS chain."""

    def __init__(self, endpoint_name: str, region: Optional[str] = None, client=None, timeout: float = 60.0):
        self.name = endpoint_name
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("sagemaker-runtime", region_name=region,
                                  config=Config(read_timeout=timeout, retries={"max_attempts": 1}, max_pool_connections=32))
        self.client = client

    def invoke(self, body: bytes) -> bytes:
        r = self.client.invoke_endpoint(EndpointName=self.name, ContentType="application/json",
                                        Accept="application/json", Body=body)
        return r["Body"].read()


class VertexTransport(Transport):
    """Vertex AI endpoint via google-cloud-aiplatform.
    mode="raw"     -> Endpoint.raw_predict (arbitrary body, our JSON goes through unchanged)
    mode="predict" -> Endpoint.predict(instances=[{"image": b64}...]) (Vertex's standard instances/predictions envelope)"""

    def __init__(self, project: str, location: str, endpoint_id: str, mode: str = "raw", endpoint=None):
        if endpoint is None:
            from google.cloud import aiplatform
            aiplatform.init(project=project, location=location)
            endpoint = aiplatform.Endpoint(endpoint_id)
        self.ep, self.mode = endpoint, mode

    def invoke(self, body: bytes) -> bytes:
        if self.mode == "raw":
            return self.ep.raw_predict(body=body, headers={"Content-Type": "application/json"}).content
        req = json.loads(body)
        params = {k: req[k] for k in ("conf", "imgsz") if k in req}
        pred = self.ep.predict(instances=[{"image": f} for f in req["frames"]], parameters=params or None)
        return json.dumps({"predictions": list(pred.predictions)}).encode()


def make_factory(args) -> Callable[[], Transport]:
    kind = args.endpoint_type
    if kind == "http":
        hdrs = dict(h.split(":", 1) for h in (args.header or []))
        return lambda: HttpTransport(args.url, {k.strip(): v.strip() for k, v in hdrs.items()}, args.timeout)
    if kind == "sagemaker":
        if not args.endpoint_name:
            raise SystemExit("--endpoint-name is required for --endpoint-type sagemaker")
        return lambda: SageMakerTransport(args.endpoint_name, args.region, timeout=args.timeout)
    if kind == "vertex":
        if not (args.project and args.endpoint_id):
            raise SystemExit("--project and --endpoint-id are required for --endpoint-type vertex")
        return lambda: VertexTransport(args.project, args.location, args.endpoint_id, args.vertex_mode)
    raise SystemExit(f"unknown endpoint type {kind}")

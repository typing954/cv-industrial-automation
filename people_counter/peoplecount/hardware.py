"""Collect hardware / cloud-instance metadata so results from different machines can be compared."""
from __future__ import annotations

import platform
import re
import urllib.request


def _http(url, headers=None, method="GET", timeout=0.4):
    try:
        req = urllib.request.Request(url, headers=headers or {}, method=method)
        return urllib.request.urlopen(req, timeout=timeout).read().decode()
    except Exception:  # noqa: BLE001
        return None


def cloud_info() -> dict:
    """Link-local metadata endpoints only (AWS IMDSv2 / GCP). Fast timeout; returns {} off-cloud."""
    tok = _http("http://169.254.169.254/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds": "60"}, "PUT")
    if tok:
        it = _http("http://169.254.169.254/latest/meta-data/instance-type", {"X-aws-ec2-metadata-token": tok})
        if it:
            return {"provider": "aws", "instance_type": it}
    mt = _http("http://metadata.google.internal/computeMetadata/v1/instance/machine-type", {"Metadata-Flavor": "Google"})
    if mt:
        return {"provider": "gcp", "instance_type": mt.rsplit("/", 1)[-1]}
    return {}


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def collect(detect_cloud: bool = True) -> dict:
    import os

    import psutil
    info = {"platform": platform.platform(), "cpu": cpu_model(), "cpu_cores_logical": os.cpu_count(),
            "cpu_cores_physical": psutil.cpu_count(logical=False), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_mem_gb"] = round(p.total_memory / 2**30, 1)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    if detect_cloud:
        info["cloud"] = cloud_info()
    return info


def auto_tag(info: dict, device: str) -> str:
    c = info.get("cloud") or {}
    base = f"{c['provider']}-{c['instance_type']}" if c else None
    if not base:
        base = info.get("gpu") if device.startswith("cuda") and info.get("gpu") else info.get("cpu", "cpu")
    tag = re.sub(r"[^A-Za-z0-9.]+", "-", base).strip("-")[:48]
    return f"{tag}-{'gpu' if device.startswith('cuda') else device.split(':')[0]}"


class ResourceSampler:
    """Background sampler: process CPU %, peak RSS, GPU util %, GPU mem MB, GPU power W (NVML if present)."""

    def __init__(self, interval: float = 0.25):
        import threading

        import psutil
        self.p, self.interval = psutil.Process(), interval
        self.cpu, self.gpu_util, self.gpu_mem, self.gpu_pw, self.rss = [], [], [], [], 0
        self._stop = threading.Event()
        self._h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv, self._h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            pass
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.p.cpu_percent(None)
        while not self._stop.wait(self.interval):
            self.cpu.append(self.p.cpu_percent(None))
            self.rss = max(self.rss, self.p.memory_info().rss)
            if self._h:
                try:
                    u = self._nv.nvmlDeviceGetUtilizationRates(self._h)
                    self.gpu_util.append(u.gpu)
                    self.gpu_mem.append(self._nv.nvmlDeviceGetMemoryInfo(self._h).used / 2**20)
                    self.gpu_pw.append(self._nv.nvmlDeviceGetPowerUsage(self._h) / 1000)
                except Exception:  # noqa: BLE001
                    pass

    def start(self):
        self.t.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self.t.join(1)
        avg = lambda x: round(sum(x) / len(x), 1) if x else None  # noqa: E731
        return {"cpu_pct_avg": avg(self.cpu), "cpu_pct_peak": round(max(self.cpu), 1) if self.cpu else None,
                "rss_peak_mb": round(self.rss / 2**20), "gpu_util_avg": avg(self.gpu_util),
                "gpu_mem_peak_mb": round(max(self.gpu_mem)) if self.gpu_mem else None, "gpu_power_avg_w": avg(self.gpu_pw)}

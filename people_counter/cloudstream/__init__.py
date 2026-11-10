"""cloudstream: stream video frames to a hosted YOLO endpoint (SageMaker / Vertex AI / any HTTP server).

Kept separate from `peoplecount/` (the local, in-process pipeline). This package only *imports* from peoplecount
(alert rules, clip recorder, tracker, overlay); peoplecount knows nothing about it.
"""
__version__ = "0.1.0"

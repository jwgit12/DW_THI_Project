"""fODF training and evaluation package.

Sibling to :mod:`dw_thi`. Imports shared infrastructure (preprocessing,
augmentation, runtime helpers, baseline metrics) directly from ``dw_thi``.
The Qt viewer is the project-root ``visualizer.py``, shared with the standard
pipeline.
"""

__all__ = [
    "dataset",
    "defaults",
    "evaluate",
    "loss",
    "model",
    "train",
]

"""Minimal stand-in for the three pytorch3d functions SDFR calls (pytorch3d has no build for torch 2.11 / Windows).
The functions follow pytorch3d's own implementations (transforms/so3.py, ops/sample_farthest_points.py) and are
checked numerically by tools/sdfr_port/verify_port.py."""
from . import ops, transforms  # noqa: F401

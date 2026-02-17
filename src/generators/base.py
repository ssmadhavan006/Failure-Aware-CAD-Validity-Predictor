"""
Base class for all CAD model generators.

Each generator produces samples for one failure class.  A sample is a dict
containing the input parameters, the resulting OCC shape (or None on
construction failure), and metadata used for labeling.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Failure class labels
LABEL_VALID = 0
LABEL_SELF_INTERSECTION = 1
LABEL_NON_MANIFOLD = 2
LABEL_DEGENERATE = 3
LABEL_TOLERANCE = 4

LABEL_NAMES = {
    LABEL_VALID: "valid",
    LABEL_SELF_INTERSECTION: "self_intersection",
    LABEL_NON_MANIFOLD: "non_manifold",
    LABEL_DEGENERATE: "degenerate_face",
    LABEL_TOLERANCE: "tolerance_error",
}


@dataclass
class Sample:
    """One generated CAD sample."""

    params: dict[str, Any]
    label: int
    failure_mode: str  # human-readable, e.g. "bowtie_extrude"
    sub_family: str  # generator sub-family name
    shape: Any = None  # OCP TopoDS_Shape or None
    error_type: str | None = None  # exception class name if construction failed
    error_msg: str | None = None
    is_valid: bool | None = None  # result of BRepCheck_Analyzer
    topology: dict[str, int] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Serialise to a flat dict suitable for JSON Lines."""
        # kernel_label: use actual kernel result as ground truth.
        # If the kernel says the shape is valid (and no construction error),
        # label it as valid (0) regardless of the intended class.
        if self.is_valid is True and self.error_type is None:
            kernel_label = LABEL_VALID
        elif self.is_valid is False or self.error_type is not None:
            kernel_label = self.label  # intended failure class
        else:
            kernel_label = self.label

        rec = {
            "label": kernel_label,  # ground truth from kernel
            "intended_label": self.label,  # what the generator tried to create
            "label_name": LABEL_NAMES.get(kernel_label, "unknown"),
            "intended_label_name": LABEL_NAMES.get(self.label, "unknown"),
            "failure_mode": self.failure_mode,
            "sub_family": self.sub_family,
            "is_valid": self.is_valid,
            "error_type": self.error_type,
            "error_msg": self.error_msg,
        }
        # Flatten params with prefix
        for k, v in self.params.items():
            rec[f"param_{k}"] = v
        # Flatten topology
        for k, v in self.topology.items():
            rec[f"topo_{k}"] = v
        return rec


class BaseGenerator(abc.ABC):
    """Abstract base for a parametric CAD generator."""

    label: int  # class label (set by subclass)
    name: str  # human-readable name

    @abc.abstractmethod
    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Return a random parameter dict for one sample."""

    @abc.abstractmethod
    def build(self, params: dict[str, Any]) -> Sample:
        """
        Build a CAD shape from *params* and return a Sample.

        The implementation should:
        1. Attempt to construct the shape via CadQuery / OCP.
        2. Catch any construction exceptions and record them.
        3. If a shape was produced, leave validity checking to the
           pipeline (kernel_check module) — just set ``sample.shape``.
        """

    # Convenience ----------------------------------------------------------

    def generate(self, rng: np.random.Generator) -> Sample:
        """Sample params then build — single entry-point for the pipeline."""
        params = self.sample_params(rng)
        return self.build(params)

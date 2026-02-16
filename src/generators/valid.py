"""
Generator for **valid** CAD shapes (label 0).

Sub-families
  • primitive_box      — random box
  • primitive_cylinder  — random cylinder
  • primitive_sphere    — random sphere
  • boolean_union       — fuse of two overlapping primitives
  • filleted_box        — box with safe fillet
"""

from __future__ import annotations

from typing import Any

import numpy as np
import cadquery as cq

from .base import BaseGenerator, Sample, LABEL_VALID


class ValidGenerator(BaseGenerator):
    label = LABEL_VALID
    name = "valid"

    # relative weights for sub-family sampling
    _families = [
        "primitive_box",
        "primitive_cylinder",
        "primitive_sphere",
        "boolean_union",
        "filleted_box",
    ]

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        family = rng.choice(self._families)
        p: dict[str, Any] = {"family": family}

        if family == "primitive_box":
            p["length"] = float(rng.uniform(5, 100))
            p["width"] = float(rng.uniform(5, 100))
            p["height"] = float(rng.uniform(5, 100))

        elif family == "primitive_cylinder":
            p["radius"] = float(rng.uniform(2, 50))
            p["height"] = float(rng.uniform(5, 100))

        elif family == "primitive_sphere":
            p["radius"] = float(rng.uniform(2, 50))

        elif family == "boolean_union":
            p["l1"] = float(rng.uniform(5, 50))
            p["w1"] = float(rng.uniform(5, 50))
            p["h1"] = float(rng.uniform(5, 50))
            p["l2"] = float(rng.uniform(5, 50))
            p["w2"] = float(rng.uniform(5, 50))
            p["h2"] = float(rng.uniform(5, 50))
            # offset large enough so shapes overlap but are clearly valid
            p["ox"] = float(rng.uniform(2, p["l1"] * 0.8))
            p["oy"] = float(rng.uniform(2, p["w1"] * 0.8))
            p["oz"] = float(rng.uniform(0, p["h1"] * 0.5))

        elif family == "filleted_box":
            p["length"] = float(rng.uniform(10, 80))
            p["width"] = float(rng.uniform(10, 80))
            p["height"] = float(rng.uniform(10, 80))
            min_edge = min(p["length"], p["width"], p["height"])
            # safe fillet: 5 % – 35 % of the shortest edge
            p["fillet_r"] = float(rng.uniform(0.05, 0.35) * min_edge)

        return p

    def build(self, params: dict[str, Any]) -> Sample:
        family = params["family"]
        try:
            if family == "primitive_box":
                shape = (
                    cq.Workplane("XY")
                    .box(params["length"], params["width"], params["height"])
                    .val()
                    .wrapped
                )

            elif family == "primitive_cylinder":
                shape = (
                    cq.Workplane("XY")
                    .cylinder(params["height"], params["radius"])
                    .val()
                    .wrapped
                )

            elif family == "primitive_sphere":
                shape = cq.Workplane("XY").sphere(params["radius"]).val().wrapped

            elif family == "boolean_union":
                b1 = cq.Workplane("XY").box(params["l1"], params["w1"], params["h1"])
                b2 = (
                    cq.Workplane("XY")
                    .transformed(offset=(params["ox"], params["oy"], params["oz"]))
                    .box(params["l2"], params["w2"], params["h2"])
                )
                shape = b1.union(b2).val().wrapped

            elif family == "filleted_box":
                shape = (
                    cq.Workplane("XY")
                    .box(params["length"], params["width"], params["height"])
                    .edges()
                    .fillet(params["fillet_r"])
                    .val()
                    .wrapped
                )
            else:
                raise ValueError(f"Unknown family: {family}")

            return Sample(
                params=params,
                label=self.label,
                failure_mode="none",
                sub_family=family,
                shape=shape,
            )

        except Exception as exc:
            return Sample(
                params=params,
                label=self.label,
                failure_mode="construction_error",
                sub_family=family,
                shape=None,
                error_type=type(exc).__name__,
                error_msg=str(exc)[:200],
            )

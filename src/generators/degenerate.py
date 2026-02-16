"""
Generator for **degenerate face** geometry (label 3).

All sub-families produce shapes with extreme proportions or degenerate
topology that a CAD kernel would flag as problematic.

Sub-families
  • zero_dim_box         — box with one dimension near zero (extreme flatness)
  • near_zero_extrude    — extrusion height extremely small
  • extreme_aspect_ratio — one dimension vastly smaller than others
  • collinear_extrude    — nearly-degenerate triangle extruded
"""

from __future__ import annotations

from typing import Any

import numpy as np
import cadquery as cq

from .base import BaseGenerator, Sample, LABEL_DEGENERATE


class DegenerateGenerator(BaseGenerator):
    label = LABEL_DEGENERATE
    name = "degenerate"

    _families = [
        "zero_dim_box",
        "near_zero_extrude",
        "extreme_aspect_ratio",
        "collinear_extrude",
    ]

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        family = rng.choice(self._families)
        p: dict[str, Any] = {"family": family}

        if family == "zero_dim_box":
            p["length"] = float(rng.uniform(5, 50))
            p["width"] = float(rng.uniform(5, 50))
            # Extremely thin but nonzero so OCP can build it
            # Range: 1e-6 to 1e-4 (sub-tolerance to near-tolerance)
            p["height"] = float(10 ** rng.uniform(-6, -4))

        elif family == "near_zero_extrude":
            p["base_l"] = float(rng.uniform(5, 50))
            p["base_w"] = float(rng.uniform(5, 50))
            # Extremely small extrusion height
            p["height"] = float(10 ** rng.uniform(-6, -4))

        elif family == "extreme_aspect_ratio":
            p["length"] = float(rng.uniform(100, 1000))
            p["width"] = float(rng.uniform(100, 1000))
            # Extremely thin in one dimension → aspect ratio 10^5+
            p["height"] = float(10 ** rng.uniform(-5, -3))

        elif family == "collinear_extrude":
            length = float(rng.uniform(5, 50))
            p["length"] = length
            p["extrude_h"] = float(rng.uniform(2, 30))
            # Almost collinear — deviation is tiny but nonzero
            p["deviation"] = float(10 ** rng.uniform(-5, -3))

        return p

    def build(self, params: dict[str, Any]) -> Sample:
        family = params["family"]
        try:
            if family == "zero_dim_box":
                shape = (
                    cq.Workplane("XY")
                    .box(params["length"], params["width"], params["height"])
                    .val()
                    .wrapped
                )

            elif family == "near_zero_extrude":
                shape = (
                    cq.Workplane("XY")
                    .rect(params["base_l"], params["base_w"])
                    .extrude(params["height"])
                    .val()
                    .wrapped
                )

            elif family == "extreme_aspect_ratio":
                shape = (
                    cq.Workplane("XY")
                    .box(params["length"], params["width"], params["height"])
                    .val()
                    .wrapped
                )

            elif family == "collinear_extrude":
                length = params["length"]
                d = params["deviation"]
                pts = [(0, 0), (length, 0), (length / 2, d)]
                shape = (
                    cq.Workplane("XY")
                    .polyline(pts)
                    .close()
                    .extrude(params["extrude_h"])
                    .val()
                    .wrapped
                )

            else:
                raise ValueError(f"Unknown family: {family}")

            return Sample(
                params=params,
                label=self.label,
                failure_mode="degenerate_face",
                sub_family=family,
                shape=shape,
            )

        except Exception as exc:
            return Sample(
                params=params,
                label=self.label,
                failure_mode="degenerate_face",
                sub_family=family,
                shape=None,
                error_type=type(exc).__name__,
                error_msg=str(exc)[:200],
            )

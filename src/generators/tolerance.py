"""
Generator for **tolerance error** shapes (label 4).

Sub-families
  • sub_tolerance_box       — all dimensions near OCC tolerance (~1e-7)
  • scale_mismatch_boolean  — boolean with 4+ orders of magnitude size difference
  • micro_fillet            — fillet with radius near kernel tolerance on a small box
  • near_coincident_boolean — boolean of shapes offset by near-zero amount
"""

from __future__ import annotations

from typing import Any

import numpy as np
import cadquery as cq

from .base import BaseGenerator, Sample, LABEL_TOLERANCE


class ToleranceGenerator(BaseGenerator):
    label = LABEL_TOLERANCE
    name = "tolerance"

    _families = [
        "sub_tolerance_box",
        "scale_mismatch_boolean",
        "micro_fillet",
        "near_coincident_boolean",
    ]

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        family = rng.choice(self._families)
        p: dict[str, Any] = {"family": family}

        if family == "sub_tolerance_box":
            # Dimensions near OCC tolerance but large enough to build
            p["length"] = float(10 ** rng.uniform(-5, -3))
            p["width"] = float(10 ** rng.uniform(-5, -3))
            p["height"] = float(10 ** rng.uniform(-5, -3))

        elif family == "scale_mismatch_boolean":
            # Large shape + small shape → tolerance confusion at interface
            p["large_size"] = float(rng.uniform(50, 200))
            p["tiny_size"] = float(10 ** rng.uniform(-4, -2))
            # Offset tiny shape near a face of the large one
            p["offset_x"] = float(p["large_size"] / 2 * rng.uniform(0.8, 1.0))
            p["offset_y"] = 0.0
            p["offset_z"] = 0.0

        elif family == "micro_fillet":
            # Small box with very small fillet
            p["box_size"] = float(rng.uniform(0.01, 1.0))
            p["fillet_r"] = float(10 ** rng.uniform(-5, -3))

        elif family == "near_coincident_boolean":
            # Two shapes at very small offset
            p["size"] = float(rng.uniform(5, 50))
            p["offset"] = float(10 ** rng.uniform(-5, -3))

        return p

    def build(self, params: dict[str, Any]) -> Sample:
        family = params["family"]
        try:
            if family == "sub_tolerance_box":
                shape = (
                    cq.Workplane("XY")
                    .box(params["length"], params["width"], params["height"])
                    .val()
                    .wrapped
                )

            elif family == "scale_mismatch_boolean":
                ls = params["large_size"]
                big = cq.Workplane("XY").box(ls, ls, ls)
                ts = params["tiny_size"]
                tiny = (
                    cq.Workplane("XY")
                    .transformed(
                        offset=(
                            params["offset_x"],
                            params["offset_y"],
                            params["offset_z"],
                        )
                    )
                    .box(ts, ts, ts)
                )
                shape = big.union(tiny).val().wrapped

            elif family == "micro_fillet":
                bs = params["box_size"]
                shape = (
                    cq.Workplane("XY")
                    .box(bs, bs, bs)
                    .edges()
                    .fillet(params["fillet_r"])
                    .val()
                    .wrapped
                )

            elif family == "near_coincident_boolean":
                sz = params["size"]
                b1 = cq.Workplane("XY").box(sz, sz, sz)
                b2 = (
                    cq.Workplane("XY")
                    .transformed(offset=(params["offset"], 0, 0))
                    .box(sz, sz, sz)
                )
                shape = b1.union(b2).val().wrapped

            else:
                raise ValueError(f"Unknown family: {family}")

            return Sample(
                params=params,
                label=self.label,
                failure_mode="tolerance_error",
                sub_family=family,
                shape=shape,
            )

        except Exception as exc:
            return Sample(
                params=params,
                label=self.label,
                failure_mode="tolerance_error",
                sub_family=family,
                shape=None,
                error_type=type(exc).__name__,
                error_msg=str(exc)[:200],
            )

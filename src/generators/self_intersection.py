"""
Generator for **self‑intersecting** CAD shapes (label 1).

Sub-families
  • bowtie_extrude      — 4-point crossing polyline extruded
  • twisted_polygon     — N-gon with crossing edges extruded
  • multi_cross         — polyline with multiple crossing points
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import cadquery as cq

from .base import BaseGenerator, Sample, LABEL_SELF_INTERSECTION


class SelfIntersectionGenerator(BaseGenerator):
    label = LABEL_SELF_INTERSECTION
    name = "self_intersection"

    _families = [
        "bowtie_extrude",
        "twisted_polygon",
        "multi_cross",
    ]

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        family = rng.choice(self._families)
        p: dict[str, Any] = {"family": family}

        if family == "bowtie_extrude":
            # Classic X-shaped crossing polyline
            w = float(rng.uniform(5, 50))  # width of the bowtie
            h = float(rng.uniform(5, 50))  # height of the bowtie
            p["w"] = w
            p["h"] = h
            p["extrude"] = float(rng.uniform(2, 40))
            # Offset the crossing point from center for variety
            p["cx"] = float(rng.uniform(0.3, 0.7) * w)
            p["cy"] = float(rng.uniform(0.3, 0.7) * h)

        elif family == "twisted_polygon":
            # A polygon whose edges cross — e.g. a star-like shape
            # with vertices reordered to create crossings
            p["n_points"] = int(rng.integers(4, 8))
            p["radius"] = float(rng.uniform(5, 40))
            p["extrude"] = float(rng.uniform(2, 30))
            # Twist: shuffle vertex order to guarantee crossings
            p["twist_seed"] = int(rng.integers(0, 2**31))

        elif family == "multi_cross":
            # Zig-zag polyline that crosses itself multiple times
            p["n_segments"] = int(rng.integers(4, 8))
            p["amplitude"] = float(rng.uniform(5, 30))
            p["wavelength"] = float(rng.uniform(3, 15))
            p["extrude"] = float(rng.uniform(2, 30))

        return p

    def _bowtie_points(self, params):
        """Create a bowtie (X-crossing) polyline."""
        w, h = params["w"], params["h"]
        cx, cy = params["cx"], params["cy"]
        # The four corners + crossing creates the self-intersection
        return [
            (0.0, 0.0),
            (w, h),
            (w, 0.0),
            (0.0, h),
        ]

    def _twisted_polygon_points(self, params):
        """Create an N-gon with vertices reordered to cause crossings."""
        n = params["n_points"]
        r = params["radius"]
        rng = np.random.default_rng(params["twist_seed"])

        # Generate regular polygon vertices
        angles = [2 * math.pi * i / n for i in range(n)]
        pts = [(r * math.cos(a), r * math.sin(a)) for a in angles]

        # Swap non-adjacent pairs to guarantee edge crossings
        indices = list(range(n))
        # Swap every other pair
        for i in range(0, n - 2, 2):
            indices[i], indices[i + 1] = indices[i + 1], indices[i]
        pts = [pts[i] for i in indices]

        return pts

    def _multi_cross_points(self, params):
        """Create a zig-zag that crosses itself."""
        n = params["n_segments"]
        amp = params["amplitude"]
        wl = params["wavelength"]

        pts = [(0.0, 0.0)]
        for i in range(1, n + 1):
            x = i * wl
            # Alternate y but with decreasing then increasing pattern
            # This creates crossing segments
            if i % 2 == 1:
                y = amp * (1.0 - (i / (n + 1)))
            else:
                y = -amp * (1.0 - (i / (n + 1)))
            pts.append((x, y))

        # Close back through the middle to guarantee crossing
        mid_x = (n * wl) / 2
        pts.append((mid_x, amp * 0.8))
        pts.append((mid_x, -amp * 0.8))

        return pts

    def build(self, params: dict[str, Any]) -> Sample:
        family = params["family"]
        try:
            if family == "bowtie_extrude":
                pts = self._bowtie_points(params)
            elif family == "twisted_polygon":
                pts = self._twisted_polygon_points(params)
            elif family == "multi_cross":
                pts = self._multi_cross_points(params)
            else:
                raise ValueError(f"Unknown family: {family}")

            result = cq.Workplane("XY").polyline(pts).close().extrude(params["extrude"])
            shape = result.val().wrapped

            return Sample(
                params=params,
                label=self.label,
                failure_mode="self_intersection",
                sub_family=family,
                shape=shape,
            )

        except Exception as exc:
            return Sample(
                params=params,
                label=self.label,
                failure_mode="self_intersection",
                sub_family=family,
                shape=None,
                error_type=type(exc).__name__,
                error_msg=str(exc)[:200],
            )

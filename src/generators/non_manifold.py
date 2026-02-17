"""
Generator for **non-manifold** topology (label 2).

Sub-families
  • face_sharing_compound  — two solids sharing a face, placed in a compound
  • edge_sharing_compound  — two solids sharing only an edge
  • vertex_sharing_compound — two solids sharing only a vertex
  • open_shell             — a box with faces removed, attempted as solid
  • t_junction             — a box with another box butting mid-face
"""

from __future__ import annotations

from typing import Any

import numpy as np
import cadquery as cq

from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCP.TopoDS import TopoDS_Compound
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.gp import gp_Pnt

from .base import BaseGenerator, Sample, LABEL_NON_MANIFOLD


class NonManifoldGenerator(BaseGenerator):
    label = LABEL_NON_MANIFOLD
    name = "non_manifold"

    _families = [
        "face_sharing_compound",
        "edge_sharing_compound",
        "vertex_sharing_compound",
        "open_shell",
        "t_junction",
    ]

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        family = rng.choice(self._families)
        p: dict[str, Any] = {"family": family}

        if family in (
            "face_sharing_compound",
            "edge_sharing_compound",
            "vertex_sharing_compound",
        ):
            p["s1"] = float(rng.uniform(5, 50))  # size of box 1
            p["s2"] = float(rng.uniform(5, 50))  # size of box 2

        elif family == "open_shell":
            p["length"] = float(rng.uniform(5, 50))
            p["width"] = float(rng.uniform(5, 50))
            p["height"] = float(rng.uniform(5, 50))
            # How many faces to KEEP (out of 6) → 3–5 keeps it open
            p["n_faces_keep"] = int(rng.integers(3, 6))

        elif family == "t_junction":
            p["main_size"] = float(rng.uniform(10, 50))
            p["branch_w"] = float(rng.uniform(2, 10))
            p["branch_h"] = float(rng.uniform(2, 10))
            p["branch_d"] = float(rng.uniform(5, 30))

        return p

    def build(self, params: dict[str, Any]) -> Sample:
        family = params["family"]
        try:
            if family == "face_sharing_compound":
                shape = self._face_sharing(params)
            elif family == "edge_sharing_compound":
                shape = self._edge_sharing(params)
            elif family == "vertex_sharing_compound":
                shape = self._vertex_sharing(params)
            elif family == "open_shell":
                shape = self._open_shell(params)
            elif family == "t_junction":
                shape = self._t_junction(params)
            else:
                raise ValueError(f"Unknown family: {family}")

            return Sample(
                params=params,
                label=self.label,
                failure_mode="non_manifold",
                sub_family=family,
                shape=shape,
            )

        except Exception as exc:
            return Sample(
                params=params,
                label=self.label,
                failure_mode="non_manifold",
                sub_family=family,
                shape=None,
                error_type=type(exc).__name__,
                error_msg=str(exc)[:200],
            )

    # sub-family builders

    def _face_sharing(self, p):
        """Two boxes sharing a face, combined into a compound (NOT fused)."""
        s1, s2 = p["s1"], p["s2"]
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        box1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(s1, s1, s1)).Shape()
        # Second box starts exactly where the first ends → shared face
        box2 = BRepPrimAPI_MakeBox(gp_Pnt(s1, 0, 0), gp_Pnt(s1 + s2, s2, s2)).Shape()
        builder.Add(compound, box1)
        builder.Add(compound, box2)
        return compound

    def _edge_sharing(self, p):
        """Two boxes sharing only an edge."""
        s1, s2 = p["s1"], p["s2"]
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        box1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(s1, s1, s1)).Shape()
        box2 = BRepPrimAPI_MakeBox(
            gp_Pnt(s1, s1, 0), gp_Pnt(s1 + s2, s1 + s2, s2)
        ).Shape()
        builder.Add(compound, box1)
        builder.Add(compound, box2)
        return compound

    def _vertex_sharing(self, p):
        """Two boxes sharing only a vertex."""
        s1, s2 = p["s1"], p["s2"]
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        box1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(s1, s1, s1)).Shape()
        box2 = BRepPrimAPI_MakeBox(
            gp_Pnt(s1, s1, s1), gp_Pnt(s1 + s2, s1 + s2, s1 + s2)
        ).Shape()
        builder.Add(compound, box1)
        builder.Add(compound, box2)
        return compound

    def _open_shell(self, p):
        """
        Take a box, keep only N of its 6 faces, sew them into a shell,
        and attempt to make a solid from the incomplete shell.
        """
        box = BRepPrimAPI_MakeBox(p["length"], p["width"], p["height"]).Shape()
        sew = BRepBuilderAPI_Sewing(1e-6)

        exp = TopExp_Explorer(box, TopAbs_FACE)
        i = 0
        while exp.More():
            if i < p["n_faces_keep"]:
                sew.Add(exp.Current())
            i += 1
            exp.Next()

        sew.Perform()
        sewn = sew.SewedShape()

        # Attempt to force a solid from the open shell
        try:
            solid_maker = BRepBuilderAPI_MakeSolid()
            # Try to feed the shell — this often fails or produces
            # a non-manifold solid
            from OCP.TopoDS import TopoDS

            shell_exp = TopExp_Explorer(sewn, TopAbs_FACE)
            # Just return the sewn shape (incomplete shell pretending to be geometry)
            return sewn
        except Exception:
            return sewn

    def _t_junction(self, p):
        """
        A main box with a branch box butting against the middle of a face.
        Combined as compound (not boolean fused) — creates non-manifold
        topology where surfaces meet without proper connectivity.
        """
        ms = p["main_size"]
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)

        main_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(ms, ms, ms)).Shape()

        # Branch box butting mid-face
        bw, bh, bd = p["branch_w"], p["branch_h"], p["branch_d"]
        branch_x = ms  # flush against +X face
        branch_y = (ms - bw) / 2  # centred on face
        branch_z = (ms - bh) / 2
        branch = BRepPrimAPI_MakeBox(
            gp_Pnt(branch_x, branch_y, branch_z),
            gp_Pnt(branch_x + bd, branch_y + bw, branch_z + bh),
        ).Shape()

        builder.Add(compound, main_box)
        builder.Add(compound, branch)
        return compound

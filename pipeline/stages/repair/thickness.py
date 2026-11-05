"""Wall-thickness heuristic: report-only, never auto-fixes (see
trimesh_repair.py docstring for the auto-fix/disclose/flag rationale).
Kept in its own module so it's unit-testable independent of the rest of the
repair pipeline.
"""

from __future__ import annotations

import numpy as np
import trimesh


def estimate_min_wall_thickness_mm(
    mesh: trimesh.Trimesh,
    scale_to_mm: float,
    sample_count: int,
    floor_mm: float,
    critical_mm: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Samples points on the surface, casts a ray inward along the negated
    normal, and measures the distance to the first back-facing self-hit as a
    cheap proxy for local wall thickness. Not a true medial-axis thickness
    field (that's a heavier computation) — a conservative heuristic, which
    is why this stays report-only rather than driving any auto-repair.

    `critical_mm` (typically config.WALL_THICKNESS_CRITICAL_MM, one nozzle
    width) distinguishes "likely an actual gap/hole when sliced" from
    "merely below the robust floor" — two very different severities that
    used to get an identical warning. Optional so existing/test call sites
    that only care about the single floor_mm threshold don't have to pass
    it; the critical fraction is None (not computed) when omitted.

    Returns (min_thickness_mm, fraction_of_samples_below_floor,
    fraction_of_samples_below_critical), or (None, None, None) if no rays
    hit anything (degenerate/empty mesh) or critical_mm wasn't given (third
    element only).
    """
    if len(mesh.faces) == 0:
        return None, None, None

    points, face_indices = trimesh.sample.sample_surface(mesh, sample_count)
    normals = mesh.face_normals[face_indices]

    # Nudge origins slightly inside the surface so the ray doesn't
    # immediately self-intersect its own starting triangle.
    eps = mesh.scale * 1e-5 if mesh.scale > 0 else 1e-6
    origins = points - normals * eps
    directions = -normals

    locations, index_ray, _index_tri = mesh.ray.intersects_location(
        origins, directions
    )
    if len(locations) == 0:
        return None, None, None

    distances = np.linalg.norm(locations - origins[index_ray], axis=1)

    # Keep the nearest hit per originating ray (a ray can cross multiple
    # back faces for non-convex geometry; the nearest is the local wall).
    nearest_by_ray: dict[int, float] = {}
    for ray_idx, dist in zip(index_ray, distances):
        if ray_idx not in nearest_by_ray or dist < nearest_by_ray[ray_idx]:
            nearest_by_ray[ray_idx] = dist

    thicknesses_mm = np.array(list(nearest_by_ray.values())) * scale_to_mm
    if len(thicknesses_mm) == 0:
        return None, None, None

    min_thickness = float(thicknesses_mm.min())
    fraction_below = float((thicknesses_mm < floor_mm).mean())
    fraction_critical = float((thicknesses_mm < critical_mm).mean()) if critical_mm is not None else None
    return min_thickness, fraction_below, fraction_critical


def thin_wall_issue(
    below_floor_fraction: float | None,
    critical_fraction: float | None,
    advice: str,
):
    """Builds the tiered `thin_walls` WARNING (or None if under threshold).

    Shared by the repair stage (first measurement) and the print-prep stage
    (re-measurement after any wall offset), so both report identically.
    `advice` is the closing sentence, since what the user can do about it
    differs by stage (e.g. "thickening was off" vs. "still thin after a
    0.6mm offset").
    """
    from pipeline import config
    from pipeline.types import Issue, Severity

    if below_floor_fraction is None or below_floor_fraction <= config.WALL_THICKNESS_WARNING_FRACTION:
        return None
    # Tiered, not a single number: a knife-edge point (likely an actual
    # gap/hole once sliced) and a wall that's uniformly 0.9mm (printable as a
    # fragile 1-2 perimeter wall) are genuinely different severities. See
    # config.WALL_THICKNESS_CRITICAL_MM for where the line is.
    fragile_fraction = below_floor_fraction - (critical_fraction or 0.0)
    if critical_fraction and critical_fraction > config.WALL_THICKNESS_WARNING_FRACTION:
        breakdown = (
            f"{critical_fraction * 100:.1f}% is below even one nozzle width "
            f"({config.WALL_THICKNESS_CRITICAL_MM}mm) — likely to come out as actual "
            f"gaps/holes once sliced, not just a fragile wall. The remaining "
            f"{fragile_fraction * 100:.1f}% should still print as a thin (1-2 perimeter) "
            "but present wall."
        )
    else:
        breakdown = (
            "None of this is below one nozzle width, so it should still print as a "
            "thin (1-2 perimeter) wall rather than an outright gap — just not the fully "
            "robust 3-perimeter thickness."
        )
    return Issue(
        code="thin_walls",
        message=(
            f"{below_floor_fraction * 100:.1f}% of sampled surface points have "
            f"estimated wall thickness below the {config.FDM_MIN_WALL_THICKNESS_MM}mm "
            f"FDM floor. {breakdown} {advice}"
        ),
        severity=Severity.WARNING,
    )

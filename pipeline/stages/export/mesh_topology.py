"""Post-export STL topology verification and repair.

STL is a triangle-soup format: it stores each triangle's 3 vertices
independently, with no shared-index topology at all. Any consumer of an STL
file (a slicer, or our own validation code) must reconstruct the mesh's
real topology by welding vertices that land at (near-)identical positions.
This module exists because that welding step can itself introduce a small
number of non-manifold edges, even when the mesh was verified perfectly
watertight *before* export — discovered during development on real
TRELLIS.2 output that had been through the repair stage's detail-recovery
vertex-snap (see pipeline/stages/repair/trimesh_repair.py). Since a real
slicer will do essentially the same position-based welding when it loads
the file, this must be fixed in the actual exported file, not just
tolerated as a quirk of our own validation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import trimesh


def find_sliver_edges(mesh: trimesh.Trimesh, length_multiple: float) -> np.ndarray:
    """Returns edges (as [n,2] vertex-index pairs) whose length is a large
    outlier relative to the mesh's own median edge length -- a degenerate
    "sliver" triangle stretched across empty space, not a real surface
    feature. Diagnosed from a real user-reported incident (2026-09-18): a
    Godzilla STL showed long thin spikes shooting out from the body in Bambu
    Studio. Traced to PyMeshLab's `meshing_decimation_quadric_edge_collapse`
    (see trimesh_repair.py's _voxel_remesh): on complex/messy marching-cubes
    input it can occasionally collapse an edge in a way that produces one of
    these, even with `preservetopology=True` -- that flag only guarantees no
    non-manifold edges or genus change, not sane triangle aspect ratios, so
    a topologically "valid" mesh can still contain a triangle with one
    absurdly long edge. Confirmed present in the raw decimated marching-
    cubes output itself, before any smoothing or detail-recovery snap runs
    (max edge 44.6mm in an 80mm model, vs. a 0.3mm median) -- this is not
    something those later steps introduce or can be blamed for.

    A global median-based threshold is safe here specifically because this
    module's input is voxel-remesh output: a fairly uniform-resolution grid
    surface, not a mesh with legitimately wide variation in local triangle
    size across regions. `length_multiple` needs to comfortably clear normal
    surface-detail edges while still catching genuine slivers -- 15x median
    was chosen as a conservative middle ground, not tuned to the exact
    incident numbers (198-514 flagged edges were seen well above this).
    """
    edges_sorted = np.sort(mesh.edges, axis=1)
    unique_edges = np.unique(edges_sorted, axis=0)
    lengths = np.linalg.norm(mesh.vertices[unique_edges[:, 0]] - mesh.vertices[unique_edges[:, 1]], axis=1)
    median = np.median(lengths)
    if median <= 0:
        return np.empty((0, 2), dtype=unique_edges.dtype)
    return unique_edges[lengths > length_multiple * median]


def remove_slivers(mesh: trimesh.Trimesh, length_multiple: float, max_iterations: int = 5) -> trimesh.Trimesh:
    """Detects and removes degenerate sliver edges (see find_sliver_edges),
    then immediately fan-fills the resulting hole(s) -- keeping the mesh
    watertight at every step, the invariant _voxel_remesh's caller relies on
    for its smoothed-but-unsnapped fallback path. A no-op (returns the input
    unchanged) when nothing is flagged.

    Iterates (bounded, cheap — a handful of edges at most per pass) because
    fan-filling a larger hole can itself introduce a new, smaller-but-still-
    outlying edge at the closure — confirmed necessary empirically on the
    owl test mesh, where a single pass reduced but didn't fully eliminate
    its worst sliver (27.6x median -> 19.0x median, still above threshold)."""
    current = mesh
    for _ in range(max_iterations):
        sliver_edges = find_sliver_edges(current, length_multiple)
        if len(sliver_edges) == 0:
            break
        cleaned = _remove_faces_touching_edges(current, sliver_edges)
        cleaned.merge_vertices(digits_vertex=8)
        boundary, _ = _boundary_and_nonmanifold_edges(cleaned)
        if len(boundary) > 0:
            cleaned = _fan_fill_boundary_loops(cleaned, boundary)
            cleaned.merge_vertices(digits_vertex=8)
        current = cleaned
    return current


def _boundary_and_nonmanifold_edges(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    edges_sorted = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    return unique_edges[counts == 1], unique_edges[counts > 2]


def _fan_fill_boundary_loops(mesh: trimesh.Trimesh, boundary_edges: np.ndarray) -> trimesh.Trimesh:
    """Closes simple boundary loops by adding a centroid vertex and fanning
    triangles to it. Always topologically valid for a simple closed loop,
    unlike trimesh's own fill_holes() (which uses ear-clipping and can
    refuse non-planar loops — observed on real output here). Uses networkx's
    cycle_basis per connected component so a "figure-8" loop (two holes
    touching at one shared vertex, seen in practice after vertex welding)
    gets filled as two separate simple cycles rather than skipped."""
    g = nx.Graph()
    g.add_edges_from(tuple(e) for e in boundary_edges)

    new_vertices = list(mesh.vertices)
    new_faces = list(mesh.faces)
    for component in nx.connected_components(g):
        sub = g.subgraph(component)
        for cycle in nx.cycle_basis(sub):
            if len(cycle) < 3:
                continue
            centroid = mesh.vertices[cycle].mean(axis=0)
            centroid_idx = len(new_vertices)
            new_vertices.append(centroid)
            n = len(cycle)
            for i in range(n):
                a, b = cycle[i], cycle[(i + 1) % n]
                new_faces.append([a, b, centroid_idx])

    return trimesh.Trimesh(vertices=np.array(new_vertices), faces=np.array(new_faces), process=False)


def _clear_junction_vertices(mesh: trimesh.Trimesh, boundary_edges: np.ndarray) -> trimesh.Trimesh | None:
    """Handles a case `_fan_fill_boundary_loops` can't: a boundary vertex
    shared by 2+ separate loops (degree > 2 in the boundary graph — e.g.
    three holes all touching at one point). `nx.cycle_basis` does return
    cycles for such a graph, but a basis cycle is only required to span the
    graph's cycle *space* (an algebraic/GF(2) property) — it doesn't have
    to be one of the geometrically distinct loops meeting at the junction,
    so fan-filling the basis cycles as-is can plateau indefinitely without
    closing anything (observed in practice on real Godzilla/owl meshes).

    Removing every face touching a junction vertex turns the single
    ambiguous multi-loop point into a plain (larger, but unambiguous) hole
    — which the next _fan_fill_boundary_loops call can then close normally,
    since every remaining boundary vertex has degree exactly 2. Returns
    None if there's no junction vertex to clear (nothing to do)."""
    g = nx.Graph()
    g.add_edges_from(tuple(e) for e in boundary_edges)
    junction_vertices = [v for v, d in g.degree() if d > 2]
    if not junction_vertices:
        return None

    face_has_junction = np.isin(mesh.faces, junction_vertices).any(axis=1)
    result = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[~face_has_junction], process=False)
    result.remove_unreferenced_vertices()
    return result


def _remove_faces_touching_edges(mesh: trimesh.Trimesh, edges: np.ndarray) -> trimesh.Trimesh:
    """Last-resort fallback: delete the (small number of) faces touching
    edges that repeated fan-filling couldn't resolve, converting a
    non-manifold defect into a small hole instead — a real slicer's own
    auto-repair is far more likely to cleanly close a plain hole than to
    make sense of a non-manifold edge.

    Vectorized via integer-encoded edges (np.isin), not a Python-level
    `tuple(e) in edge_set` loop over every mesh edge — a real performance
    bug caught during development: for a mesh with hundreds of thousands of
    faces, `mesh.edges` has millions of entries, and a per-edge Python
    tuple/hash/dict-lookup loop over all of them (run on every repair
    iteration) took minutes and multiple GB of RAM instead of well under a
    second."""
    mesh_edges_sorted = np.sort(mesh.edges, axis=1)
    # Encode each edge's two vertex indices as one integer so membership
    # testing is a single vectorized np.isin call. `base` just needs to
    # exceed the largest vertex index so the two indices don't collide.
    base = int(mesh.vertices.shape[0]) + 1
    mesh_edge_codes = mesh_edges_sorted[:, 0].astype(np.int64) * base + mesh_edges_sorted[:, 1]
    bad_edge_codes = edges[:, 0].astype(np.int64) * base + edges[:, 1]

    bad_edge_mask = np.isin(mesh_edge_codes, bad_edge_codes)
    bad_faces = np.unique(mesh.edges_face[bad_edge_mask])
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[bad_faces] = False
    result = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep], process=False)
    result.remove_unreferenced_vertices()
    return result


def weld_and_repair(mesh: trimesh.Trimesh, max_iterations: int = 15) -> tuple[trimesh.Trimesh, bool, dict]:
    """Welds near-coincident vertices (as any STL consumer must) and closes
    any resulting boundary/non-manifold edges.

    Each iteration fixes non-manifold edges first (removing their faces —
    safer than trying to fill "through" an over-connected region) *then*
    fan-fills whatever boundary loops result in the same pass, since face
    removal typically converts a non-manifold edge into a plain hole that's
    trivial to close immediately rather than needing a whole extra
    iteration. Cheap per iteration (pure numpy/networkx, no PyMeshLab) —
    a full 500K+ face mesh's worth of iterations here cost single-digit
    seconds total, so a generous iteration budget is not expensive.

    Returns (repaired_mesh, is_clean, stats) where stats reports how many
    boundary/non-manifold edges were found and whether they were fully
    resolved within max_iterations.
    """
    welded = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    welded.merge_vertices(digits_vertex=8)

    boundary, nonmanifold = _boundary_and_nonmanifold_edges(welded)
    initial_boundary, initial_nonmanifold = len(boundary), len(nonmanifold)
    if initial_boundary == 0 and initial_nonmanifold == 0:
        return welded, True, {"initial_boundary": 0, "initial_nonmanifold": 0, "iterations": 0}

    current = welded
    prev_boundary_count = None
    for i in range(max_iterations):
        boundary, nonmanifold = _boundary_and_nonmanifold_edges(current)
        if len(boundary) == 0 and len(nonmanifold) == 0:
            return current, True, {
                "initial_boundary": initial_boundary,
                "initial_nonmanifold": initial_nonmanifold,
                "iterations": i,
            }
        if len(nonmanifold) > 0:
            current = _remove_faces_touching_edges(current, nonmanifold)
            current.merge_vertices(digits_vertex=8)
            boundary, _ = _boundary_and_nonmanifold_edges(current)
        if len(boundary) > 0:
            # A plain fan-fill pass alone can plateau forever on a junction
            # vertex shared by multiple loops (see _clear_junction_vertices'
            # docstring) — detect no progress and clear junctions first.
            if prev_boundary_count is not None and len(boundary) >= prev_boundary_count:
                cleared = _clear_junction_vertices(current, boundary)
                if cleared is not None:
                    current = cleared
                    current.merge_vertices(digits_vertex=8)
                    boundary, _ = _boundary_and_nonmanifold_edges(current)
            if len(boundary) > 0:
                current = _fan_fill_boundary_loops(current, boundary)
                current.merge_vertices(digits_vertex=8)
            prev_boundary_count = len(boundary)

    boundary, nonmanifold = _boundary_and_nonmanifold_edges(current)
    is_clean = len(boundary) == 0 and len(nonmanifold) == 0
    if not is_clean:
        # Bounded effort: rather than loop indefinitely on a pathological
        # case, do one last resort — delete the residual faces outright
        # (accepting a small hole over an unresolved defect) and report it
        # honestly (see STLExportStage) instead of hanging the pipeline.
        remaining = np.concatenate([e for e in (boundary, nonmanifold) if len(e)])
        current = _remove_faces_touching_edges(current, remaining)
        current.merge_vertices(digits_vertex=8)
        boundary, nonmanifold = _boundary_and_nonmanifold_edges(current)
        is_clean = len(boundary) == 0 and len(nonmanifold) == 0

    return current, is_clean, {
        "initial_boundary": initial_boundary,
        "initial_nonmanifold": initial_nonmanifold,
        "iterations": max_iterations,
        "final_boundary": len(boundary),
        "final_nonmanifold": len(nonmanifold),
    }

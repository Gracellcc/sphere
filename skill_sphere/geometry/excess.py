"""Spherical excess (area of spherical triangles) for combination quality assessment.

The spherical excess E = A + B + C - pi equals the area of a spherical triangle
on the unit sphere. Larger excess means the three skills cover more diverse
directions, indicating a higher-quality combination.

For high-dimensional vectors (where cross product is undefined), we project
the three vertices into their spanning 2D subspace first, then compute the
excess in the equivalent 3D geometry.
"""

import torch
from torch import Tensor

from .sphere import geodesic_distance


def _to_3d_subspace(a: Tensor, b: Tensor, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Project three high-dimensional unit vectors into a 3D subspace.

    Uses Gram-Schmidt to find an orthonormal basis for the subspace
    spanned by a, b, c, then expresses each vector in this basis.

    Args:
        a, b, c: (D,) unit vectors in high-dimensional space.

    Returns:
        Tuple of three (3,) unit vectors in the subspace.
    """
    # First basis vector: a itself
    e1 = a / a.norm().clamp(min=1e-10)

    # Second basis vector: orthogonal to e1, in plane of a and b
    b_proj = b - (b @ e1) * e1
    b_proj_norm = b_proj.norm()
    if b_proj_norm < 1e-10:
        # a and b are nearly parallel; pick arbitrary orthogonal direction
        e2 = torch.zeros_like(a)
        e2[0] = -a[1]
        e2[1] = a[0]
        e2 = e2 / e2.norm().clamp(min=1e-10)
    else:
        e2 = b_proj / b_proj_norm

    # Third basis vector: orthogonal to e1 and e2
    c_proj = c - (c @ e1) * e1 - (c @ e2) * e2
    c_proj_norm = c_proj.norm()
    if c_proj_norm < 1e-10:
        # All three vectors are coplanar; use any orthogonal direction
        # The excess will be ~0 in this case
        e3 = torch.zeros_like(a)
        for i in range(a.shape[0]):
            candidate = torch.zeros_like(a)
            candidate[i] = 1.0
            candidate = candidate - (candidate @ e1) * e1 - (candidate @ e2) * e2
            if candidate.norm() > 1e-8:
                e3 = candidate / candidate.norm()
                break
    else:
        e3 = c_proj / c_proj_norm

    # Project each original vector into the 3D basis
    basis = torch.stack([e1, e2, e3], dim=0)  # (3, D)
    a_3d = basis @ a  # (3,)
    b_3d = basis @ b
    c_3d = basis @ c

    # Re-normalize to unit sphere
    a_3d = a_3d / a_3d.norm().clamp(min=1e-10)
    b_3d = b_3d / b_3d.norm().clamp(min=1e-10)
    c_3d = c_3d / c_3d.norm().clamp(min=1e-10)

    return a_3d, b_3d, c_3d


def spherical_excess_3d(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Compute spherical excess using Eriksson's formula for 3D unit vectors.

    E = 2 * atan2((a x b) . c, 1 + a.b + b.c + c.a)

    Args:
        a, b, c: (3,) or (..., 3) unit vectors.

    Returns:
        Scalar or (...) tensor of spherical excess values.
    """
    # Triple product: (a x b) . c
    cross_ab = torch.cross(a, b, dim=-1)
    numerator = (cross_ab * c).sum(dim=-1)

    # Denominator: 1 + a.b + b.c + c.a
    denominator = (
        1.0
        + (a * b).sum(dim=-1)
        + (b * c).sum(dim=-1)
        + (c * a).sum(dim=-1)
    )

    return 2.0 * torch.atan2(numerator, denominator)


def spherical_excess(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Compute spherical excess for unit vectors of any dimension.

    For 3D vectors, uses Eriksson's formula directly.
    For higher dimensions, projects to the 3D subspace spanned by a, b, c.

    The spherical excess equals the area of the spherical triangle and
    measures the "diversity" of the three skill directions. Larger values
    indicate the skills cover more distinct directions.

    Range: [0, 2*pi] for a single hemisphere, but typically much smaller.

    Args:
        a, b, c: (D,) unit vectors (D >= 3).

    Returns:
        Scalar tensor of spherical excess.
    """
    dim = a.shape[-1]

    if dim == 3:
        return spherical_excess_3d(a, b, c).abs()

    # High-dimensional case: project to 3D subspace
    a_3d, b_3d, c_3d = _to_3d_subspace(a, b, c)
    return spherical_excess_3d(a_3d, b_3d, c_3d).abs()


def combination_diversity(vectors: Tensor) -> Tensor:
    """Measure the diversity of a set of skill vectors using spherical excess.

    For K >= 3 vectors, computes the average spherical excess over all
    (K choose 3) triangles. For K < 3, falls back to average pairwise distance.

    Args:
        vectors: (K, D) unit vectors.

    Returns:
        Scalar diversity score (higher = more diverse).
    """
    k = vectors.shape[0]

    if k < 2:
        return torch.tensor(0.0, device=vectors.device)

    if k == 2:
        return geodesic_distance(vectors[0], vectors[1])

    # Average spherical excess over all triples
    total_excess = torch.tensor(0.0, device=vectors.device)
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            for m in range(j + 1, k):
                total_excess = total_excess + spherical_excess(
                    vectors[i], vectors[j], vectors[m]
                )
                count += 1

    return total_excess / count if count > 0 else total_excess

"""Core spherical geometry operations on the unit hypersphere S^{d-1}.

All operations work with L2-normalized vectors (unit vectors) in arbitrary dimensions.
Numerical stability is handled for both fp32 and fp16/bf16.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

# Threshold for falling back from slerp to lerp when vectors are nearly parallel
_SLERP_DOT_THRESHOLD = 0.9995


def l2_normalize(x: Tensor, dim: int = -1, eps: float | None = None) -> Tensor:
    """Project vectors onto the unit hypersphere via L2 normalization.

    Args:
        x: Input tensor of any shape.
        dim: Dimension along which to normalize.
        eps: Small constant for numerical stability. Auto-selected based on dtype
             if not provided (1e-6 for half precision, 1e-12 for float32).

    Returns:
        Unit vectors on S^{d-1} with the same shape as input.
    """
    if eps is None:
        eps = 1e-6 if x.dtype in (torch.float16, torch.bfloat16) else 1e-12
    return F.normalize(x, p=2, dim=dim, eps=eps)


def cosine_similarity_matrix(a: Tensor, b: Tensor) -> Tensor:
    """Compute pairwise cosine similarity between two sets of unit vectors.

    For unit vectors, cosine similarity equals the dot product, which equals
    cos(theta) where theta is the geodesic angle between the two points.

    Args:
        a: (N, D) tensor of unit vectors.
        b: (M, D) tensor of unit vectors.

    Returns:
        (N, M) tensor of cosine similarities in [-1, 1].
    """
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    return a @ b.T


def geodesic_distance(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute geodesic (arc) distance between unit vectors on the sphere.

    d(a, b) = arccos(a . b)

    This is the true shortest-path distance along the sphere surface.

    Args:
        a: (..., D) unit vectors.
        b: (..., D) unit vectors (broadcastable with a).
        eps: Clamp margin to avoid arccos gradient explosion at +/-1.

    Returns:
        (...) tensor of geodesic distances in [0, pi].
    """
    dot = (a * b).sum(dim=-1)
    # Clamp to [-1, 1] for numerical safety, then arccos
    dot = dot.clamp(-1.0, 1.0)
    dist = torch.acos(dot)
    # For gradient computation, the caller can use cosine_distance() instead
    # which avoids the arccos gradient singularity at dot = +/-1.
    return dist


def cosine_distance(a: Tensor, b: Tensor) -> Tensor:
    """Compute cosine distance as a smooth proxy for geodesic distance.

    d_cos(a, b) = 1 - cos(theta) = 1 - a . b

    Monotonically related to geodesic distance for theta in [0, pi].
    Has well-behaved gradients everywhere (unlike arccos).

    Args:
        a: (..., D) unit vectors.
        b: (..., D) unit vectors.

    Returns:
        (...) tensor of cosine distances in [0, 2].
    """
    return 1.0 - (a * b).sum(dim=-1)


def slerp(v0: Tensor, v1: Tensor, t: float | Tensor) -> Tensor:
    """Spherical linear interpolation (Slerp) between unit vectors.

    Interpolates along the geodesic (great circle arc) between v0 and v1.
    Falls back to linear interpolation when vectors are nearly parallel
    or anti-parallel (where slerp is numerically unstable).

    Args:
        v0: (..., D) starting unit vectors.
        v1: (..., D) ending unit vectors.
        t: Interpolation parameter in [0, 1]. Can be a scalar or tensor
           broadcastable with v0/v1's batch dimensions.

    Returns:
        (..., D) interpolated unit vectors on the sphere.
    """
    v0 = l2_normalize(v0)
    v1 = l2_normalize(v1)

    dot = (v0 * v1).sum(dim=-1, keepdim=True)
    dot = dot.clamp(-1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Avoid division by zero when sin(theta) ≈ 0
    safe_sin = torch.where(
        sin_theta.abs() < 1e-8,
        torch.ones_like(sin_theta),
        sin_theta,
    )

    if isinstance(t, (int, float)):
        s0 = torch.sin((1.0 - t) * theta) / safe_sin
        s1 = torch.sin(t * theta) / safe_sin
        result_lerp = (1.0 - t) * v0 + t * v1
    else:
        if isinstance(t, Tensor) and t.dim() < v0.dim():
            t = t.unsqueeze(-1)
        s0 = torch.sin((1.0 - t) * theta) / safe_sin
        s1 = torch.sin(t * theta) / safe_sin
        result_lerp = (1.0 - t) * v0 + t * v1

    result_slerp = s0 * v0 + s1 * v1

    # Use lerp for nearly parallel vectors
    use_lerp = (dot > _SLERP_DOT_THRESHOLD)
    result = torch.where(use_lerp, result_lerp, result_slerp)

    # For nearly anti-parallel vectors (dot < -threshold), the lerp midpoint
    # is near zero. Handle by adding a small perturbation before normalizing.
    is_antipodal = (dot < -_SLERP_DOT_THRESHOLD)
    if is_antipodal.any():
        # Perturb to break the degeneracy
        perturb = torch.randn_like(v0) * 1e-4
        antipodal_result = l2_normalize((1.0 - 0.5) * v0 + 0.5 * v1 + perturb)
        result = torch.where(is_antipodal, antipodal_result, result)

    # Re-project to sphere to handle floating-point drift
    return l2_normalize(result)


def multi_slerp(vectors: Tensor, weights: Tensor) -> Tensor:
    """Weighted Slerp combination of multiple unit vectors.

    Iteratively combines vectors pairwise using Slerp, weighted by
    their relative importance. The order is sorted by descending weight
    to reduce accumulated interpolation error.

    Args:
        vectors: (K, D) tensor of K unit vectors to combine.
        weights: (K,) tensor of non-negative weights (will be normalized).

    Returns:
        (D,) combined unit vector on the sphere.
    """
    assert vectors.shape[0] == weights.shape[0]
    assert vectors.shape[0] >= 1

    if vectors.shape[0] == 1:
        return l2_normalize(vectors[0])

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Sort by descending weight for stability
    sorted_idx = torch.argsort(weights, descending=True)
    vectors = vectors[sorted_idx]
    weights = weights[sorted_idx]

    # Iterative pairwise slerp
    result = vectors[0]
    cumulative_weight = weights[0]

    for i in range(1, vectors.shape[0]):
        # t is the fraction of the new vector in the combination
        cumulative_weight = cumulative_weight + weights[i]
        t = weights[i] / cumulative_weight
        result = slerp(result.unsqueeze(0), vectors[i].unsqueeze(0), t).squeeze(0)

    return result


def pairwise_geodesic_distance(points: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute pairwise geodesic distance matrix for a set of sphere points.

    Args:
        points: (N, D) unit vectors.
        eps: Clamp margin for arccos stability.

    Returns:
        (N, N) symmetric distance matrix with zeros on diagonal.
    """
    sim = cosine_similarity_matrix(points, points)
    sim = sim.clamp(-1.0, 1.0)
    return torch.acos(sim)


def find_nearest_neighbors(
    query: Tensor, candidates: Tensor, k: int
) -> tuple[Tensor, Tensor]:
    """Find k nearest neighbors on the sphere by cosine similarity.

    Args:
        query: (D,) or (Q, D) query unit vector(s).
        candidates: (N, D) candidate unit vectors.
        k: Number of nearest neighbors.

    Returns:
        Tuple of (indices, similarities):
            indices: (k,) or (Q, k) indices into candidates.
            similarities: (k,) or (Q, k) cosine similarities.
    """
    single_query = query.dim() == 1
    if single_query:
        query = query.unsqueeze(0)

    # Ensure matching dtypes (embedding models may return bfloat16)
    if query.dtype != candidates.dtype:
        query = query.to(candidates.dtype)
    sims = query @ candidates.T  # (Q, N)
    k = min(k, candidates.shape[0])
    top_sims, top_idx = torch.topk(sims, k, dim=-1)

    if single_query:
        return top_idx.squeeze(0), top_sims.squeeze(0)
    return top_idx, top_sims


def find_antipodal_pairs(points: Tensor, threshold: float = -0.95) -> list[tuple[int, int]]:
    """Find pairs of nearly antipodal (opposite) skill vectors.

    Antipodal skills represent opposing strategies for the same problem.

    Args:
        points: (N, D) unit vectors.
        threshold: Cosine similarity threshold for antipodal detection.
                   Default -0.95 means angle > ~162 degrees.

    Returns:
        List of (i, j) index pairs where cos(angle) < threshold.
    """
    sim = cosine_similarity_matrix(points, points)
    pairs = []
    n = points.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] < threshold:
                pairs.append((i, j))
    return pairs


def find_redundant_pairs(points: Tensor, threshold: float = 0.95) -> list[tuple[int, int]]:
    """Find pairs of nearly identical (redundant) skill vectors.

    Args:
        points: (N, D) unit vectors.
        threshold: Cosine similarity threshold for redundancy detection.
                   Default 0.95 means angle < ~18 degrees.

    Returns:
        List of (i, j) index pairs where cos(angle) > threshold.
    """
    sim = cosine_similarity_matrix(points, points)
    pairs = []
    n = points.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > threshold:
                pairs.append((i, j))
    return pairs

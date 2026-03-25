"""Implicit Voronoi analysis on high-dimensional spheres.

Since scipy.spatial.SphericalVoronoi only supports 3D, we use an implicit
approach: the Voronoi region of a centroid is the set of all sphere points
closer to it than to any other centroid. We estimate region "areas" by
Monte Carlo sampling on the sphere.
"""

import torch
from torch import Tensor

from .sphere import l2_normalize


def _sample_sphere_uniform(n_samples: int, dim: int, device: torch.device) -> Tensor:
    """Sample points uniformly on the unit hypersphere S^{d-1}.

    Uses the standard method: sample from N(0, I) and normalize.

    Args:
        n_samples: Number of points to sample.
        dim: Embedding dimension d.
        device: Torch device.

    Returns:
        (n_samples, dim) unit vectors uniformly distributed on S^{d-1}.
    """
    points = torch.randn(n_samples, dim, device=device)
    return l2_normalize(points)


def voronoi_assignments(
    sample_points: Tensor, centroids: Tensor
) -> Tensor:
    """Assign each sample point to its nearest centroid (Voronoi cell).

    Args:
        sample_points: (N, D) unit vectors sampled on the sphere.
        centroids: (K, D) unit vectors representing skill positions.

    Returns:
        (N,) tensor of centroid indices, one per sample point.
    """
    # Cosine similarity = dot product for unit vectors
    # Ensure matching dtypes (embedding models may return bfloat16)
    if sample_points.dtype != centroids.dtype:
        sample_points = sample_points.to(centroids.dtype)
    sims = sample_points @ centroids.T  # (N, K)
    return sims.argmax(dim=-1)


def voronoi_areas(
    skill_vectors: Tensor,
    n_samples: int = 10000,
) -> Tensor:
    """Estimate Voronoi region sizes for each skill on the sphere.

    Larger area means the skill covers a wider region (fewer nearby neighbors),
    indicating a potential sparse zone that needs more skills.

    Args:
        skill_vectors: (K, D) unit vectors of skill positions.
        n_samples: Number of Monte Carlo samples on the sphere.

    Returns:
        (K,) tensor of relative Voronoi areas (sums to 1.0).
    """
    k, dim = skill_vectors.shape
    device = skill_vectors.device

    samples = _sample_sphere_uniform(n_samples, dim, device)
    assignments = voronoi_assignments(samples, skill_vectors)

    # Count how many samples fall in each cell
    counts = torch.zeros(k, device=device)
    for i in range(k):
        counts[i] = (assignments == i).sum().float()

    # Normalize to relative areas
    return counts / counts.sum()


def find_sparse_regions(
    skill_vectors: Tensor,
    n_samples: int = 10000,
    top_k: int = 5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Identify the most sparsely covered regions on the skill sphere.

    Returns the skills with the largest Voronoi regions (least neighbors)
    and representative points from those sparse regions.

    Args:
        skill_vectors: (K, D) unit vectors.
        n_samples: Monte Carlo samples for area estimation.
        top_k: Number of sparse regions to return.

    Returns:
        Tuple of (sparse_skill_indices, areas, representative_points):
            sparse_skill_indices: (top_k,) indices of skills with largest regions.
            areas: (top_k,) their Voronoi areas.
            representative_points: (top_k, D) centroid of each sparse region
                (average of samples assigned to that skill, re-normalized).
    """
    k, dim = skill_vectors.shape
    device = skill_vectors.device
    top_k = min(top_k, k)

    samples = _sample_sphere_uniform(n_samples, dim, device)
    assignments = voronoi_assignments(samples, skill_vectors)

    # Compute areas
    counts = torch.zeros(k, device=device)
    for i in range(k):
        counts[i] = (assignments == i).sum().float()
    areas = counts / counts.sum()

    # Find top-k sparse skills
    sparse_areas, sparse_indices = torch.topk(areas, top_k)

    # Compute representative points (centroids of sparse regions)
    rep_points = torch.zeros(top_k, dim, device=device)
    for rank, idx in enumerate(sparse_indices):
        mask = assignments == idx.item()
        if mask.sum() > 0:
            region_mean = samples[mask].mean(dim=0)
            rep_points[rank] = l2_normalize(region_mean.unsqueeze(0)).squeeze(0)
        else:
            rep_points[rank] = skill_vectors[idx.item()]

    return sparse_indices, sparse_areas, rep_points


def coverage_uniformity(skill_vectors: Tensor, n_samples: int = 10000) -> float:
    """Measure how uniformly skills cover the sphere.

    Returns the entropy of the Voronoi area distribution, normalized by
    the maximum possible entropy (uniform distribution). A value of 1.0
    means perfectly uniform coverage.

    Args:
        skill_vectors: (K, D) unit vectors.
        n_samples: Monte Carlo samples.

    Returns:
        Uniformity score in [0, 1].
    """
    areas = voronoi_areas(skill_vectors, n_samples)
    k = skill_vectors.shape[0]

    # Avoid log(0)
    areas = areas.clamp(min=1e-10)
    entropy = -(areas * areas.log()).sum()
    max_entropy = torch.log(torch.tensor(k, dtype=areas.dtype, device=areas.device))

    if max_entropy < 1e-10:
        return 1.0

    return (entropy / max_entropy).item()

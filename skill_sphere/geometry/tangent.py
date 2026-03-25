"""Tangent space operations on the unit hypersphere.

The tangent space T_x(S^{d-1}) at a point x on the sphere is the set of
all vectors orthogonal to x. Operations in the tangent space enable
gradient-based skill revision while staying on the sphere.
"""

import torch
from torch import Tensor

from .sphere import l2_normalize


def project_to_tangent(x: Tensor, v: Tensor) -> Tensor:
    """Project a vector v onto the tangent space at x on the sphere.

    Removes the component of v along x, keeping only the part orthogonal to x.

    Args:
        x: (..., D) base point(s) on the unit sphere.
        v: (..., D) vector(s) to project.

    Returns:
        (..., D) projected vector(s) in T_x(S^{d-1}).
    """
    # Component of v along x
    dot = (v * x).sum(dim=-1, keepdim=True)
    return v - dot * x


def exponential_map(x: Tensor, u: Tensor) -> Tensor:
    """Map a tangent vector to a point on the sphere (exponential map).

    exp_x(u) = cos(||u||) * x + sin(||u||) / ||u|| * u

    Walks along the geodesic (great circle) from x in direction u
    for arc-length ||u||.

    Args:
        x: (..., D) base point(s) on the unit sphere.
        u: (..., D) tangent vector(s) at x (must satisfy u . x = 0).

    Returns:
        (..., D) resulting point(s) on the unit sphere.
    """
    norm_u = u.norm(dim=-1, keepdim=True)

    # For very small tangent vectors, use retraction (first-order approx)
    eps = 1e-7 if x.dtype == torch.float32 else 1e-5
    small = norm_u < eps

    # Exponential map
    cos_nu = torch.cos(norm_u)
    # Safe division: sin(t)/t → 1 as t → 0
    sin_nu_over_nu = torch.where(
        small,
        torch.ones_like(norm_u),
        torch.sin(norm_u) / norm_u.clamp(min=eps),
    )
    result_exp = cos_nu * x + sin_nu_over_nu * u

    # Retraction fallback for small vectors: normalize(x + u)
    result_ret = l2_normalize(x + u)

    result = torch.where(small, result_ret, result_exp)
    return l2_normalize(result)


def logarithmic_map(x: Tensor, y: Tensor, eps: float = 1e-7) -> Tensor:
    """Map a sphere point back to a tangent vector (logarithmic map).

    log_x(y) is the tangent vector at x that points toward y, with
    magnitude equal to the geodesic distance between x and y.

    Args:
        x: (..., D) base point(s) on the unit sphere.
        y: (..., D) target point(s) on the unit sphere.
        eps: Numerical stability constant.

    Returns:
        (..., D) tangent vector(s) at x pointing toward y.
    """
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)

    # Direction from x toward y in the tangent space
    direction = y - dot * x
    direction_norm = direction.norm(dim=-1, keepdim=True).clamp(min=eps)
    direction = direction / direction_norm

    # When x ≈ y, the log map should return ~zero vector
    return theta * direction


def parallel_transport(x: Tensor, y: Tensor, v: Tensor) -> Tensor:
    """Parallel transport a tangent vector from T_x to T_y along the geodesic.

    Moves v from the tangent space at x to the tangent space at y while
    preserving its length and the angle with the geodesic.

    Args:
        x: (..., D) source point on the sphere.
        y: (..., D) target point on the sphere.
        v: (..., D) tangent vector at x to transport.

    Returns:
        (..., D) transported tangent vector at y.
    """
    log_xy = logarithmic_map(x, y)
    norm = log_xy.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    direction = log_xy / norm
    theta = norm

    # Component of v along the geodesic direction
    v_parallel = (v * direction).sum(dim=-1, keepdim=True) * direction
    v_perp = v - v_parallel

    # The parallel component rotates; the perpendicular component stays the same
    v_parallel_transported = (
        -torch.sin(theta) * (v * direction).sum(dim=-1, keepdim=True) * x
        + torch.cos(theta) * v_parallel
    )

    return v_parallel_transported + v_perp


def riemannian_gradient(x: Tensor, euclidean_grad: Tensor) -> Tensor:
    """Convert Euclidean gradient to Riemannian gradient on the sphere.

    Simply projects the Euclidean gradient onto the tangent space at x.

    Args:
        x: (..., D) point(s) on the unit sphere.
        euclidean_grad: (..., D) Euclidean gradient of the loss.

    Returns:
        (..., D) Riemannian gradient in T_x(S^{d-1}).
    """
    return project_to_tangent(x, euclidean_grad)


def gradient_step(
    x: Tensor, grad: Tensor, lr: float, use_expmap: bool = True
) -> Tensor:
    """Take a gradient descent step on the sphere.

    Args:
        x: (..., D) current point(s) on the unit sphere.
        grad: (..., D) Euclidean gradient (will be projected to tangent space).
        lr: Learning rate (step size).
        use_expmap: If True, use exponential map; otherwise use retraction.

    Returns:
        (..., D) updated point(s) on the unit sphere.
    """
    # Project gradient to tangent space
    rgrad = riemannian_gradient(x, grad)

    # Step in negative gradient direction
    tangent_step = -lr * rgrad

    if use_expmap:
        return exponential_map(x, tangent_step)
    else:
        # Retraction: simpler and cheaper
        return l2_normalize(x + tangent_step)

"""Strategy intent point tracker on the unit sphere.

Tracks the agent's policy intent point t as it evolves during an episode,
computing adaptive momentum, drift rate, and trajectory coherence for the
dynamic injection pipeline.

Implements MD Section 6.3: "t suddenly drifting = the agent hit difficulty."

Key sphere-unique properties:
- Drift rate is geodesic distance (bounded, calibratable against d_typical)
- Adaptive momentum via sigmoid mapping of drift vs reference
- Trajectory coherence (path efficiency) as sphere-geometric confidence proxy
- All quantities are sphere-relative, not absolute
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from skill_sphere.geometry.sphere import geodesic_distance


@dataclass
class DriftInfo:
    """Per-step drift information from IntentTracker."""
    alpha: float          # Adaptive momentum used for this step
    drift_rate: float     # Geodesic distance from previous t to current context
    drift_norm: float     # drift_rate / drift_typical (sphere-relative)
    recent_avg: float     # Average drift over last 3 steps
    coherence: float = 1.0    # Trajectory coherence η = displacement / path_length
    stability: float = 1.0    # Drift stability = 1 - smoothed(drift_norm)


class IntentTracker:
    """Track strategy intent point trajectory on sphere with drift detection.

    Provides two key signals:
    1. Adaptive momentum (alpha): how fast t should chase the new context.
       - High drift (agent confused) → high alpha → t adapts quickly
       - Low drift (agent stable) → low alpha → t stays stable
    2. Drift rate: geodesic velocity of t, calibrated against d_typical.
       Used downstream by DynamicInjector for the drift-confidence gamma.

    Parameters are calibrated from the sphere's own geometry (d_typical),
    not set manually per environment.
    """

    def __init__(
        self,
        alpha_base: float = 0.15,
        alpha_max: float = 0.6,
        sensitivity: float = 5.0,
    ):
        """
        Args:
            alpha_base: Minimum momentum (stable agent, low drift).
            alpha_max: Maximum momentum (confused agent, high drift).
            sensitivity: Steepness of the sigmoid transition between base and max.
        """
        self.alpha_base = alpha_base
        self.alpha_max = alpha_max
        self.sensitivity = sensitivity

        # State (reset per episode)
        self._prev_t: Tensor | None = None
        self._initial_t: Tensor | None = None  # Starting point for coherence
        self._drift_history: list[float] = []
        self._path_length: float = 0.0  # Cumulative geodesic path length

        # Calibration (set once from sphere geometry)
        self._drift_typical: float | None = None

    def calibrate(self, d_typical: float) -> None:
        """Calibrate drift scale from the sphere's characteristic distance.

        Context vectors (task + observation) change more slowly than
        random skill-skill distances, so we use half d_typical as the
        reference drift rate.

        Args:
            d_typical: Median inter-skill geodesic distance from SphereRetriever.
        """
        self._drift_typical = d_typical * 0.5

    def reset(self, initial_t: Tensor) -> None:
        """Reset tracker for a new episode.

        Args:
            initial_t: (D,) initial intent point (encoded task description).
        """
        self._prev_t = initial_t.detach().clone()
        self._initial_t = initial_t.detach().clone()
        self._drift_history = []
        self._path_length = 0.0

    def update(self, t_new: Tensor) -> DriftInfo:
        """Record new intent point position, compute adaptive momentum.

        Should be called AFTER computing the new context vector but BEFORE
        applying slerp. The caller uses the returned alpha for slerp:
            alpha, drift = tracker.update(context_vec)
            t = slerp(t, context_vec, alpha)

        Args:
            t_new: (D,) new context vector (encoded task + observation).

        Returns:
            DriftInfo with adaptive momentum and drift metrics.
        """
        if self._prev_t is None:
            self._prev_t = t_new.detach().clone()
            self._initial_t = t_new.detach().clone()
            return DriftInfo(
                alpha=self.alpha_base,
                drift_rate=0.0,
                drift_norm=0.0,
                recent_avg=0.0,
                coherence=1.0,
                stability=1.0,
            )

        # Geodesic distance between previous and new positions
        drift = geodesic_distance(self._prev_t, t_new).item()
        self._drift_history.append(drift)
        self._path_length += drift
        self._prev_t = t_new.detach().clone()

        # Reference drift rate: calibrated from sphere, or rolling median
        ref = self._drift_typical if self._drift_typical is not None else self._rolling_median()

        # Sigmoid mapping: centered at reference, steepness = sensitivity
        z = self.sensitivity * (drift - ref)
        z = max(-20.0, min(20.0, z))  # Clamp for numerical stability
        sigmoid_z = 1.0 / (1.0 + math.exp(-z))
        alpha = self.alpha_base + (self.alpha_max - self.alpha_base) * sigmoid_z

        # Drift normalized by reference
        drift_norm = drift / ref if ref > 1e-8 else 0.0

        # Trajectory coherence: displacement / path_length
        # η ≈ 1 → straight path (agent purposeful) → high confidence
        # η ≈ 0 → wandering (agent lost) → low confidence
        displacement = geodesic_distance(self._initial_t, t_new).item()
        coherence = displacement / self._path_length if self._path_length > 1e-8 else 1.0

        # Drift stability via sigmoid centered at drift_norm = 1.0
        # sigmoid(-k*(drift_norm - 1)): drift_norm < 1 → stable, > 1 → unstable
        recent_drift_norm = self._recent_avg() / ref if ref > 1e-8 else 0.0
        _stab_z = -4.0 * (recent_drift_norm - 1.0)
        _stab_z = max(-20.0, min(20.0, _stab_z))
        stability = 1.0 / (1.0 + math.exp(-_stab_z))

        return DriftInfo(
            alpha=alpha,
            drift_rate=drift,
            drift_norm=drift_norm,
            recent_avg=self._recent_avg(),
            coherence=coherence,
            stability=stability,
        )

    @property
    def drift_typical(self) -> float | None:
        """Calibrated typical drift rate (d_typical / 2)."""
        return self._drift_typical

    @property
    def path_length(self) -> float:
        """Cumulative geodesic path length of intent trajectory."""
        return self._path_length

    def _rolling_median(self) -> float:
        """Compute rolling median of observed drifts as fallback reference."""
        if not self._drift_history:
            return 0.1  # Safe default
        sorted_drifts = sorted(self._drift_history)
        return sorted_drifts[len(sorted_drifts) // 2]

    def _recent_avg(self) -> float:
        """Average drift over last 3 steps."""
        if not self._drift_history:
            return 0.0
        recent = self._drift_history[-3:]
        return sum(recent) / len(recent)

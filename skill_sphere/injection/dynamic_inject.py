"""Dynamic skill injection controller — Unified Sphere Inference Pipeline.

Implements the MD's injection mechanism with sphere-relative normalization:
  injection_weight = γ × (1 - confidence) × exp(-d_norm² / σ²)

where d_norm = d / d_typical is the geodesic distance normalized by the
sphere's characteristic scale (median inter-skill distance).

Sphere-unique features (not possible with flat retrieval):
- γ factor: boost-only (≥1.0) when agent is stuck (MD 6.2 + 6.3)
- Isolation: d_nn / d_typical detects uncharted territory (MD 4.5)
- Combined vector re-ranking: skills ordered by distance to Slerp center (MD 7.①)
- Skill rotation: penalize recently-injected skills to force exploration (MD 4.5)

This normalization is critical for generality:
- In a clustered sphere (WebShop, d_typical ≈ 0.4 rad), a skill at d=0.3
  has d_norm=0.75 → moderate weight. Without normalization, d=0.3 would
  give very high weight (exp(-0.36/0.25) = 0.24).
- In a spread sphere (ALFWorld, d_typical ≈ 0.8 rad), a skill at d=0.3
  has d_norm=0.375 → high weight. Same raw distance, different meaning.

The σ parameter now controls selectivity relative to the sphere's scale,
not in absolute radians. σ=0.5 means "inject significantly only when the
skill is within half the typical inter-skill distance."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from skill_sphere.geometry.sphere import (
    geodesic_distance,
    multi_slerp,
    slerp,
    l2_normalize,
)
from skill_sphere.skill_bank.retrieval import RetrievalResult
from skill_sphere.skill_bank.retrieval import SphereRetriever, RetrievalResult


@dataclass
class InjectionResult:
    """Result of the dynamic injection decision."""
    should_inject: bool
    selected_indices: list[int] = field(default_factory=list)
    injection_weights: list[float] = field(default_factory=list)
    combined_vector: Tensor | None = None
    total_injection_strength: float = 0.0
    confidence: float = 0.0
    # Sphere-geometric signals
    alignment: float = 1.0          # cos(d(t, q̂)) — intent-combination alignment (diagnostic)
    isolation_score: float = 0.0    # d_nn / d_typical — coverage proximity
    in_uncharted: bool = False      # isolation > threshold → uncharted territory
    gamma: float = 1.0              # drift-confidence interaction factor (≥1.0, boost only)
    regime: str = "neutral"         # drift-confidence regime label


class DynamicInjector:
    """Confidence-guided dynamic skill injection controller.

    Follows the MD specification with sphere-relative normalization:
    - injection_weight_i = γ × (1 - confidence) × exp(-d_norm_i² / σ²)
    - d_norm_i = d_i / d_typical (sphere-relative distance)
    - Uses SphereRetriever for complementarity-aware skill selection
    - Combines selected skills via multi-Slerp with computed weights

    Sphere-unique enhancements:
    - Boost-only γ factor: only increases injection when stuck (MD 6.2 + 6.3)
    - Uncharted territory detection via isolation score (MD 4.5)
    - Skills re-ranked by distance to Slerp center (MD 7.①)
    - Skill rotation: penalize recently-injected skills in retrieval
    """

    UNCHARTED_THRESHOLD = 1.5  # isolation_score above this → uncharted territory
    ROTATION_PENALTY = 0.15    # similarity penalty for recently-used skills
    # Phase 2 (cross-env complementary bridge) parameters
    BRIDGE_COMPLEMENTARITY_WEIGHT = 0.7   # weight for complementarity vs relevance in Phase 2
    BRIDGE_MIN_RELEVANCE = 0.3            # minimum cosine sim to query for bridge candidates
    BRIDGE_MIN_COMPLEMENTARITY = 0.5      # minimum avg sin(θ) with Phase 1 skills

    def __init__(
        self,
        sigma: float = 1.5,
        min_weight: float = 0.05,
        min_inject_strength: float = 0.15,
        max_skills: int = 5,
        relevance_k: int = 10,
        redundancy_threshold: float = 0.85,
        enable_bridge: bool = False,
        bridge_k: int = 1,
    ):
        """
        Args:
            sigma: Width of the distance decay Gaussian, in units of d_typical.
                   σ=1.5 means skills at 1.5× the typical inter-skill distance receive
                   moderate weight (exp(-1) ≈ 0.37), while skills much farther
                   receive near-zero weight.
            min_weight: Minimum injection weight to include a skill.
            min_inject_strength: Floor for base_strength = max(min_inject_strength, 1-conf).
                   Prevents high confidence from completely suppressing injection.
            max_skills: Maximum number of skills to inject.
            relevance_k: Number of candidate skills to consider.
            redundancy_threshold: Skills with cos > this are considered redundant.
        """
        self.sigma = sigma
        self.min_weight = min_weight
        self.min_inject_strength = min_inject_strength
        self.max_skills = max_skills
        self.enable_bridge = enable_bridge
        self.bridge_k = bridge_k
        self.retriever = SphereRetriever(
            relevance_k=relevance_k,
            final_k=max_skills,
            redundancy_threshold=redundancy_threshold,
        )

    def calibrate(self, skill_vectors: Tensor) -> dict:
        """Calibrate both retriever and injector to the sphere's geometry.

        Must be called before decide(), or will be called lazily.

        Returns:
            Dict with calibration statistics.
        """
        return self.retriever.calibrate(skill_vectors)

    def compute_gamma(
        self,
        confidence: float,
        drift_rate: float,
        drift_typical: float,
    ) -> tuple[float, str]:
        """Compute the drift-confidence interaction factor (MD 6.2 + 6.3).

        Boost-only design: γ ≥ 1.0 always. Never suppress injection.
        For an untrained model whose confidence is uncalibrated, suppression
        systematically hurts. Boosting when stuck is safe and helpful.

        | Regime      | Confidence | Drift  | γ   | Rationale                        |
        |-------------|------------|--------|-----|----------------------------------|
        | stuck       | low        | low    | 1.5 | Agent stuck, needs strong help   |
        | confused    | low        | high   | 1.3 | Agent flailing, boost help       |
        | exploring   | high       | high   | 1.0 | Agent exploring, don't interfere |
        | on_track    | high       | low    | 1.0 | Agent executing plan smoothly    |
        | neutral     | mid        | mid    | 1.0 | Default                          |

        Args:
            confidence: Agent's current confidence in [0, 1].
            drift_rate: Geodesic distance between consecutive intent points.
            drift_typical: Reference drift rate (calibrated from sphere).

        Returns:
            (gamma, regime_name) where gamma modulates injection strength.
        """
        if drift_typical < 1e-8:
            return 1.0, "neutral"

        low_conf = confidence < 0.4
        high_drift = drift_rate > drift_typical
        low_drift = drift_rate < drift_typical * 0.5

        if low_conf and low_drift:
            return 1.5, "stuck"
        elif low_conf and high_drift:
            return 1.3, "confused"
        else:
            return 1.0, "neutral"

    def compute_isolation(self, intent_point: Tensor, skill_vectors: Tensor) -> float:
        """Compute isolation score for intent point (d_nn / d_typical).

        Lightweight method that can be called before decide() to provide
        the coverage signal for Sphere-Geometric Confidence (SGC).

        Returns:
            isolation_score: ratio of nearest-neighbor distance to d_typical.
                > UNCHARTED_THRESHOLD means uncharted territory.
        """
        d_typical = max(self.retriever.d_typical, 1e-6)
        nn_sim = (intent_point @ skill_vectors.T).max().clamp(-1, 1)
        d_nn = torch.acos(nn_sim).item()
        return d_nn / d_typical

    def decide(
        self,
        intent_point: Tensor,
        confidence: float,
        skill_vectors: Tensor,
        force_inject: bool = False,
        drift_rate: float = 0.0,
        recently_used: list[int] | None = None,
    ) -> InjectionResult:
        """Decide whether and what to inject.

        Args:
            intent_point: (D,) current policy intent point on sphere.
            confidence: Agent's current confidence in [0, 1].
            skill_vectors: (N, D) all skill vectors in the sphere.
            force_inject: Override confidence check (e.g., when in a loop).
            drift_rate: Geodesic drift rate from IntentTracker (0.0 if unavailable).
            recently_used: Skill indices injected in recent steps (for rotation).

        Returns:
            InjectionResult with injection decision and sphere-geometric signals.
        """
        # --- Coverage-Aware: compute isolation score (MD 4.5) ---
        d_typical = max(self.retriever.d_typical, 1e-6)
        nn_sim = (intent_point @ skill_vectors.T).max().clamp(-1, 1)
        d_nn = torch.acos(nn_sim).item()
        isolation_score = d_nn / d_typical
        in_uncharted = isolation_score > self.UNCHARTED_THRESHOLD

        # --- Boost-only γ factor (MD 6.2 + 6.3) ---
        drift_ref = d_typical * 0.5
        gamma, regime = self.compute_gamma(confidence, drift_rate, drift_ref)

        # Step 1: Retrieve complementary skills using SphereRetriever
        # Apply skill rotation: penalize recently-used skills in similarity
        if recently_used:
            # Temporarily reduce similarity for recently-used skills
            sims = intent_point @ skill_vectors.T  # (N,)
            sims_adj = sims.clone()
            for idx in recently_used:
                if 0 <= idx < sims_adj.shape[0]:
                    sims_adj[idx] -= self.ROTATION_PENALTY
            retrieval = self.retriever.retrieve(
                query=intent_point,
                skill_vectors=skill_vectors,
                precomputed_sims=sims_adj,
            )
        else:
            retrieval = self.retriever.retrieve(
                query=intent_point,
                skill_vectors=skill_vectors,
            )

        if not retrieval.indices:
            if force_inject:
                # Force mode: bypass adaptive threshold, use top-K by raw similarity
                sims = intent_point @ skill_vectors.T
                k = min(self.retriever.final_k, skill_vectors.shape[0])
                topk_vals, topk_idx = torch.topk(sims, k)
                retrieval = RetrievalResult(
                    indices=topk_idx.tolist(),
                    scores=topk_vals.tolist(),
                    complementarity_scores=[0.0] * k,
                )
            else:
                return InjectionResult(
                    should_inject=False,
                    confidence=confidence,
                    isolation_score=isolation_score,
                    in_uncharted=in_uncharted,
                    gamma=gamma,
                    regime=regime,
                )

        # Phase 2: Cross-env complementary bridge selection
        # Search the FULL sphere for skills that are complementary to Phase 1
        # selections, even if they're beyond the normal distance threshold.
        # This enables cross-env skill transfer in unified spheres.
        selected_indices = retrieval.indices

        if self.enable_bridge and len(selected_indices) > 0:
            bridge_indices = self._select_bridge_skills(
                intent_point, skill_vectors, selected_indices,
            )
            # Append bridge skills (avoid duplicates)
            existing = set(selected_indices)
            for bi in bridge_indices:
                if bi not in existing:
                    selected_indices.append(bi)
                    existing.add(bi)

        # Step 2: Compute injection weights with sphere-relative distances
        # w_i = γ × base_strength × coverage_factor × exp(-d_norm_i² / σ²)
        selected_vectors = skill_vectors[selected_indices]

        # Compute geodesic distances from intent point to each selected skill
        intent_expanded = intent_point.unsqueeze(0).expand_as(selected_vectors)
        distances = geodesic_distance(intent_expanded, selected_vectors)  # (K,)

        # Coverage factor: sigmoid gate based on isolation score
        # High isolation → retrieval unreliable → suppress injection weights
        # sigmoid(-4 * (iso - 1.0)): iso=0.5→0.88, iso=1.0→0.50, iso=1.5→0.12
        _cov_z = -4.0 * (isolation_score - 1.0)
        _cov_z = max(-20.0, min(20.0, _cov_z))
        coverage_factor = 1.0 / (1.0 + math.exp(-_cov_z))

        # base_strength: force mode overrides to 1.0, otherwise confidence-based
        if force_inject:
            base_strength = 1.0
            coverage_factor = 1.0  # Don't gate when forced
        else:
            base_strength = max(self.min_inject_strength, 1.0 - confidence)
        weights = []
        for d in distances:
            d_norm = d.item() / d_typical
            w = gamma * base_strength * coverage_factor * math.exp(-(d_norm ** 2) / (self.sigma ** 2))
            weights.append(w)

        # Filter by minimum weight (skip when force injecting)
        filtered_indices = []
        filtered_weights = []
        for idx, w in zip(selected_indices, weights):
            if w >= self.min_weight or force_inject:
                filtered_indices.append(idx)
                filtered_weights.append(w)

        if not filtered_indices:
            return InjectionResult(
                should_inject=False,
                confidence=confidence,
                total_injection_strength=0.0,
                isolation_score=isolation_score,
                in_uncharted=in_uncharted,
                gamma=gamma,
                regime=regime,
            )

        # Step 3: Combine via multi-Slerp (MD 4.4)
        filtered_vectors = skill_vectors[filtered_indices]
        weight_tensor = torch.tensor(filtered_weights, device=skill_vectors.device)
        combined = multi_slerp(filtered_vectors, weight_tensor)

        # Compute alignment as diagnostic (not used for modulation)
        alignment = (intent_point @ combined).clamp(-1, 1).item()
        total_strength = sum(filtered_weights) / len(filtered_weights)

        return InjectionResult(
            should_inject=True,
            selected_indices=filtered_indices,
            injection_weights=filtered_weights,
            combined_vector=combined,
            total_injection_strength=total_strength,
            confidence=confidence,
            alignment=alignment,
            isolation_score=isolation_score,
            in_uncharted=in_uncharted,
            gamma=gamma,
            regime=regime,
        )

    def set_bridge_indices(self, bridge_indices: list[int]):
        """Set which skill indices are bridge skills (cross-env).

        Only these skills will be considered in Phase 2 bridge selection.
        Call this after building the sphere with bridge skills loaded.
        """
        self._bridge_indices = set(bridge_indices)

    def _select_bridge_skills(
        self,
        intent_point: Tensor,
        skill_vectors: Tensor,
        phase1_indices: list[int],
    ) -> list[int]:
        """Phase 2: Select complementary bridge skills.

        Searches ONLY bridge skills (set via set_bridge_indices) for skills
        that complement Phase 1 selections. Bridge skills are designed to
        be positioned between environment clusters on the sphere, containing
        vocabulary from multiple environments.

        This is sphere-specific: cosine top-k cannot do this because it
        has no complementarity mechanism — it always selects the closest
        skills, which are same-environment.
        """
        if not hasattr(self, '_bridge_indices') or not self._bridge_indices:
            return []
        if not phase1_indices:
            return []

        # Compute similarity of bridge skills to the query
        all_sims = intent_point @ skill_vectors.T  # (N,)

        # Phase 1 skill vectors for complementarity computation
        p1_vecs = skill_vectors[phase1_indices]  # (K1, D)
        p1_set = set(phase1_indices)

        best_bridges = []  # (score, index)

        for i in self._bridge_indices:
            if i in p1_set:
                continue

            relevance = all_sims[i].item()
            if relevance < self.BRIDGE_MIN_RELEVANCE:
                continue

            # Complementarity: average sin(θ) with all Phase 1 skills
            cos_with_p1 = (skill_vectors[i] @ p1_vecs.T)  # (K1,)
            sin_with_p1 = (1.0 - cos_with_p1 ** 2).clamp(min=0).sqrt()
            avg_complementarity = sin_with_p1.mean().item()

            if avg_complementarity < self.BRIDGE_MIN_COMPLEMENTARITY:
                continue

            # Combined score: heavily weight complementarity
            α = self.BRIDGE_COMPLEMENTARITY_WEIGHT
            score = (1 - α) * relevance + α * avg_complementarity

            best_bridges.append((score, i))

        # Select top bridge_k
        best_bridges.sort(key=lambda x: -x[0])
        return [idx for _, idx in best_bridges[:self.bridge_k]]

    def format_injected_skills(
        self,
        result: InjectionResult,
        skills: list,
        skill_vectors: Tensor | None = None,
        compact: bool = True,
    ) -> str:
        """Format selected skills as text for prompt injection.

        Skills are re-ranked by geodesic distance to the Slerp combined vector
        (Fréchet mean). The most central skill appears first. All skills get
        full principle text — no truncation.

        Args:
            result: InjectionResult from decide().
            skills: List of Skill objects (indexed by result.selected_indices).
            skill_vectors: (N, D) all skill vectors (needed for re-ranking).
            compact: Unused, kept for API compatibility.

        Returns:
            Formatted skill text for injection into the prompt.
        """
        if not result.should_inject or not result.selected_indices:
            return ""

        # Re-rank by distance to combined vector (closest = most central first)
        if result.combined_vector is not None and skill_vectors is not None and len(result.selected_indices) > 1:
            dists_to_center = []
            for idx in result.selected_indices:
                d = geodesic_distance(result.combined_vector, skill_vectors[idx]).item()
                dists_to_center.append(d)
            order = sorted(range(len(result.selected_indices)), key=lambda i: dists_to_center[i])
        else:
            order = list(range(len(result.selected_indices)))

        # Full text for ALL skills — no compact truncation
        lines = ["### Retrieved Skills"]
        for rank, i in enumerate(order):
            idx = result.selected_indices[i]
            skill = skills[idx]
            name = skill.name or "Tip"
            principle = skill.principle or skill.text
            lines.append(f"- **{name}**: {principle}")

        return "\n".join(lines)

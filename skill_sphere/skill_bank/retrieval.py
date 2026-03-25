"""Sphere-based skill retrieval with complementarity-aware selection.

Instead of just finding the closest skills (like standard embedding retrieval),
this module also considers:
1. Relevance: cosine similarity (cos θ) between skill and query
2. Complementarity: cross product magnitude (sin θ) between selected skills
3. Diversity: spherical excess of the selected combination

Key design: The retriever calibrates itself to the sphere's own geometry.
Instead of a fixed similarity threshold (which is environment-specific),
it uses the intra-sphere similarity distribution as the baseline.
A skill is considered relevant only if its similarity to the query is
above the sphere's own characteristic similarity level (p75 of pairwise sims).

This is general because:
- Dense/clustered spheres → high baseline → harder to inject (conservative)
- Diverse/spread spheres → low baseline → easier to inject (expressive)
- No environment-specific parameters needed
"""

from __future__ import annotations

import torch
from torch import Tensor

from skill_sphere.geometry.sphere import (
    geodesic_distance,
    cosine_similarity_matrix,
    find_nearest_neighbors,
)
from skill_sphere.geometry.excess import spherical_excess, combination_diversity


class SphereRetriever:
    """Retrieves skills from the sphere with complementarity-aware selection.

    Uses sphere-adaptive thresholding: the similarity threshold is derived
    from the sphere's own intra-skill similarity distribution, not set manually.
    """

    def __init__(
        self,
        relevance_k: int = 10,
        final_k: int = 3,
        redundancy_threshold: float = 0.9,
    ):
        """
        Args:
            relevance_k: Number of top-relevant candidates to consider.
            final_k: Maximum number of skills to select.
            redundancy_threshold: Pairs above this similarity are considered redundant.
        """
        self.relevance_k = relevance_k
        self.final_k = final_k
        self.redundancy_threshold = redundancy_threshold

        # Sphere calibration (computed lazily on first retrieve)
        self._calibrated = False
        self._adaptive_threshold = 0.0  # will be set by calibrate()
        self._d_typical = 1.0  # median geodesic distance between skills
        self._min_excess = 0.0  # minimum spherical excess for quality gate

    def calibrate(self, skill_vectors: Tensor) -> dict:
        """Compute sphere-adaptive thresholds from the skill distribution.

        Uses the intra-sphere pairwise similarity distribution as baseline.
        The adaptive threshold is set to p75 of pairwise similarities:
        a skill must be more similar to the query than 75% of inter-skill pairs.

        Also computes d_typical (median geodesic distance) for distance
        normalization in the injection formula.

        Args:
            skill_vectors: (N, D) all skill vectors in the sphere.

        Returns:
            Dict with calibration statistics.
        """
        n = skill_vectors.shape[0]
        if n < 2:
            self._adaptive_threshold = 0.0
            self._d_typical = 1.0
            self._calibrated = True
            return {"n_skills": n, "threshold": 0.0, "d_typical": 1.0}

        # Compute pairwise cosine similarities (upper triangle, no self-pairs)
        sim_matrix = skill_vectors @ skill_vectors.T
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pairwise_sims = sim_matrix[mask]

        # Percentile statistics
        sorted_sims, _ = pairwise_sims.sort()
        n_pairs = len(sorted_sims)
        p25 = sorted_sims[int(n_pairs * 0.25)].item()
        p50 = sorted_sims[int(n_pairs * 0.50)].item()
        p75 = sorted_sims[int(n_pairs * 0.75)].item()
        p90 = sorted_sims[int(n_pairs * 0.90)].item()
        mean = pairwise_sims.mean().item()

        # Adaptive threshold: p50 (median) of inter-skill similarity.
        # Rationale: queries and skills are from different semantic domains
        # (tasks vs principles), so query-skill similarity is systematically
        # lower than skill-skill similarity. Using p50 (not p75) as threshold
        # means: "the best skill must be at least as similar to the query
        # as the typical skill is to another skill."
        # This lets the injection weight formula do the fine-grained filtering.
        self._adaptive_threshold = p50

        # Typical geodesic distance (median) for distance normalization
        pairwise_d = torch.acos(pairwise_sims.clamp(-1.0, 1.0))
        self._d_typical = pairwise_d.median().item()

        # Compute reference spherical excess from sampled triples (MD 4.7)
        # Used as quality gate: combinations with excess below p25 are
        # near-degenerate (collinear) even if pairwise sin θ is acceptable.
        self._min_excess = 0.0
        if n >= 3:
            import itertools
            import random as _random

            all_triples = list(itertools.combinations(range(n), 3))
            sampled = _random.sample(all_triples, min(100, len(all_triples)))
            excesses = []
            for i, j, m in sampled:
                e = spherical_excess(
                    skill_vectors[i], skill_vectors[j], skill_vectors[m]
                ).item()
                excesses.append(abs(e))
            excesses.sort()
            self._min_excess = excesses[len(excesses) // 4]  # p25

        self._calibrated = True

        stats = {
            "n_skills": n,
            "pairwise_sim": {"p25": p25, "p50": p50, "p75": p75, "p90": p90, "mean": mean},
            "adaptive_threshold": self._adaptive_threshold,
            "d_typical": self._d_typical,
            "min_excess": self._min_excess,
        }
        return stats

    @property
    def d_typical(self) -> float:
        """Median geodesic distance between skills. Used by DynamicInjector."""
        return self._d_typical

    def retrieve(
        self,
        query: Tensor,
        skill_vectors: Tensor,
        general_indices: list[int] | None = None,
        precomputed_sims: Tensor | None = None,
    ) -> RetrievalResult:
        """Retrieve complementary skills for a query.

        Step 1: Calibrate threshold from sphere geometry (once).
        Step 2: Find top-K most relevant skills by cosine similarity.
        Step 3: Filter by adaptive threshold derived from sphere statistics.
        Step 4: Greedily select a complementary subset that maximizes diversity.

        If no skill passes the adaptive threshold, returns an EMPTY result.
        This is intentional: injecting irrelevant skills is worse than
        injecting nothing.

        Args:
            query: (D,) query unit vector (encoded task description).
            skill_vectors: (N, D) all skill vectors in the sphere.
            general_indices: Indices of general skills (always included if relevant).
            precomputed_sims: (N,) precomputed similarity scores (for skill rotation).
                If provided, used instead of query @ skill_vectors.T for ranking.

        Returns:
            RetrievalResult with selected skill indices, scores, and metadata.
        """
        n_skills = skill_vectors.shape[0]
        if n_skills == 0:
            return RetrievalResult(indices=[], scores=[], complementarity_scores=[])

        # Lazy calibration on first call
        if not self._calibrated:
            self.calibrate(skill_vectors)

        # Step 1: Find top-K relevant candidates
        k = min(self.relevance_k, n_skills)
        if precomputed_sims is not None:
            # Use adjusted similarities (e.g., with rotation penalty)
            topk_vals, topk_raw_idx = torch.topk(precomputed_sims, k)
            top_indices = topk_raw_idx
            # Use ORIGINAL similarities for threshold comparison (rotation only affects ranking)
            original_sims = query @ skill_vectors.T
            top_sims = original_sims[top_indices]
        else:
            top_indices, top_sims = find_nearest_neighbors(query, skill_vectors, k)

        # Step 2: Filter by sphere-adaptive threshold
        mask = top_sims >= self._adaptive_threshold
        if mask.sum() == 0:
            # No skill is more relevant than the sphere's baseline.
            # Return empty — do not force-inject irrelevant skills.
            return RetrievalResult(indices=[], scores=[], complementarity_scores=[])

        candidate_indices = top_indices[mask].tolist()
        candidate_sims = top_sims[mask].tolist()
        candidate_vectors = skill_vectors[candidate_indices]

        # Step 3: Greedy complementary selection
        selected = self._greedy_complementary_select(
            candidate_indices,
            candidate_sims,
            candidate_vectors,
            query,
        )

        return selected

    def _greedy_complementary_select(
        self,
        candidate_indices: list[int],
        candidate_sims: list[float],
        candidate_vectors: Tensor,
        query: Tensor,
    ) -> RetrievalResult:
        """Greedy selection maximizing relevance + complementarity.

        Start with the most relevant skill, then iteratively add the skill
        that best complements the current selection (highest sin θ with
        existing selections, while maintaining relevance).

        Args:
            candidate_indices: Original indices of candidate skills.
            candidate_sims: Cosine similarities of candidates with query.
            candidate_vectors: (K, D) vectors of candidates.
            query: (D,) query vector.

        Returns:
            RetrievalResult with the selected complementary subset.
        """
        k = len(candidate_indices)
        if k <= self.final_k:
            # Not enough candidates to select from
            return RetrievalResult(
                indices=candidate_indices,
                scores=candidate_sims,
                complementarity_scores=[0.0] * k,
            )

        # Pairwise similarity among candidates
        pair_sim = cosine_similarity_matrix(candidate_vectors, candidate_vectors)

        # Start with the most relevant candidate
        selected_local = [0]  # Local index within candidates
        remaining = list(range(1, k))

        while len(selected_local) < self.final_k and remaining:
            best_score = -float("inf")
            best_idx = -1

            for r in remaining:
                # Complementarity: average sin(θ) with already-selected skills
                complementarity = 0.0
                for s in selected_local:
                    cos_val = pair_sim[r, s].item()
                    # sin(θ) = sqrt(1 - cos²(θ)), clamped for safety
                    sin_val = (1.0 - cos_val ** 2) ** 0.5
                    complementarity += sin_val
                complementarity /= len(selected_local)

                # Skip if too similar to any selected skill
                max_sim_with_selected = max(
                    pair_sim[r, s].item() for s in selected_local
                )
                if max_sim_with_selected > self.redundancy_threshold:
                    continue

                # Combined score: relevance * complementarity
                # Both are in [0, 1] so their product balances the two
                relevance = candidate_sims[r]
                score = relevance * (0.5 + 0.5 * complementarity)

                if score > best_score:
                    best_score = score
                    best_idx = r

            if best_idx < 0:
                break

            selected_local.append(best_idx)
            remaining.remove(best_idx)

        # Spherical excess quality gate (MD 4.7): detect near-degenerate
        # combinations that pairwise sin θ cannot catch. Three skills that are
        # almost collinear have low spherical excess even if each pair looks
        # complementary. This is a sphere-unique capability.
        if len(selected_local) >= 3 and self._min_excess > 0:
            selected_vecs_check = candidate_vectors[selected_local]
            excess = combination_diversity(selected_vecs_check).item()
            # If below p25 threshold, drop the weakest (last added) skill
            while len(selected_local) > 2 and excess < self._min_excess:
                selected_local.pop()  # Remove last added (lowest combined score)
                selected_vecs_check = candidate_vectors[selected_local]
                excess = combination_diversity(selected_vecs_check).item()

        # Build result
        selected_indices = [candidate_indices[i] for i in selected_local]
        selected_scores = [candidate_sims[i] for i in selected_local]

        # Compute complementarity scores
        comp_scores = []
        selected_vecs = candidate_vectors[selected_local]
        for i, idx in enumerate(selected_local):
            if i == 0:
                comp_scores.append(0.0)
            else:
                # Average sin(θ) with all previously selected
                sims_with_prev = [
                    pair_sim[idx, selected_local[j]].item() for j in range(i)
                ]
                avg_sin = sum(
                    (1.0 - c ** 2) ** 0.5 for c in sims_with_prev
                ) / len(sims_with_prev)
                comp_scores.append(avg_sin)

        return RetrievalResult(
            indices=selected_indices,
            scores=selected_scores,
            complementarity_scores=comp_scores,
            diversity=combination_diversity(selected_vecs).item()
            if len(selected_local) >= 2
            else 0.0,
        )


class RetrievalResult:
    """Result of a skill retrieval operation."""

    def __init__(
        self,
        indices: list[int],
        scores: list[float],
        complementarity_scores: list[float],
        diversity: float = 0.0,
    ):
        self.indices = indices
        self.scores = scores
        self.complementarity_scores = complementarity_scores
        self.diversity = diversity

    def __repr__(self) -> str:
        items = []
        for i, (idx, score, comp) in enumerate(
            zip(self.indices, self.scores, self.complementarity_scores)
        ):
            items.append(f"  #{i}: skill[{idx}] sim={score:.3f} comp={comp:.3f}")
        lines = "\n".join(items)
        return f"RetrievalResult(n={len(self.indices)}, diversity={self.diversity:.4f})\n{lines}"

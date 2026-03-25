"""Confidence estimation for skill injection gating.

Two confidence sources:

1. Logit-based (compute_logit_confidence):
   H_k = -Σ p_i * log(p_i) over top-k tokens at each position
   confidence = 1 - H_k / log(k)  (normalized to [0, 1])
   Problem: always ~0.83 for instruction-tuned 7B models.

2. Sphere-Geometric Confidence (SphereConfidence):
   Combines three sphere-native signals into a confidence proxy:
   - Trajectory coherence η = displacement / path_length
   - Drift stability = 1 - smoothed(drift_norm)
   - Coverage proximity = 1 - isolation / threshold

   SGC = w₁·η + w₂·stability + w₃·coverage

   This is purely geometric, general across environments, and actually
   varies meaningfully — unlike logit entropy which is nearly constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def compute_logit_confidence(
    token_logprobs: list[list[tuple[str, float]]],
    top_k: int = 10,
    temperature: float = 2.0,
    action_region_only: bool = True,
    raw_output: str = "",
    action_tag: str = "action",
) -> float:
    """Compute confidence from truncated entropy over top-k logprobs.

    Temperature scaling is applied to logprobs before entropy computation
    to counteract overconfidence in instruction-tuned models (ATS paper).

    Args:
        token_logprobs: Per-token list of [(token, logprob), ...] from LLM.
        top_k: Number of top tokens used for truncated entropy.
        temperature: Temperature for calibration. T=1 is raw, T>1 flattens
            distribution (lowers confidence). Recommended 1.5-3.0 for
            RLHF models (ATS, EMNLP 2024).
        action_region_only: If True, focus on the action region tokens.
        raw_output: Raw output text (needed to locate action region).
        action_tag: Tag name to locate action region (e.g., "action", "answer").

    Returns:
        Confidence score in [0, 1]. Higher = more confident.
    """
    if not token_logprobs:
        return 0.5  # Fallback if no logprobs available

    # Focus on action region if requested
    region_logprobs = token_logprobs
    if action_region_only and raw_output:
        region_logprobs = _extract_action_region(
            token_logprobs, raw_output, action_tag,
        )

    if not region_logprobs:
        # If action region not found, fall back to full output
        region_logprobs = token_logprobs

    # Compute per-position truncated entropy, then average
    log_k = math.log(top_k) if top_k > 1 else 1.0
    entropies = []

    for top_tokens in region_logprobs:
        if not top_tokens:
            continue

        # Apply temperature scaling to logprobs:
        # logprob_scaled = logprob / T  (equivalent to logit / T before softmax)
        scaled_logprobs = [lp / temperature for _, lp in top_tokens[:top_k]]

        # Convert to probs
        # For numerical stability, subtract max before exp
        max_lp = max(scaled_logprobs)
        probs = [math.exp(lp - max_lp) for lp in scaled_logprobs]

        # Renormalize
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]

        # Truncated entropy: H_k = -Σ p_i * log(p_i)
        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * math.log(p)

        entropies.append(h)

    if not entropies:
        return 0.5

    avg_entropy = sum(entropies) / len(entropies)
    confidence = 1.0 - avg_entropy / log_k

    return max(0.0, min(1.0, confidence))


def _extract_action_region(
    token_logprobs: list[list[tuple[str, float]]],
    raw_output: str,
    action_tag: str = "action",
) -> list[list[tuple[str, float]]]:
    """Extract logprobs for tokens within the action region.

    Tries multiple tag patterns: <action>, <answer>, <search>.
    """
    # Try the specified tag first, then common alternatives
    tags_to_try = [action_tag, "action", "answer", "search"]
    seen = set()

    for tag in tags_to_try:
        if tag in seen:
            continue
        seen.add(tag)

        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        start_pos = raw_output.find(open_tag)
        end_pos = raw_output.find(close_tag)

        if start_pos >= 0 and end_pos >= 0:
            # Map character positions to token indices
            char_pos = 0
            start_idx = None
            end_idx = None

            for i, top_tokens in enumerate(token_logprobs):
                if top_tokens:
                    token_text = top_tokens[0][0]
                    if start_idx is None and char_pos >= start_pos + len(open_tag):
                        start_idx = i
                    if char_pos >= end_pos:
                        end_idx = i
                        break
                    char_pos += len(token_text)

            if start_idx is not None:
                end_idx = end_idx or len(token_logprobs)
                return token_logprobs[start_idx:end_idx]

    return []  # No action region found


# ---------------------------------------------------------------------------
# Sphere-Geometric Confidence (SGC)
# ---------------------------------------------------------------------------

@dataclass
class SGCSignals:
    """Raw signals that compose the Sphere-Geometric Confidence."""
    coherence: float      # η = displacement / path_length  ∈ [0, 1]
    stability: float      # 1 - smoothed(drift_norm)         ∈ [0, 1]
    coverage: float       # 1 - isolation / threshold         ∈ [0, 1]
    sgc: float            # Weighted combination              ∈ [0, 1]


class SphereConfidence:
    """Sphere-Geometric Confidence — replaces logit entropy as injection gate.

    Combines three sphere-native signals that are available at every step
    without depending on LLM internals or environment-specific feedback.

    Properties:
    - General: works identically across all environments
    - Formal: each component has clear geometric definition
    - Sphere-native: only meaningful on a unit sphere manifold
    - Actually varies: unlike logit entropy (~0.83 constant for 7B models)

    The output SGC ∈ [0, 1] is used as a drop-in replacement for logit
    confidence in the injection formula:
        base_strength = max(min_inject_strength, 1.0 - SGC)
    """

    def __init__(
        self,
        w_coherence: float = 0.55,
        w_stability: float = 0.45,
        warmup_steps: int = 1,
    ):
        """
        Args:
            w_coherence: Weight for trajectory coherence signal.
            w_stability: Weight for drift stability signal.
            warmup_steps: Steps before SGC activates (use default 0.5 during warmup).

        Note: Coverage (isolation-based retrieval quality) is handled separately
        as a multiplicative gate in DynamicInjector.decide(), not as part of
        confidence. This avoids the tension where low coverage would lower
        confidence → increase injection strength (opposite of desired behavior).
        """
        self.w_coherence = w_coherence
        self.w_stability = w_stability
        self.warmup_steps = warmup_steps
        self._step_count = 0

    def reset(self) -> None:
        """Reset for a new episode."""
        self._step_count = 0

    def compute(
        self,
        coherence: float,
        stability: float,
        isolation_score: float = 0.0,
    ) -> SGCSignals:
        """Compute Sphere-Geometric Confidence from sphere trajectory signals.

        Coverage (isolation-based) is NOT included here — it's applied as a
        separate multiplicative gate on injection weights in DynamicInjector.

        Args:
            coherence: Trajectory coherence η from IntentTracker.DriftInfo.
            stability: Drift stability from IntentTracker.DriftInfo.
            isolation_score: Unused, kept for API compatibility.

        Returns:
            SGCSignals with individual components and combined SGC.
        """
        self._step_count += 1

        # During warmup, not enough trajectory data → moderate confidence
        if self._step_count <= self.warmup_steps:
            return SGCSignals(
                coherence=coherence,
                stability=stability,
                coverage=0.0,
                sgc=0.5,
            )

        # Weighted combination of agent-state signals → SGC ∈ [0, 1]
        sgc = self.w_coherence * coherence + self.w_stability * stability
        sgc = max(0.0, min(1.0, sgc))

        return SGCSignals(
            coherence=coherence,
            stability=stability,
            coverage=0.0,
            sgc=sgc,
        )

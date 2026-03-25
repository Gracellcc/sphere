"""Skill combination via Slerp and LLM-based text synthesis.

Combines multiple selected skills into a single coherent guidance:
1. Geometric combination: Slerp to find the combined point on the sphere.
2. Text synthesis: LLM merges the skill texts into a cohesive instruction.
"""

from __future__ import annotations

import torch
from torch import Tensor

from skill_sphere.geometry.sphere import slerp, multi_slerp, l2_normalize, geodesic_distance


class SkillCombiner:
    """Combines selected skills via spherical interpolation and text synthesis."""

    def __init__(
        self,
        synthesis_model: str | None = None,
        synthesis_device: str = "cuda",
    ):
        """
        Args:
            synthesis_model: Model name for LLM text synthesis.
                If None, combines by concatenation instead of LLM synthesis.
            synthesis_device: Device for the synthesis model.
        """
        self.synthesis_model = synthesis_model
        self.synthesis_device = synthesis_device
        self._llm = None

    def combine_vectors(
        self,
        vectors: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Combine skill vectors via weighted Slerp on the sphere.

        Args:
            vectors: (K, D) unit vectors of selected skills.
            weights: (K,) non-negative weights. If None, uses uniform weights.

        Returns:
            (D,) combined unit vector on the sphere.
        """
        k = vectors.shape[0]
        if k == 1:
            return l2_normalize(vectors[0])

        if weights is None:
            weights = torch.ones(k, device=vectors.device)

        return multi_slerp(vectors, weights)

    def combine_texts(
        self,
        skill_texts: list[str],
        task_description: str,
        weights: list[float] | None = None,
    ) -> str:
        """Synthesize a coherent guidance text from multiple skill texts.

        Args:
            skill_texts: List of skill description texts to combine.
            task_description: The current task being performed.
            weights: Optional relevance weights for each skill.

        Returns:
            A single coherent guidance string.
        """
        if len(skill_texts) == 1:
            return skill_texts[0]

        if self.synthesis_model is not None:
            return self._llm_synthesize(skill_texts, task_description, weights)
        else:
            return self._template_combine(skill_texts, task_description, weights)

    def _template_combine(
        self,
        skill_texts: list[str],
        task_description: str,
        weights: list[float] | None = None,
    ) -> str:
        """Simple template-based combination (no LLM needed)."""
        parts = []
        for i, text in enumerate(skill_texts):
            weight_str = f" (relevance: {weights[i]:.2f})" if weights else ""
            parts.append(f"- {text}{weight_str}")

        guidance = (
            f"For the task: {task_description}\n"
            f"Apply these complementary strategies:\n"
            + "\n".join(parts)
        )
        return guidance

    def _llm_synthesize(
        self,
        skill_texts: list[str],
        task_description: str,
        weights: list[float] | None = None,
    ) -> str:
        """Use LLM to synthesize a coherent guidance from multiple skills."""
        if self._llm is None:
            self._load_llm()

        skill_block = "\n".join(
            f"{i + 1}. {text}" for i, text in enumerate(skill_texts)
        )

        prompt = (
            f"You are a skill synthesis assistant. Given multiple skill descriptions "
            f"and a task, merge them into a single coherent action guideline.\n\n"
            f"Task: {task_description}\n\n"
            f"Skills to combine:\n{skill_block}\n\n"
            f"Synthesize a concise, actionable guidance that integrates all the above "
            f"skills into a coherent strategy for the task. Be specific and direct."
        )

        try:
            from transformers import pipeline

            if self._llm is None:
                return self._template_combine(skill_texts, task_description, weights)

            output = self._llm(
                prompt,
                max_new_tokens=256,
                do_sample=False,
            )
            return output[0]["generated_text"].split("Synthesize a concise")[-1].strip()
        except Exception:
            # Fallback to template if LLM fails
            return self._template_combine(skill_texts, task_description, weights)

    def _load_llm(self):
        """Lazy load the synthesis LLM."""
        try:
            from transformers import pipeline

            self._llm = pipeline(
                "text-generation",
                model=self.synthesis_model,
                device=self.synthesis_device,
            )
        except Exception:
            self._llm = None

    def compute_combination_weights(
        self,
        query: Tensor,
        skill_vectors: Tensor,
        confidence: float = 1.0,
        sigma: float = 1.0,
    ) -> Tensor:
        """Compute injection weights based on relevance and confidence.

        weight_i = (1 - confidence) * exp(-distance_i² / σ²)

        Args:
            query: (D,) query unit vector.
            skill_vectors: (K, D) selected skill vectors.
            confidence: Agent's confidence in [0, 1]. Lower = stronger injection.
            sigma: Distance decay parameter.

        Returns:
            (K,) non-negative weights.
        """
        distances = torch.stack([
            geodesic_distance(query, v) for v in skill_vectors
        ])

        weights = (1.0 - confidence) * torch.exp(-distances ** 2 / (sigma ** 2))
        return weights

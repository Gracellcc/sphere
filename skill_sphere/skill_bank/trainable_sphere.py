"""TrainableSphere: SkillSphere with learnable skill positions.

Makes skill vectors nn.Parameters so their positions on the hypersphere
can be optimized via gradient descent (REINFORCE on sphere).

Key difference from SkillSphere:
- Skill vectors are nn.Parameters (requires_grad=True)
- Provides soft_retrieve() with differentiable skill selection
- After each gradient step, vectors are projected back to the sphere
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from skill_sphere.geometry.sphere import l2_normalize
from skill_sphere.skill_bank.skill_sphere import Skill, SkillSphere
from skill_sphere.skill_bank.encoder import SkillEncoder


class TrainableSphere(nn.Module):
    """Skill sphere with learnable vector positions.

    Wraps a SkillSphere and makes skill vectors trainable.
    Uses REINFORCE to update positions based on episode rewards.
    """

    def __init__(
        self,
        skill_sphere: SkillSphere,
        temperature: float = 0.1,
        top_k: int = 3,
    ):
        super().__init__()
        self.skills = skill_sphere.skills  # List[Skill] - text metadata
        self.temperature = temperature
        self.top_k = top_k

        # Make skill vectors trainable nn.Parameters
        vectors = skill_sphere.vectors.clone().detach()  # (N, D)
        self.skill_vectors = nn.Parameter(vectors, requires_grad=True)

        # Store original positions for regularization
        self.register_buffer("initial_vectors", vectors.clone())

        # Skill texts for injection
        self.skill_texts = [s.text for s in self.skills]
        self.skill_names = [s.name for s in self.skills]
        self.skill_categories = [s.category for s in self.skills]

        # Running baseline for REINFORCE variance reduction
        self.register_buffer("reward_baseline", torch.tensor(0.0))
        self.baseline_momentum = 0.9

        # Tracking
        self.update_count = 0

    @property
    def n_skills(self) -> int:
        return self.skill_vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.skill_vectors.shape[1]

    def normalized_vectors(self) -> Tensor:
        """Return L2-normalized skill vectors (project to sphere)."""
        return l2_normalize(self.skill_vectors)

    @torch.no_grad()
    def project_to_sphere(self):
        """Project skill vectors back to unit sphere after gradient step."""
        self.skill_vectors.data = l2_normalize(self.skill_vectors.data)

    def soft_retrieve(
        self,
        query: Tensor,
        k: int | None = None,
        category_filter: str | None = None,
    ) -> dict:
        """Differentiable skill retrieval using temperature-scaled softmax.

        Args:
            query: (D,) unit vector query (encoded task description).
            k: Number of skills to select. Defaults to self.top_k.
            category_filter: If set, only consider skills in this category.

        Returns:
            Dict with:
                - indices: selected skill indices
                - texts: selected skill texts
                - log_probs: log probabilities of selection (for REINFORCE)
                - probs: selection probabilities
        """
        k = k or self.top_k
        vectors = self.normalized_vectors()  # (N, D)

        # Apply category filter
        if category_filter:
            mask = torch.tensor(
                [c == category_filter for c in self.skill_categories],
                device=vectors.device,
            )
            if mask.sum() == 0:
                mask = torch.ones(self.n_skills, dtype=torch.bool, device=vectors.device)
        else:
            mask = torch.ones(self.n_skills, dtype=torch.bool, device=vectors.device)

        # Compute similarities
        sims = query @ vectors.T  # (N,)

        # Mask out irrelevant categories
        masked_sims = sims.clone()
        masked_sims[~mask] = -float("inf")

        # Temperature-scaled softmax
        logits = masked_sims / self.temperature
        probs = torch.softmax(logits, dim=0)  # (N,)

        # Sample top-k skills (without replacement)
        # Use Gumbel-top-k for differentiability during training
        k = min(k, mask.sum().item())
        if k == 0:
            return {
                "indices": [],
                "texts": [],
                "log_probs": torch.tensor(0.0, device=vectors.device),
                "probs": probs,
            }

        # Top-k by probability (deterministic during eval, stochastic during train)
        if self.training:
            # Gumbel noise for exploration
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            noisy_logits = logits + gumbel * 0.5  # moderate exploration
            noisy_logits[~mask] = -float("inf")
            _, indices = torch.topk(noisy_logits, k)
        else:
            _, indices = torch.topk(logits, k)

        indices = indices.tolist()
        texts = [self.skill_texts[i] for i in indices]
        log_probs = torch.log(probs[indices] + 1e-10).sum()

        return {
            "indices": indices,
            "texts": texts,
            "log_probs": log_probs,
            "probs": probs,
            "similarities": sims,
        }

    def compute_reinforce_loss(
        self,
        log_probs: Tensor,
        reward: float,
    ) -> Tensor:
        """Compute REINFORCE loss for skill selection.

        Args:
            log_probs: Sum of log probabilities of selected skills.
            reward: Episode reward (0 or 1 for ALFWorld).

        Returns:
            Scalar loss for backpropagation.
        """
        # Update baseline
        advantage = reward - self.reward_baseline.item()
        self.reward_baseline.data = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * reward
        )

        # REINFORCE: maximize E[R * log p(selected)]
        # Loss = -advantage * log_prob
        reinforce_loss = -advantage * log_probs

        # Regularization: don't drift too far from initial positions
        # This prevents catastrophic forgetting of the initial skill structure
        drift = 1.0 - (
            self.normalized_vectors() * self.initial_vectors
        ).sum(dim=1).mean()
        reg_loss = 0.1 * drift

        return reinforce_loss + reg_loss

    def get_movement_stats(self) -> dict:
        """Track how much skills have moved from their initial positions."""
        with torch.no_grad():
            current = self.normalized_vectors()
            initial = self.initial_vectors
            cos_sims = (current * initial).sum(dim=1)  # (N,)
            geodesic_dists = torch.acos(cos_sims.clamp(-1.0, 1.0))

        return {
            "mean_drift": geodesic_dists.mean().item(),
            "max_drift": geodesic_dists.max().item(),
            "min_cos_sim": cos_sims.min().item(),
            "mean_cos_sim": cos_sims.mean().item(),
        }

    def save(self, path: str | Path):
        """Save trainable sphere state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "skill_vectors": self.skill_vectors.data,
            "initial_vectors": self.initial_vectors,
            "reward_baseline": self.reward_baseline,
            "update_count": self.update_count,
            "skills": [s.to_dict() for s in self.skills],
        }, path / "trainable_sphere.pt")

    def load(self, path: str | Path):
        """Load trainable sphere state."""
        path = Path(path)
        state = torch.load(path / "trainable_sphere.pt", map_location="cpu")
        self.skill_vectors.data = state["skill_vectors"].to(self.skill_vectors.device)
        self.initial_vectors = state["initial_vectors"].to(self.skill_vectors.device)
        self.reward_baseline = state["reward_baseline"].to(self.skill_vectors.device)
        self.update_count = state["update_count"]

    def update_skill(self, idx: int, new_text: str, encoder: SkillEncoder):
        """Update a skill's text and re-encode its vector.

        The new vector inherits the current (trained) position as a starting
        point but is blended with the fresh encoding to reflect the new text.
        """
        self.skills[idx] = Skill(
            text=new_text,
            name=self.skill_names[idx],
            principle=new_text,
            when_to_apply=self.skills[idx].when_to_apply,
            category=self.skill_categories[idx],
            source="evolved",
        )
        self.skill_texts[idx] = new_text

        with torch.no_grad():
            new_vec = encoder.encode(new_text).squeeze(0).to(self.skill_vectors.device)
            # Blend: 50% new encoding + 50% current trained position
            blended = l2_normalize(
                (0.5 * new_vec + 0.5 * self.skill_vectors.data[idx]).unsqueeze(0)
            ).squeeze(0)
            self.skill_vectors.data[idx] = blended

    def add_skill(self, text: str, name: str, category: str, encoder: SkillEncoder):
        """Add a new skill to the sphere."""
        skill = Skill(
            text=text, name=name, principle=text,
            category=category, source="evolved",
        )
        self.skills.append(skill)
        self.skill_texts.append(text)
        self.skill_names.append(name)
        self.skill_categories.append(category)

        with torch.no_grad():
            new_vec = encoder.encode(text).squeeze(0).to(self.skill_vectors.device)
            new_vecs = torch.cat([self.skill_vectors.data, new_vec.unsqueeze(0)], dim=0)
            new_init = torch.cat([self.initial_vectors, new_vec.unsqueeze(0)], dim=0)

        # Rebuild parameter with new size
        self.skill_vectors = nn.Parameter(new_vecs, requires_grad=True)
        self.register_buffer("initial_vectors", new_init)

    def remove_skill(self, idx: int):
        """Remove a skill from the sphere."""
        del self.skills[idx]
        del self.skill_texts[idx]
        del self.skill_names[idx]
        del self.skill_categories[idx]

        with torch.no_grad():
            mask = list(range(self.skill_vectors.shape[0]))
            mask.pop(idx)
            new_vecs = self.skill_vectors.data[mask]
            new_init = self.initial_vectors[mask]

        self.skill_vectors = nn.Parameter(new_vecs, requires_grad=True)
        self.register_buffer("initial_vectors", new_init)

    def get_skill_usage_stats(self, selected_indices_list: list[list[int]]) -> dict:
        """Compute per-skill selection frequency and success correlation."""
        n = self.n_skills
        select_count = [0] * n
        success_count = [0] * n
        for indices in selected_indices_list:
            for i in indices:
                if i < n:
                    select_count[i] += 1
        return {
            "select_count": select_count,
            "never_selected": [i for i in range(n) if select_count[i] == 0],
        }

    @classmethod
    def from_skill_json(
        cls,
        json_path: str | Path,
        encoder: SkillEncoder | None = None,
        device: str = "cuda",
        temperature: float = 0.1,
        top_k: int = 3,
    ) -> TrainableSphere:
        """Create TrainableSphere from SkillRL-format JSON."""
        sphere = SkillSphere.from_skillrl_json(json_path, encoder=encoder, device=device)
        return cls(sphere, temperature=temperature, top_k=top_k)

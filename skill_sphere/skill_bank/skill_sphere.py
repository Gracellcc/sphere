"""SkillSphere: the main skill library living on a unit hypersphere.

Stores skill texts with their sphere embeddings and metadata,
provides retrieval, combination, analysis, and persistence operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from torch import Tensor

from skill_sphere.geometry.sphere import (
    l2_normalize,
    cosine_similarity_matrix,
    find_nearest_neighbors,
    find_redundant_pairs,
    find_antipodal_pairs,
    pairwise_geodesic_distance,
)
from skill_sphere.geometry.voronoi import voronoi_areas, find_sparse_regions, coverage_uniformity
from skill_sphere.skill_bank.encoder import SkillEncoder


@dataclass
class Skill:
    """A single skill entry in the sphere."""

    text: str
    name: str = ""
    principle: str = ""
    when_to_apply: str = ""
    category: str = "general"  # "general" or task-specific category name
    source: str = "distilled"  # "distilled", "synthesized", "merged", "generalized"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SkillSphere:
    """Spherical skill library: stores, retrieves, and manipulates skills on S^{d-1}."""

    def __init__(
        self,
        encoder: SkillEncoder | None = None,
        device: str = "cuda",
    ):
        """
        Args:
            encoder: SkillEncoder instance. If None, will be created lazily.
            device: Torch device for storing vectors.
        """
        self.device = device
        self._encoder = encoder
        self.skills: list[Skill] = []
        self._vectors: list[Tensor] = []  # Each (D,) unit vector
        self._vectors_stacked: Tensor | None = None  # Cached (N, D) matrix

    @property
    def encoder(self) -> SkillEncoder:
        if self._encoder is None:
            self._encoder = SkillEncoder(device=self.device)
        return self._encoder

    @property
    def vectors(self) -> Tensor:
        """Return (N, D) tensor of all skill vectors, using cache when valid."""
        if self._vectors_stacked is None or self._vectors_stacked.shape[0] != len(self._vectors):
            if len(self._vectors) == 0:
                dim = self.encoder.embedding_dim
                return torch.zeros(0, dim, device=self.device)
            self._vectors_stacked = torch.stack(self._vectors, dim=0)
        return self._vectors_stacked

    def _invalidate_cache(self):
        self._vectors_stacked = None

    def __len__(self) -> int:
        return len(self.skills)

    def add_skill(
        self,
        text: str,
        vector: Tensor | None = None,
        name: str = "",
        principle: str = "",
        when_to_apply: str = "",
        category: str = "general",
        source: str = "distilled",
        metadata: dict | None = None,
    ) -> int:
        """Add a skill to the sphere.

        Args:
            text: Full skill description text.
            vector: Pre-computed sphere vector. If None, will be encoded.
            name: Short name for the skill.
            principle: Core principle of the skill.
            when_to_apply: When this skill should be used.
            category: "general" or a task-specific category.
            source: How this skill was created.
            metadata: Additional metadata.

        Returns:
            Index of the newly added skill.
        """
        skill = Skill(
            text=text,
            name=name,
            principle=principle,
            when_to_apply=when_to_apply,
            category=category,
            source=source,
            metadata=metadata or {},
        )

        if vector is None:
            vector = self.encoder.encode(text).squeeze(0).to(self.device)
        else:
            vector = l2_normalize(vector.to(self.device).unsqueeze(0)).squeeze(0)

        self.skills.append(skill)
        self._vectors.append(vector)
        self._invalidate_cache()

        return len(self.skills) - 1

    def remove_skill(self, idx: int):
        """Remove a skill by index."""
        del self.skills[idx]
        del self._vectors[idx]
        self._invalidate_cache()

    def get_skill(self, idx: int) -> tuple[Skill, Tensor]:
        """Get a skill and its vector by index."""
        return self.skills[idx], self._vectors[idx]

    def find_nearest(
        self, query: Tensor, k: int = 5, category: str | None = None
    ) -> list[tuple[int, float]]:
        """Find k nearest skills to a query vector.

        Args:
            query: (D,) query unit vector.
            k: Number of neighbors.
            category: If set, only search within this category.

        Returns:
            List of (skill_index, cosine_similarity) tuples, sorted by similarity.
        """
        if len(self) == 0:
            return []

        if category is not None:
            # Filter by category
            mask = [i for i, s in enumerate(self.skills) if s.category == category]
            if not mask:
                return []
            filtered_vectors = self.vectors[mask]
            indices, sims = find_nearest_neighbors(query, filtered_vectors, k)
            return [(mask[i.item()], s.item()) for i, s in zip(indices, sims)]

        indices, sims = find_nearest_neighbors(query, self.vectors, k)
        return [(i.item(), s.item()) for i, s in zip(indices, sims)]

    def find_redundant(self, threshold: float = 0.95) -> list[tuple[int, int]]:
        """Find pairs of redundant (nearly identical) skills."""
        if len(self) < 2:
            return []
        return find_redundant_pairs(self.vectors, threshold)

    def find_antipodal(self, threshold: float = -0.95) -> list[tuple[int, int]]:
        """Find pairs of opposing skills."""
        if len(self) < 2:
            return []
        return find_antipodal_pairs(self.vectors, threshold)

    def get_voronoi_areas(self, n_samples: int = 10000) -> Tensor:
        """Compute Voronoi area distribution across skills."""
        if len(self) == 0:
            return torch.tensor([], device=self.device)
        return voronoi_areas(self.vectors, n_samples)

    def get_sparse_regions(
        self, n_samples: int = 10000, top_k: int = 5
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Find the most sparsely covered regions."""
        return find_sparse_regions(self.vectors, n_samples, top_k)

    def get_coverage_uniformity(self, n_samples: int = 10000) -> float:
        """Measure how uniformly skills cover the sphere (0-1)."""
        if len(self) < 2:
            return 0.0
        return coverage_uniformity(self.vectors, n_samples)

    def get_distance_matrix(self) -> Tensor:
        """Compute pairwise geodesic distance matrix."""
        return pairwise_geodesic_distance(self.vectors)

    def get_general_skills(self) -> list[tuple[int, Skill]]:
        """Return all general (cross-task) skills with their indices."""
        return [(i, s) for i, s in enumerate(self.skills) if s.category == "general"]

    def get_task_skills(self, category: str) -> list[tuple[int, Skill]]:
        """Return skills for a specific task category."""
        return [(i, s) for i, s in enumerate(self.skills) if s.category == category]

    def get_common_mistakes(self) -> list[tuple[int, Skill]]:
        """Return common mistakes with their indices."""
        return [(i, s) for i, s in enumerate(self.skills) if s.category == "common_mistakes"]

    def get_skills_by_category(self, category: str) -> list[dict]:
        """Return skill dicts for a given category.

        Returns list of dicts with keys: title, principle, when_to_apply, text, category.
        """
        results = []
        for skill in self.skills:
            if skill.category == category:
                results.append({
                    "title": skill.name,
                    "principle": skill.principle,
                    "when_to_apply": skill.when_to_apply,
                    "text": skill.text,
                    "category": skill.category,
                })
        return results

    def get_categories(self) -> list[str]:
        """Return all unique skill categories."""
        return list(set(s.category for s in self.skills))

    # --- Persistence ---

    def save(self, path: str | Path):
        """Save the SkillSphere to disk (JSON + tensor)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save skill metadata
        skills_data = [s.to_dict() for s in self.skills]
        with open(path / "skills.json", "w") as f:
            json.dump(skills_data, f, indent=2, ensure_ascii=False)

        # Save vectors
        if len(self._vectors) > 0:
            torch.save(self.vectors, path / "vectors.pt")

    @classmethod
    def load(
        cls,
        path: str | Path,
        encoder: SkillEncoder | None = None,
        device: str = "cuda",
    ) -> SkillSphere:
        """Load a SkillSphere from disk."""
        path = Path(path)

        sphere = cls(encoder=encoder, device=device)

        # Load skill metadata
        with open(path / "skills.json") as f:
            skills_data = json.load(f)
        sphere.skills = [Skill.from_dict(d) for d in skills_data]

        # Load vectors
        vectors_path = path / "vectors.pt"
        if vectors_path.exists():
            vectors = torch.load(vectors_path, map_location=device, weights_only=True)
            sphere._vectors = [vectors[i] for i in range(vectors.shape[0])]
            sphere._vectors_stacked = vectors

        return sphere

    @classmethod
    def from_skillrl_json(
        cls,
        json_path: str | Path,
        encoder: SkillEncoder | None = None,
        device: str = "cuda",
    ) -> SkillSphere:
        """Load skills from SkillRL's claude_style_skills.json format.

        Supports two formats:

        Format A (nested, actual SkillRL format):
        {
            "general_skills": [{"skill_id": ..., "title": ..., "principle": ..., "when_to_apply": ...}, ...],
            "task_specific_skills": {
                "pick_and_place": [...],
                "heat": [...],
                ...
            }
        }

        Format B (flat list):
        [{"title": ..., "principle": ..., "when_to_apply": ..., "category": ...}, ...]
        """
        with open(json_path) as f:
            raw_data = json.load(f)

        # Normalize to a flat list of (skill_dict, category) pairs
        entries: list[tuple[dict, str]] = []

        if isinstance(raw_data, dict) and "general_skills" in raw_data:
            # Format A: nested
            for s in raw_data.get("general_skills", []):
                entries.append((s, "general"))
            # task_specific_skills (our V2 format) or query_type_skills (SkillRL search format)
            task_skills = raw_data.get("task_specific_skills", raw_data.get("query_type_skills", {}))
            for category, skills in task_skills.items():
                for s in skills:
                    entries.append((s, category))
            # Common mistakes (SkillRL format: mistake_id, description, why_it_happens, how_to_avoid)
            for m in raw_data.get("common_mistakes", []):
                entries.append(({
                    "title": m.get("description", "")[:60],
                    "principle": m.get("description", ""),
                    "when_to_apply": m.get("how_to_avoid", ""),
                    "skill_id": m.get("mistake_id", ""),
                }, "common_mistakes"))
            # Gap-filling skills (V2: added by fill_sphere_gaps.py)
            for s in raw_data.get("gap_filling_skills", []):
                entries.append((s, "gap_filling"))
        elif isinstance(raw_data, list):
            # Format B: flat list
            for s in raw_data:
                entries.append((s, s.get("category", "general")))
        else:
            raise ValueError(f"Unrecognized SkillRL JSON format in {json_path}")

        sphere = cls(encoder=encoder, device=device)

        # Compose full text for each skill
        texts = []
        for s, _category in entries:
            text_parts = []
            if s.get("title"):
                text_parts.append(s["title"])
            if s.get("principle"):
                text_parts.append(s["principle"])
            if s.get("when_to_apply"):
                text_parts.append(f"Apply when: {s['when_to_apply']}")
            texts.append(". ".join(text_parts))

        # Batch encode
        if texts:
            vectors = sphere.encoder.encode(texts).to(device)

            for i, ((s, category), text) in enumerate(zip(entries, texts)):
                sphere.skills.append(Skill(
                    text=text,
                    name=s.get("title", s.get("name", "")),
                    principle=s.get("principle", ""),
                    when_to_apply=s.get("when_to_apply", ""),
                    category=category,
                    source="distilled",
                    metadata={"skill_id": s.get("skill_id", "")},
                ))
                sphere._vectors.append(vectors[i])

        return sphere

    # --- Stats ---

    def summary(self) -> dict:
        """Return summary statistics of the skill sphere."""
        categories = {}
        for s in self.skills:
            categories[s.category] = categories.get(s.category, 0) + 1

        result = {
            "total_skills": len(self),
            "categories": categories,
            "embedding_dim": self.encoder.embedding_dim if self._encoder else "not loaded",
        }

        if len(self) >= 2:
            dist_matrix = self.get_distance_matrix()
            # Exclude diagonal
            mask = ~torch.eye(len(self), dtype=torch.bool, device=self.device)
            dists = dist_matrix[mask]
            result["avg_pairwise_distance"] = dists.mean().item()
            result["min_pairwise_distance"] = dists.min().item()
            result["max_pairwise_distance"] = dists.max().item()
            result["coverage_uniformity"] = self.get_coverage_uniformity()

        return result

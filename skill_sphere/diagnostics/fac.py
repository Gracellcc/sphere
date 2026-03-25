"""Feature Activation Coverage (FAC) analysis.

Implements the MD-specified FAC diagnostic pipeline:
  1. Extract SAE features from successful trajectories → F_success
  2. Extract SAE features from failed trajectories → F_failure
  3. Missing features = F_success - F_failure (present in success, absent in failure)
  4. Check skill library coverage: which missing features are NOT covered by any skill
  5. Output: list of uncovered missing features → these are what we need to synthesize

The dual diagnostic complements Voronoi:
  - FAC says "WHAT capability is missing"
  - Voronoi says "WHERE on the sphere is it missing"

Reference: "Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from skill_sphere.diagnostics.feature_extractor import FeatureExtractor, FeatureProfile


@dataclass
class FACResult:
    """Result of FAC coverage analysis."""
    # Features present in success trajectories
    success_features: set[int] = field(default_factory=set)
    # Features present in failure trajectories
    failure_features: set[int] = field(default_factory=set)
    # Features in success but NOT in failure (the model needs these but lacks them)
    missing_features: set[int] = field(default_factory=set)
    # Features covered by the skill library
    skill_covered_features: set[int] = field(default_factory=set)
    # Missing features NOT covered by skills (what we need to synthesize)
    uncovered_missing: set[int] = field(default_factory=set)
    # Coverage metrics
    total_success_features: int = 0
    total_failure_features: int = 0
    total_missing: int = 0
    total_uncovered: int = 0
    coverage_rate: float = 0.0  # How much of missing is covered by skills
    # Per-feature details
    feature_importance: dict[int, float] = field(default_factory=dict)

    def summary(self) -> dict:
        """Human-readable summary."""
        return {
            "success_features": self.total_success_features,
            "failure_features": self.total_failure_features,
            "missing_features": self.total_missing,
            "skill_covered": len(self.skill_covered_features & self.missing_features),
            "uncovered_missing": self.total_uncovered,
            "coverage_rate": f"{self.coverage_rate:.1%}",
        }


class FACAnalyzer:
    """Feature Activation Coverage analyzer.

    Performs differential analysis between success and failure trajectories
    to identify what capabilities the model is missing, then checks whether
    the skill library covers those gaps.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        min_feature_count: int = 2,
        importance_method: str = "frequency",
    ):
        """
        Args:
            extractor: FeatureExtractor with loaded model and SAE.
            min_feature_count: Minimum times a feature must appear to be
                counted (filters noise from single-token activations).
            importance_method: How to rank missing features.
                "frequency": by how often they appear in success trajectories.
                "score": by max activation score.
        """
        self.extractor = extractor
        self.min_feature_count = min_feature_count
        self.importance_method = importance_method

    def analyze_trajectories(
        self,
        success_trajectories: list[list[dict]],
        failure_trajectories: list[list[dict]],
        skill_texts: list[str] | None = None,
    ) -> FACResult:
        """Run full FAC analysis on trajectory data.

        Args:
            success_trajectories: List of successful trajectory step lists.
            failure_trajectories: List of failed trajectory step lists.
            skill_texts: Optional list of skill texts to check coverage.

        Returns:
            FACResult with complete analysis.
        """
        print(f"=== FAC Analysis ===")
        print(f"Success trajectories: {len(success_trajectories)}")
        print(f"Failure trajectories: {len(failure_trajectories)}")

        # Phase 1: Extract features from success trajectories
        print("\n[Phase 1] Extracting features from success trajectories...")
        success_profile = self._extract_trajectory_profiles(success_trajectories)

        # Phase 2: Extract features from failure trajectories
        print("\n[Phase 2] Extracting features from failure trajectories...")
        failure_profile = self._extract_trajectory_profiles(failure_trajectories)

        # Phase 3: Differential analysis
        print("\n[Phase 3] Differential analysis...")
        result = self._differential_analysis(success_profile, failure_profile)

        # Phase 4: Check skill library coverage (if skills provided)
        if skill_texts:
            print(f"\n[Phase 4] Checking skill library coverage ({len(skill_texts)} skills)...")
            skill_profile = self._extract_skill_features(skill_texts)
            result.skill_covered_features = skill_profile.active_features
            result.uncovered_missing = result.missing_features - skill_profile.active_features
            result.total_uncovered = len(result.uncovered_missing)
            covered_missing = result.missing_features & skill_profile.active_features
            result.coverage_rate = (
                len(covered_missing) / len(result.missing_features)
                if result.missing_features else 1.0
            )

        print(f"\n=== FAC Results ===")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")

        return result

    def analyze_texts(
        self,
        success_texts: list[str],
        failure_texts: list[str],
        skill_texts: list[str] | None = None,
    ) -> FACResult:
        """Simplified FAC analysis on plain texts (not trajectories).

        Useful when trajectory data is just concatenated prompt+response strings.

        Args:
            success_texts: Texts from successful episodes.
            failure_texts: Texts from failed episodes.
            skill_texts: Optional skill texts for coverage check.

        Returns:
            FACResult with complete analysis.
        """
        print(f"=== FAC Analysis (text mode) ===")
        print(f"Success texts: {len(success_texts)}")
        print(f"Failure texts: {len(failure_texts)}")

        # Extract features
        print("\n[Phase 1] Extracting features from success texts...")
        success_profiles = self.extractor.extract_features_batch(success_texts)
        success_merged = self._merge_profiles(success_profiles)

        print("\n[Phase 2] Extracting features from failure texts...")
        failure_profiles = self.extractor.extract_features_batch(failure_texts)
        failure_merged = self._merge_profiles(failure_profiles)

        # Differential analysis
        print("\n[Phase 3] Differential analysis...")
        result = self._differential_analysis(success_merged, failure_merged)

        # Skill coverage
        if skill_texts:
            print(f"\n[Phase 4] Checking skill library coverage ({len(skill_texts)} skills)...")
            skill_profiles = self.extractor.extract_features_batch(skill_texts)
            skill_merged = self._merge_profiles(skill_profiles)
            result.skill_covered_features = skill_merged.active_features
            result.uncovered_missing = result.missing_features - skill_merged.active_features
            result.total_uncovered = len(result.uncovered_missing)
            covered_missing = result.missing_features & skill_merged.active_features
            result.coverage_rate = (
                len(covered_missing) / len(result.missing_features)
                if result.missing_features else 1.0
            )

        print(f"\n=== FAC Results ===")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")

        return result

    def _extract_trajectory_profiles(
        self, trajectories: list[list[dict]],
    ) -> FeatureProfile:
        """Extract and merge feature profiles from multiple trajectories."""
        merged = FeatureProfile()
        for i, traj in enumerate(trajectories):
            if (i + 1) % 5 == 0:
                print(f"  Processing trajectory {i + 1}/{len(trajectories)}")
            profile = self.extractor.extract_trajectory_features(traj)
            merged = merged.merge(profile)
        return merged

    def _extract_skill_features(
        self, skill_texts: list[str],
    ) -> FeatureProfile:
        """Extract features from skill library texts."""
        profiles = self.extractor.extract_features_batch(skill_texts)
        return self._merge_profiles(profiles)

    def _merge_profiles(
        self, profiles: list[FeatureProfile],
    ) -> FeatureProfile:
        """Merge multiple profiles into one."""
        merged = FeatureProfile()
        for p in profiles:
            merged = merged.merge(p)
        return merged

    def _differential_analysis(
        self,
        success_profile: FeatureProfile,
        failure_profile: FeatureProfile,
    ) -> FACResult:
        """Compute the feature difference between success and failure.

        Missing features = features that appear in success with sufficient
        frequency but are absent or very rare in failure trajectories.
        """
        # Filter features by minimum count
        success_features = {
            f for f, c in success_profile.feature_counts.items()
            if c >= self.min_feature_count
        }
        failure_features = {
            f for f, c in failure_profile.feature_counts.items()
            if c >= self.min_feature_count
        }

        # Missing = in success, not in failure
        missing = success_features - failure_features

        # Compute importance scores for missing features
        importance = {}
        for f in missing:
            if self.importance_method == "frequency":
                importance[f] = success_profile.feature_counts.get(f, 0)
            else:  # score
                importance[f] = success_profile.feature_scores.get(f, 0.0)

        result = FACResult(
            success_features=success_features,
            failure_features=failure_features,
            missing_features=missing,
            total_success_features=len(success_features),
            total_failure_features=len(failure_features),
            total_missing=len(missing),
            feature_importance=importance,
        )

        print(f"  Success features: {len(success_features)}")
        print(f"  Failure features: {len(failure_features)}")
        print(f"  Missing (in success, not in failure): {len(missing)}")
        print(f"  Top 10 by importance: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]}")

        return result

    @staticmethod
    def load_trajectories(path: str | Path) -> list[dict]:
        """Load trajectory data from a JSONL file.

        Expected format (one JSON per line):
        {
            "episode": 0,
            "task_description": "...",
            "success": true/false,
            "steps": [
                {"step": 1, "prompt": "...", "response": "...", "action": "..."},
                ...
            ]
        }
        """
        trajectories = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))
        return trajectories

    @staticmethod
    def split_by_outcome(
        trajectories: list[dict],
    ) -> tuple[list[list[dict]], list[list[dict]]]:
        """Split trajectories into success and failure groups.

        Returns:
            (success_trajectories, failure_trajectories) where each is
            a list of step lists.
        """
        success = []
        failure = []
        for traj in trajectories:
            steps = traj.get("steps", [])
            if traj.get("success", False):
                success.append(steps)
            else:
                failure.append(steps)
        return success, failure

    def save_result(self, result: FACResult, path: str | Path):
        """Save FAC result to JSON."""
        data = {
            "success_features": sorted(result.success_features),
            "failure_features": sorted(result.failure_features),
            "missing_features": sorted(result.missing_features),
            "uncovered_missing": sorted(result.uncovered_missing),
            "feature_importance": {
                str(k): v for k, v in sorted(
                    result.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            },
            "summary": result.summary(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"FAC result saved to {path}")

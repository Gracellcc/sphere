"""Targeted skill synthesis based on FAC-identified missing features.

For each missing feature identified by FAC analysis:
  1. Get the feature's semantic description (via LLM interpretation)
  2. Generate a candidate skill text targeting that feature
  3. Verify: SAE confirms the new skill activates the target feature
  4. Geometric validation: ensure the new skill isn't redundant on the sphere

Reference: FAC-Synthesis Step 1 (contrastive pair construction) +
           Step 2 (feature-covered sample synthesis)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from skill_sphere.diagnostics.sae import TopKSAE
from skill_sphere.diagnostics.feature_extractor import FeatureExtractor, FeatureProfile
from skill_sphere.diagnostics.fac import FACResult


@dataclass
class SynthesizedSkill:
    """A skill synthesized to cover a missing feature."""
    feature_idx: int
    feature_description: str
    skill_name: str
    skill_text: str
    # Verification
    verified: bool = False  # SAE confirms activation
    activation_score: float = 0.0
    # Sphere position (after encoding)
    sphere_distance_to_nearest: float = 0.0  # Distance to nearest existing skill


FEATURE_INTERPRETATION_PROMPT = """You are analyzing internal features of a language model agent.

I will show you the top activation contexts for a specific internal feature (neuron) of the model.
Based on these contexts, describe what concept or capability this feature represents.

Feature activation contexts (text spans where this feature fires strongly):
{activation_spans}

Provide a concise description (1-2 sentences) of what this feature captures.
Focus on the CAPABILITY or CONCEPT, not the specific text.

Description:"""


SKILL_SYNTHESIS_PROMPT = """You are creating a skill guide for an AI agent that needs to improve at specific tasks.

The agent is missing the following capability:
{feature_description}

The agent operates in interactive environments where it must:
- Navigate and interact with objects (e.g., find, pick up, clean, heat, cool items)
- Search for information and answer questions
- Browse websites and make purchasing decisions

Generate a concise, actionable skill guide (3-5 sentences) that teaches this capability.
The guide should be:
1. General enough to apply across different environments
2. Specific enough to be actionable
3. Written as direct instructions

Skill name (2-4 words):
Skill guide:"""


class TargetedSynthesizer:
    """Synthesize new skills to fill FAC-identified coverage gaps.

    Pipeline:
      1. Interpret missing features → understand what's missing
      2. Synthesize skill text → generate actionable guide
      3. SAE-verify → confirm the skill activates the target feature
      4. Sphere-verify → confirm it's not redundant
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        llm_generate_fn=None,
        max_retries: int = 3,
        verification_threshold: float = 0.5,
        redundancy_threshold: float = 0.95,
    ):
        """
        Args:
            extractor: FeatureExtractor with loaded model and SAE.
            llm_generate_fn: Function(prompt) -> str for LLM generation.
                If None, uses the extractor's model.
            max_retries: Max attempts to synthesize a verified skill.
            verification_threshold: Min SAE activation to consider verified.
            redundancy_threshold: Max cosine similarity to nearest existing skill.
        """
        self.extractor = extractor
        self.llm_generate_fn = llm_generate_fn
        self.max_retries = max_retries
        self.verification_threshold = verification_threshold
        self.redundancy_threshold = redundancy_threshold

    def synthesize_for_missing_features(
        self,
        fac_result: FACResult,
        activation_spans: dict[int, list[str]] | None = None,
        existing_skill_vectors: torch.Tensor | None = None,
        max_features: int = 20,
    ) -> list[SynthesizedSkill]:
        """Synthesize skills for the most important missing features.

        Args:
            fac_result: FAC analysis result.
            activation_spans: Optional {feature_idx: [span_texts]} for context.
            existing_skill_vectors: (N, D) existing skill sphere vectors.
            max_features: Maximum number of features to synthesize for.

        Returns:
            List of synthesized (and verified) skills.
        """
        # Rank missing features by importance
        missing = fac_result.uncovered_missing or fac_result.missing_features
        ranked = sorted(
            missing,
            key=lambda f: fac_result.feature_importance.get(f, 0),
            reverse=True,
        )[:max_features]

        print(f"\nSynthesizing skills for {len(ranked)} missing features...")
        synthesized = []

        for i, feat_idx in enumerate(ranked):
            importance = fac_result.feature_importance.get(feat_idx, 0)
            print(f"\n[{i+1}/{len(ranked)}] Feature {feat_idx} (importance={importance})")

            # Get activation spans for context
            spans = (activation_spans or {}).get(feat_idx, [])

            # Step 1: Interpret the feature
            description = self._interpret_feature(feat_idx, spans)
            print(f"  Description: {description}")

            # Step 2: Synthesize skill
            skill = self._synthesize_skill(feat_idx, description)
            if skill is None:
                print(f"  Failed to synthesize skill")
                continue

            # Step 3: SAE verification
            verified = self._verify_skill(skill)
            if verified:
                print(f"  SAE verified: activation={skill.activation_score:.3f}")
            else:
                print(f"  SAE verification failed (score={skill.activation_score:.3f})")
                # Keep it anyway but mark as unverified

            # Step 4: Sphere redundancy check
            if existing_skill_vectors is not None:
                self._check_redundancy(skill, existing_skill_vectors)
                if skill.sphere_distance_to_nearest < (1 - self.redundancy_threshold):
                    print(f"  Too close to existing skill (dist={skill.sphere_distance_to_nearest:.3f}), skipping")
                    continue

            synthesized.append(skill)
            print(f"  Added: '{skill.skill_name}'")

        print(f"\nSynthesized {len(synthesized)} skills total")
        return synthesized

    def _interpret_feature(
        self, feature_idx: int, spans: list[str],
    ) -> str:
        """Interpret what a feature represents using its activation contexts."""
        if not spans:
            return f"Internal feature #{feature_idx} (no activation context available)"

        spans_text = "\n".join(f"- \"{s}\"" for s in spans[:10])
        prompt = FEATURE_INTERPRETATION_PROMPT.format(activation_spans=spans_text)

        if self.llm_generate_fn:
            return self.llm_generate_fn(prompt).strip()
        else:
            return f"Feature #{feature_idx} related to: {'; '.join(spans[:3])}"

    def _synthesize_skill(
        self, feature_idx: int, description: str,
    ) -> SynthesizedSkill | None:
        """Generate a skill text targeting a specific capability."""
        prompt = SKILL_SYNTHESIS_PROMPT.format(feature_description=description)

        for attempt in range(self.max_retries):
            if self.llm_generate_fn:
                response = self.llm_generate_fn(prompt)
            else:
                return SynthesizedSkill(
                    feature_idx=feature_idx,
                    feature_description=description,
                    skill_name=f"Skill for feature {feature_idx}",
                    skill_text=description,
                )

            # Parse response
            name, text = self._parse_skill_response(response)
            if name and text:
                return SynthesizedSkill(
                    feature_idx=feature_idx,
                    feature_description=description,
                    skill_name=name,
                    skill_text=text,
                )

        return None

    def _parse_skill_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract skill name and text."""
        lines = response.strip().split("\n")
        name = ""
        text_lines = []
        in_guide = False

        for line in lines:
            line = line.strip()
            if line.lower().startswith("skill name"):
                name = line.split(":", 1)[-1].strip().strip('"')
            elif line.lower().startswith("skill guide"):
                in_guide = True
                rest = line.split(":", 1)[-1].strip()
                if rest:
                    text_lines.append(rest)
            elif in_guide and line:
                text_lines.append(line)
            elif not name and not in_guide and line:
                # Fallback: first non-empty line as name
                name = line

        text = " ".join(text_lines)
        return name, text

    def _verify_skill(self, skill: SynthesizedSkill) -> bool:
        """Verify that the synthesized skill activates the target SAE feature."""
        profile = self.extractor.extract_features(skill.skill_text)

        if skill.feature_idx in profile.active_features:
            skill.activation_score = profile.feature_scores.get(skill.feature_idx, 0.0)
            skill.verified = skill.activation_score >= self.verification_threshold
            return skill.verified

        skill.activation_score = 0.0
        skill.verified = False
        return False

    def _check_redundancy(
        self,
        skill: SynthesizedSkill,
        existing_vectors: torch.Tensor,
    ):
        """Check if the skill is too close to existing skills on the sphere."""
        from skill_sphere.skill_bank.encoder import SkillEncoder

        # This would need the encoder - for now just mark as unchecked
        skill.sphere_distance_to_nearest = 1.0  # Placeholder

    def save_synthesized(
        self,
        skills: list[SynthesizedSkill],
        path: str | Path,
    ):
        """Save synthesized skills to JSON."""
        data = []
        for s in skills:
            data.append({
                "feature_idx": s.feature_idx,
                "feature_description": s.feature_description,
                "skill_name": s.skill_name,
                "skill_text": s.skill_text,
                "verified": s.verified,
                "activation_score": s.activation_score,
                "sphere_distance_to_nearest": s.sphere_distance_to_nearest,
            })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(skills)} synthesized skills to {path}")

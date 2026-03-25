"""FAC diagnostics: SAE-based feature analysis and targeted skill synthesis.

Modules:
  - sae: TopK Sparse Autoencoder loader (pretrained on Qwen2-7B Layer 14)
  - feature_extractor: Extract SAE features from transformer hidden states
  - fac: Feature Activation Coverage analysis (success vs failure diff)
  - targeted_synthesis: Synthesize skills for missing features
"""

from skill_sphere.diagnostics.sae import TopKSAE
from skill_sphere.diagnostics.feature_extractor import FeatureExtractor, FeatureProfile
from skill_sphere.diagnostics.fac import FACAnalyzer, FACResult

__all__ = [
    "TopKSAE",
    "FeatureExtractor",
    "FeatureProfile",
    "FACAnalyzer",
    "FACResult",
]

"""Extract SAE features from Qwen2.5-7B hidden states.

Two-phase process:
  1. Run text through Qwen2.5-7B → extract Layer 14 hidden states
  2. Encode hidden states through SAE → 65536-dim sparse features

Supports both:
  - Hook-based extraction (modifies model forward, memory efficient)
  - output_hidden_states extraction (simpler, uses more memory)
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from skill_sphere.diagnostics.sae import TopKSAE


@dataclass
class FeatureProfile:
    """Feature activation profile for a trajectory or text."""
    # Set of active feature indices across all tokens
    active_features: set[int] = field(default_factory=set)
    # Feature -> max activation score
    feature_scores: dict[int, float] = field(default_factory=dict)
    # Feature -> count of tokens that activated it
    feature_counts: dict[int, int] = field(default_factory=dict)
    # Total tokens analyzed
    total_tokens: int = 0
    # Metadata
    metadata: dict = field(default_factory=dict)

    def merge(self, other: "FeatureProfile") -> "FeatureProfile":
        """Merge two profiles (union of features, max of scores)."""
        merged = FeatureProfile()
        merged.active_features = self.active_features | other.active_features
        merged.total_tokens = self.total_tokens + other.total_tokens
        # Merge scores (take max)
        all_features = self.feature_scores.keys() | other.feature_scores.keys()
        for f in all_features:
            s1 = self.feature_scores.get(f, 0.0)
            s2 = other.feature_scores.get(f, 0.0)
            merged.feature_scores[f] = max(s1, s2)
            c1 = self.feature_counts.get(f, 0)
            c2 = other.feature_counts.get(f, 0)
            merged.feature_counts[f] = c1 + c2
        return merged


class FeatureExtractor:
    """Extract SAE features from transformer hidden states.

    Uses a pretrained TopKSAE on Layer 14 of Qwen2.5-7B to decompose
    hidden states into interpretable sparse features.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        sae: TopKSAE | None = None,
        sae_repo: str = "Zhongzhi1228/sae_qwen_l14_h65536",
        sae_filename: str = "TopK7_l14_h3584_epoch3.pth",
        layer: int = 14,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            model_name: HuggingFace model ID for Qwen2.5-7B.
            sae: Pre-loaded SAE, or None to load from HuggingFace.
            sae_repo: HuggingFace repo for pretrained SAE.
            sae_filename: SAE checkpoint filename.
            layer: Which transformer layer to extract from.
            device: Device for computation.
            dtype: Model dtype (bfloat16 recommended).
            max_seq_len: Maximum sequence length to process.
        """
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # Load SAE
        if sae is not None:
            self.sae = sae.to(device)
        else:
            print(f"Loading SAE from {sae_repo}...")
            self.sae = TopKSAE.from_huggingface(
                repo_id=sae_repo,
                filename=sae_filename,
                device=device,
            )
        self.sae.eval()

        # Model and tokenizer loaded lazily
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        """Lazy-load the transformer model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} for feature extraction...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"Model loaded on {self.device}")

    @torch.no_grad()
    def extract_hidden_states(
        self, text: str,
    ) -> Tensor:
        """Extract Layer-N hidden states from a single text.

        Args:
            text: Input text to process.

        Returns:
            (seq_len, hidden_dim) tensor of hidden states from target layer.
        """
        self._ensure_model()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.device)

        outputs = self._model(
            **inputs,
            output_hidden_states=True,
        )

        # hidden_states is a tuple of (n_layers + 1,) tensors
        # Index 0 = embedding, Index 1 = layer 0, ..., Index N+1 = layer N
        hidden = outputs.hidden_states[self.layer + 1]  # +1 for embedding layer
        return hidden.squeeze(0).float()  # (seq_len, hidden_dim)

    @torch.no_grad()
    def extract_features(
        self, text: str,
    ) -> FeatureProfile:
        """Extract SAE features from text.

        Full pipeline: text → tokenize → model → Layer 14 hidden → SAE encode → features.

        Args:
            text: Input text.

        Returns:
            FeatureProfile with active features and scores.
        """
        hidden = self.extract_hidden_states(text)  # (seq_len, hidden_dim)

        # Process in chunks to avoid OOM with large SAE
        chunk_size = 64
        profile = FeatureProfile(total_tokens=hidden.shape[0])

        for start in range(0, hidden.shape[0], chunk_size):
            chunk = hidden[start:start + chunk_size]  # (chunk, hidden_dim)
            active_list = self.sae.get_active_features(chunk)

            for token_features in active_list:
                for feat_idx, score in token_features:
                    profile.active_features.add(feat_idx)
                    if feat_idx not in profile.feature_scores or score > profile.feature_scores[feat_idx]:
                        profile.feature_scores[feat_idx] = score
                    profile.feature_counts[feat_idx] = profile.feature_counts.get(feat_idx, 0) + 1

        return profile

    @torch.no_grad()
    def extract_features_batch(
        self, texts: list[str],
        show_progress: bool = True,
    ) -> list[FeatureProfile]:
        """Extract SAE features from a batch of texts.

        Processes one text at a time to manage memory.

        Args:
            texts: List of input texts.
            show_progress: Print progress.

        Returns:
            List of FeatureProfile for each text.
        """
        profiles = []
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Extracting features: {i + 1}/{len(texts)}")
            profile = self.extract_features(text)
            profiles.append(profile)

            # Periodic GPU memory cleanup
            if (i + 1) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        return profiles

    def extract_trajectory_features(
        self, trajectory: list[dict],
    ) -> FeatureProfile:
        """Extract features from a full agent trajectory.

        A trajectory is a list of step dicts, each containing at least:
        - "prompt": the full prompt sent to the LLM
        - "response": the LLM's response

        Features are extracted from BOTH prompt+response (the full context
        the model processed), merged across all steps.

        Args:
            trajectory: List of step dicts with "prompt" and "response".

        Returns:
            Merged FeatureProfile across all steps.
        """
        merged = FeatureProfile()

        for step in trajectory:
            # Combine prompt and response as the model saw them
            text = step.get("prompt", "") + "\n" + step.get("response", "")
            step_profile = self.extract_features(text)
            step_profile.metadata = {
                "step": step.get("step", -1),
                "success": step.get("success", None),
            }
            merged = merged.merge(step_profile)

        return merged

    def unload_model(self):
        """Free GPU memory by unloading the transformer model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded, GPU memory freed")

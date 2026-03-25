"""TopK Sparse Autoencoder loader.

Implements the TopKSAE architecture from FAC-Synthesis to load the
pretrained SAE for Qwen2-7B (Layer 14, 65536 features, TopK=7).

Architecture: tied-weight SAE with TopK sparsity enforcement.
  encode: h = ReLU(TopK((x - b_dec) @ W_enc + b_enc))
  decode: x_hat = h @ W_enc.T + b_dec

Reference: "Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs"
Pretrained model: Zhongzhi1228/sae_qwen_l14_h65536
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


def _mask_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """Create binary mask keeping only top-k values per row."""
    _, idx = torch.topk(x, k=k, dim=-1)
    return torch.zeros_like(x).scatter_(-1, idx, 1.0)


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder with tied encoder-decoder weights.

    Matches the FAC-Synthesis architecture exactly so we can load
    their pretrained checkpoint.
    """

    def __init__(
        self,
        d_inp: int = 3584,
        d_hide: int = 65536,
        top_k: int = 7,
        device: str = "cuda",
    ):
        super().__init__()
        self.d_inp = d_inp
        self.d_hide = d_hide
        self.top_k = top_k

        # Tied weights: W_dec = W_enc.T
        self.W_enc = nn.Parameter(torch.empty(d_inp, d_hide, device=device))
        self.b_enc = nn.Parameter(torch.zeros(d_hide, device=device))
        self.b_dec = nn.Parameter(torch.zeros(d_inp, device=device))

        # Non-parameter buffers
        self.register_buffer("freq", torch.zeros(d_hide, device=device))
        self.register_buffer("mask", torch.ones(d_hide, device=device))

        nn.init.kaiming_uniform_(self.W_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations.

        Args:
            x: (batch, d_inp) hidden states from transformer layer.

        Returns:
            (batch, d_hide) sparse feature activations (top_k nonzero per row).
        """
        h = (x - self.b_dec) @ self.W_enc + self.b_enc
        mask = _mask_topk(h, self.top_k)
        return torch.relu(h * mask)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space.

        Args:
            h: (batch, d_hide) sparse feature activations.

        Returns:
            (batch, d_inp) reconstructed hidden states.
        """
        return h @ self.W_enc.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.

        Returns:
            (reconstructed, features) tuple.
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features

    def get_active_features(
        self, x: torch.Tensor, threshold: float = 0.0,
    ) -> list[list[tuple[int, float]]]:
        """Get active feature indices and scores for each input.

        Args:
            x: (batch, d_inp) hidden states.
            threshold: Minimum activation to count as active.

        Returns:
            List of [(feature_idx, activation_score), ...] per input.
        """
        features = self.encode(x)  # (batch, d_hide)
        result = []
        for row in features:
            active = (row > threshold).nonzero(as_tuple=True)[0]
            pairs = [(idx.item(), row[idx].item()) for idx in active]
            pairs.sort(key=lambda p: p[1], reverse=True)
            result.append(pairs)
        return result

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str = "cuda",
    ) -> "TopKSAE":
        """Load pretrained SAE from checkpoint file.

        Compatible with FAC-Synthesis checkpoint format:
        {
            "weight": state_dict,
            "config": {"d_inp": 3584, "d_hide": 65536, "topK": 7}
        }
        """
        states = torch.load(path, map_location=device, weights_only=False)

        # Parse config
        if "config" in states:
            cfg = states["config"]
            d_inp = cfg.get("d_inp", 3584)
            d_hide = cfg.get("d_hide", 65536)
            top_k = cfg.get("topK", 7)
        else:
            # Infer from filename: TopK7_l14_h3584_epoch3.pth
            name = Path(path).stem
            d_inp = 3584
            d_hide = 65536
            top_k = 7
            if "_h" in name:
                try:
                    d_inp = int(name.split("_h")[1].split("_")[0])
                except (ValueError, IndexError):
                    pass

        model = cls(d_inp=d_inp, d_hide=d_hide, top_k=top_k, device=device)

        # Load weights
        weight_dict = states.get("weight", states)
        # Handle potential key mismatches
        model_keys = set(model.state_dict().keys())
        load_keys = set(weight_dict.keys())

        # Filter to only matching keys
        filtered = {k: v for k, v in weight_dict.items() if k in model_keys}
        if filtered:
            model.load_state_dict(filtered, strict=False)
        else:
            # Try loading directly (old format)
            model.load_state_dict(weight_dict, strict=False)

        model.eval()
        return model

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str = "Zhongzhi1228/sae_qwen_l14_h65536",
        filename: str = "TopK7_l14_h3584_epoch3.pth",
        cache_dir: str | None = None,
        device: str = "cuda",
    ) -> "TopKSAE":
        """Download and load pretrained SAE from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID.
            filename: Checkpoint filename in the repo.
            cache_dir: Local cache directory.
            device: Target device.
        """
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        return cls.from_pretrained(path, device=device)

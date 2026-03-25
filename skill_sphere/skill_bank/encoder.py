"""Skill text encoder: maps skill descriptions to unit vectors on the sphere.

Uses Qwen3-Embedding-0.6B (1024-dim) by default, with L2 normalization
to project onto the unit hypersphere.
"""

import torch
from torch import Tensor

from skill_sphere.geometry.sphere import l2_normalize

# Lazy-loaded globals
_model = None
_tokenizer = None
_device = None


def _get_model(
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "cuda",
):
    """Lazy-load the embedding model."""
    global _model, _tokenizer, _device

    if _model is not None and _device == device:
        return _model, _tokenizer

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name, device=device)
        _tokenizer = None  # sentence-transformers handles tokenization
        _device = device
        return _model, _tokenizer
    except ImportError:
        # Fallback to transformers
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name).to(device)
        _model.eval()
        _device = device
        return _model, _tokenizer


class SkillEncoder:
    """Encodes skill text descriptions into unit vectors on the sphere."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        instruction: str = "Given a skill description, represent it for retrieval",
        dim: int | None = None,
    ):
        """
        Args:
            model_name: HuggingFace model name for the embedding model.
            device: Torch device.
            instruction: Task instruction prefix for the embedding model.
            dim: Output dimension. If None, uses the model's native dimension (1024).
                 Qwen3-Embedding supports MRL dimensions: 32, 64, 128, 256, 512, 1024.
        """
        self.model_name = model_name
        self.device = device
        self.instruction = instruction
        self.dim = dim
        self._use_sentence_transformers = True

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=device)
        except ImportError:
            self._use_sentence_transformers = False
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()

    @torch.no_grad()
    def encode(self, texts: str | list[str]) -> Tensor:
        """Encode skill text(s) into unit vectors on the sphere.

        Args:
            texts: Single skill text or list of skill texts.

        Returns:
            (N, D) tensor of unit vectors on S^{d-1}, where D is the
            embedding dimension and N is the number of input texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        if self._use_sentence_transformers:
            # sentence-transformers handles instruction prefix and pooling
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                prompt=self.instruction,
                convert_to_tensor=True,
            )
            # Ensure float32 for consistent downstream operations
            embeddings = embeddings.float()
            if self.dim is not None and embeddings.shape[-1] != self.dim:
                embeddings = embeddings[..., : self.dim]
                embeddings = l2_normalize(embeddings)
            return embeddings
        else:
            # Manual encoding with transformers
            prefixed = [f"Instruct: {self.instruction}\nQuery: {t}" for t in texts]
            inputs = self.tokenizer(
                prefixed,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            # Last-token pooling (Qwen3-Embedding convention)
            embeddings = outputs.last_hidden_state[:, -1, :].float()

            if self.dim is not None:
                embeddings = embeddings[..., : self.dim]

            return l2_normalize(embeddings)

    @torch.no_grad()
    def encode_query(self, query_text: str) -> Tensor:
        """Encode a task description query into a unit vector.

        Uses a retrieval-oriented instruction prefix.

        Args:
            query_text: Task description to encode.

        Returns:
            (D,) unit vector on the sphere.
        """
        return self.encode(query_text).squeeze(0)

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        if self.dim is not None:
            return self.dim
        if self._use_sentence_transformers:
            return self.model.get_sentence_embedding_dimension()
        return self.model.config.hidden_size

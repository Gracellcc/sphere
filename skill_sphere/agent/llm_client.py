"""LLM client wrapper for agent inference.

Supports:
- Azure OpenAI API (GPT-4o, o3, etc.)
- Local vLLM server (Qwen2.5-7B, etc.) via OpenAI-compatible API
"""

from __future__ import annotations

import time
import logging

from openai import AzureOpenAI, OpenAI

log = logging.getLogger(__name__)


def _is_reasoning_model(model: str) -> bool:
    """Check if model is an o-series reasoning model (o1, o3, etc.).

    These models don't support temperature or max_tokens parameters;
    they use max_completion_tokens instead, and don't support system role.
    """
    name = model.lower()
    return any(name.startswith(prefix) for prefix in ("o1", "o3", "o4"))


def _uses_max_completion_tokens(model: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens.

    GPT-5.x and o-series models use max_completion_tokens.
    """
    if _is_reasoning_model(model):
        return True
    return model.lower().startswith("gpt-5")


class LLMClient:
    """Wrapper around OpenAI-compatible APIs for agent inference.

    Supports Azure OpenAI, local vLLM servers, and reasoning models (o3).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        backend: str = "azure",
        # Azure settings (reads from env vars, falls back to defaults)
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "2025-01-01-preview",
        # vLLM settings
        vllm_base_url: str = "http://localhost:8000/v1",
        # Qwen3 thinking mode control
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
    ):
        """
        Args:
            model: Model name (e.g., "gpt-4o", "o3", or "Qwen/Qwen2.5-7B-Instruct").
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (ignored for o-series models).
            backend: "azure" for Azure OpenAI, "vllm" for local vLLM server.
            vllm_base_url: Base URL for vLLM OpenAI-compatible server.
        """
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.backend = backend
        self.is_reasoning = _is_reasoning_model(model)
        self.use_max_completion_tokens = _uses_max_completion_tokens(model)
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        if backend == "azure":
            import os
            api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
            azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            if not api_key or not azure_endpoint:
                raise ValueError(
                    "Azure backend requires AZURE_OPENAI_API_KEY and "
                    "AZURE_OPENAI_ENDPOINT environment variables."
                )
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        elif backend == "vllm":
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=vllm_base_url,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'azure' or 'vllm'.")

    def generate(self, messages: list[dict], logprobs: bool = False, top_logprobs: int = 5) -> str:
        """Generate a response from the LLM.

        Args:
            messages: List of {"role": ..., "content": ...} message dicts.
            logprobs: If True, request log probabilities.
            top_logprobs: Number of top tokens to return logprobs for.

        Returns:
            The assistant's response text.
        """
        if self.is_reasoning:
            # O-series models (o1, o3, o4): no temperature, use max_completion_tokens,
            # no logprobs, no system messages (merge into first user message).
            merged_messages = self._merge_system_for_reasoning(messages)
            kwargs = dict(
                model=self.model,
                messages=merged_messages,
                max_completion_tokens=self.max_new_tokens,
            )
        elif self.use_max_completion_tokens:
            # GPT-5.x: supports system role and temperature, but uses
            # max_completion_tokens instead of max_tokens.
            kwargs = dict(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            if logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = top_logprobs
        else:
            kwargs = dict(
                model=self.model,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            if logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = top_logprobs

        # Qwen3 thinking mode control via vLLM chat_template_kwargs
        if self.backend == "vllm" and (self.enable_thinking is not None or self.thinking_budget is not None):
            chat_kwargs = {}
            if self.enable_thinking is not None:
                chat_kwargs["enable_thinking"] = self.enable_thinking
            if self.thinking_budget is not None:
                chat_kwargs["thinking_budget"] = self.thinking_budget
            kwargs["extra_body"] = {"chat_template_kwargs": chat_kwargs}

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._last_response = response  # Store for logprobs access
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                # Content filter — don't retry
                if any(kw in error_str for kw in ("content_filter", "content_management", "invalid_prompt", "violating our usage")):
                    self._last_response = None
                    return '<think>Content filter triggered.</think>\n<code>print("Content filter triggered, skipping step.")</code>'
                # Rate limit, timeout, server error — retry with backoff
                retryable = any(kw in error_str.lower() for kw in (
                    "429", "rate limit", "rate_limit", "too many requests",
                    "timeout", "timed out", "connection", "server error",
                    "500", "502", "503", "504", "overloaded",
                ))
                if retryable and attempt < max_retries:
                    wait = min(2 ** attempt * 5, 120)  # 5, 10, 20, 40, 80s
                    log.warning(f"API error (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {error_str[:200]}")
                    time.sleep(wait)
                    continue
                raise

    @staticmethod
    def _merge_system_for_reasoning(messages: list[dict]) -> list[dict]:
        """Merge system messages into the first user message for o-series models.

        O-series models don't support system role. We prepend system content
        to the first user message as a clearly delimited instruction block.
        """
        system_parts = []
        other_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                other_msgs.append(m)

        if not system_parts:
            return other_msgs

        system_block = "\n\n".join(system_parts)

        if other_msgs and other_msgs[0]["role"] == "user":
            merged = list(other_msgs)
            merged[0] = {
                "role": "user",
                "content": f"[INSTRUCTIONS]\n{system_block}\n[/INSTRUCTIONS]\n\n{other_msgs[0]['content']}",
            }
            return merged

        # No user message to merge into — create a user message from system content
        return [{"role": "user", "content": system_block}] + other_msgs

    def generate_with_logprobs(
        self,
        system_prompt: str,
        user_prompt: str,
        top_k: int = 5,
    ) -> tuple[str, list[list[tuple[str, float]]]]:
        """Generate action and return per-token top-k logprobs.

        Args:
            system_prompt: System-level instructions.
            user_prompt: Current observation and actions.
            top_k: Number of top tokens per position.

        Returns:
            (raw_output, token_logprobs) where token_logprobs is a list of
            [(token, logprob), ...] for each generated token.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.generate(messages, logprobs=True, top_logprobs=top_k)

        token_logprobs = []
        resp = getattr(self, "_last_response", None)
        if resp and resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            for token_info in resp.choices[0].logprobs.content:
                top_tokens = []
                if token_info.top_logprobs:
                    for tlp in token_info.top_logprobs:
                        top_tokens.append((tlp.token, tlp.logprob))
                token_logprobs.append(top_tokens)

        return text, token_logprobs

    def generate_action(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate an action given system and user prompts.

        Args:
            system_prompt: System-level instructions (includes skill guidance).
            user_prompt: Current observation and admissible actions.

        Returns:
            Raw model output (should contain <think> and <action> tags).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate(messages)

    def generate_chat(self, messages: list[dict]) -> str:
        """Generate from a full multi-turn chat message list."""
        return self.generate(messages)

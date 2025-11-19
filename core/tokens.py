from __future__ import annotations

import threading
from dataclasses import dataclass

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# Centralized token counting using the Qwen3-VL-32B-Instruct tokenizer for better alignment.
# This module is intentionally opinionated: if the tokenizer cannot be initialized,
# an exception is raised immediately rather than silently degrading behavior.


@dataclass
class _TokenizerHolder:
    tokenizer: PreTrainedTokenizerBase | None = None


_tokenizer_holder = _TokenizerHolder()
_tokenizer_lock = threading.Lock()


def _get_tokenizer() -> PreTrainedTokenizerBase:
    """
    Lazily construct and return a shared Qwen3-VL tokenizer instance.

    Raises:
        RuntimeError: if the tokenizer cannot be created for any reason.
    """
    if _tokenizer_holder.tokenizer is not None:
        return _tokenizer_holder.tokenizer

    with _tokenizer_lock:
        if _tokenizer_holder.tokenizer is not None:
            return _tokenizer_holder.tokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-VL-32B-Instruct",
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize tokenizer for 'Qwen/Qwen3-VL-32B-Instruct'. "
                "Ensure that the appropriate model files are available and that "
                "the 'transformers' library is up to date."
            ) from exc

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(
                f"Expected a PreTrainedTokenizerBase instance, got {type(tokenizer).__name__}."
            )

        _tokenizer_holder.tokenizer = tokenizer
        return tokenizer


def count_tokens(text: str) -> int:
    """
    Count tokens in `text` using the shared Qwen3-VL tokenizer instance.

    Raises:
        RuntimeError / TypeError: if the tokenizer cannot be initialized correctly.
    """
    tokenizer = _get_tokenizer()
    # The encode method returns a sequence of token ids; we only care about its length.
    token_ids = tokenizer.encode(text)
    return len(token_ids)

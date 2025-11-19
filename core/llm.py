# core/llm.py

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias, Union

import httpx

from core.tokens import count_tokens


# --- Type aliases ---

ModelName: TypeAlias = Literal["qwen3-vl:32b-instruct", "qwen3-vl:32b-thinking"]
MessageRole: TypeAlias = Literal["system", "user", "assistant"]
NumericOption = int | float
OllamaOptionValue = NumericOption | bool
OllamaOptions = dict[str, OllamaOptionValue]


# --- Message types for requests ---

@dataclass
class ChatMessageBase:
    role: MessageRole
    content: str


@dataclass
class ChatTextMessage(ChatMessageBase):
    """
    Plain text-only chat message.
    """
    pass


@dataclass
class ChatVisionMessage(ChatMessageBase):
    """
    Vision-capable chat message that carries base64-encoded image content.
    """
    images_b64: list[str]


ChatMessage = Union[ChatTextMessage, ChatVisionMessage]


# --- Configuration & helpers ---

@dataclass
class OllamaConnectionConfig:
    """
    Connection details and HTTP timeouts for an Ollama-compatible endpoint.

    base_url:
        The full base URL of the Ollama server, including scheme and host,
        e.g. "http://127.0.0.1:11434" or "https://ollama.internal:11434".
    """
    base_url: str = "http://127.0.0.1:11434"
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 300.0  # 5 minutes
    write_timeout_seconds: float = 10.0

    def normalized_base_url(self) -> str:
        """
        Strictly validate and normalize the base URL.

        Raises:
            ValueError: if the URL is empty or does not start with http/https.
        """
        url = self.base_url.strip().rstrip("/")
        if not url:
            raise ValueError("OllamaConnectionConfig.base_url must be a non-empty string.")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(
                f"OllamaConnectionConfig.base_url must start with 'http://' or 'https://', "
                f"got: {self.base_url!r}"
            )
        return url


def get_client(config: OllamaConnectionConfig) -> httpx.Client:
    """
    Returns a configured httpx.Client. Keeping the name for backward compatibility.
    """
    normalized_url = config.normalized_base_url()
    try:
        client = httpx.Client(
            timeout=httpx.Timeout(
                connect=config.connect_timeout_seconds,
                read=config.read_timeout_seconds,
                write=config.write_timeout_seconds,
                pool=None,
            )
        )
        return client
    except Exception as exc:
        raise RuntimeError(f"Error initializing httpx client for Ollama at {normalized_url}: {exc}") from exc


# --- Public API compatibility layer ---

@dataclass
class _Message:
    role: str = "assistant"
    content: str = ""
    thinking: Optional[str] = None  # extracted if available (or from <think>...</think>)


@dataclass
class ChatResponse:
    """
    Drop-in stand-in for `ollama.ChatResponse` attributes your code uses.
    """
    message: _Message
    # The following names match your print_stats() expectations
    prompt_eval_duration: Optional[int] = None   # nanoseconds
    prompt_eval_count: Optional[int] = None
    eval_duration: Optional[int] = None          # nanoseconds (includes think+response)
    eval_count: Optional[int] = None             # tokens generated (incl. think where applicable)
    ran_out_of_tokens: bool = False


# Advanced parameters for qwen3-vl:32b (applied on every request).

_QWEN3_VL_32B_BASE_OPTIONS: OllamaOptions = {
    # Good context length value for 32GB VRAM and flash attention enabled
    # Ends up using ~31 GB in "ollama ps" when context length is full.
    "num_ctx": 19456,  # 19k

    # Setting -1 (infinite) would cause infinite generation once in a while.
    # Infinite generations are observed to be exactly 239,998 thinking tokens
    # plus 2 response tokens.
    # Avoid the issue of Ollama call getting stuck waiting for almost 2 hours,
    # grinding the GPU for nothing on gibberish.
    "num_predict": 11264,

    # Layers to offload, all of them.
    "num_gpu": 65,
}

# Recommended: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct#text
_QWEN3_VL_32B_INSTRUCT_TEXT_ONLY_OPTIONS: OllamaOptions = {
    **_QWEN3_VL_32B_BASE_OPTIONS,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 40,
    "min_p": 0.0,
    "presence_penalty": 2.0,
    "repeat_penalty": 1.0,
}

# Recommended: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct#vl
_QWEN3_VL_32B_INSTRUCT_VL_OPTIONS: OllamaOptions = {
    **_QWEN3_VL_32B_BASE_OPTIONS,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
}

# Recommended: https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking#text
_QWEN3_VL_32B_THINKING_TEXT_ONLY_OPTIONS: OllamaOptions = {
    **_QWEN3_VL_32B_BASE_OPTIONS,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
}

# Recommended: https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking#vl
_QWEN3_VL_32B_THINKING_VL_OPTIONS: OllamaOptions = {
    **_QWEN3_VL_32B_BASE_OPTIONS,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}


def get_ollama_options(model: ModelName, please_no_thinking: bool, has_images: bool) -> OllamaOptions:
    """
    Return per-model advanced options for Qwen3-VL 32B.

    `please_no_thinking` is accepted for API compatibility but does not affect
    the Qwen3-VL configuration – use the explicit `model` flag to pick
    between the Instruct and Thinking variants.
    """
    # Validate that boolean flag is actually a bool (and not e.g. an int masquerading as one).
    if not isinstance(please_no_thinking, bool):
        raise TypeError(f"please_no_thinking must be of type bool, got {type(please_no_thinking).__name__}.")

    if model == "qwen3-vl:32b-thinking":
        return dict(_QWEN3_VL_32B_THINKING_VL_OPTIONS) if has_images else dict(_QWEN3_VL_32B_THINKING_TEXT_ONLY_OPTIONS)
    if model == "qwen3-vl:32b-instruct":
        return dict(_QWEN3_VL_32B_INSTRUCT_VL_OPTIONS) if has_images else dict(_QWEN3_VL_32B_INSTRUCT_TEXT_ONLY_OPTIONS)
    raise ValueError(
        f"Unrecognized Qwen3-VL model '{model}'. Expected one of: 'qwen3-vl:32b-instruct', 'qwen3-vl:32b-thinking'."
    )


def _supports_thinking(model: ModelName) -> bool:
    return model in {
        "qwen3-vl:32b-thinking",
    }


def _supports_qwen3_hybrid(model: ModelName) -> bool:
    # Qwen3-VL does not use the hybrid "no_think" system prompt trick;
    # keep the helper for structural compatibility, but always return False.
    return False


def _extract_thinking(message_obj: dict[str, object], can_think: bool, content: str) -> Optional[str]:
    """
    Extract a 'thinking' trace if present. Ollama sometimes emits it as a separate
    field, or inline inside <think>...</think> tags. We support both.
    """
    # 1) explicit field (future / variant-friendly)
    if "thinking" in message_obj:
        raw_thinking = message_obj["thinking"]
        if isinstance(raw_thinking, str):
            return raw_thinking

    # 2) inline tags (case-insensitive search, but preserve original text)
    if can_think and content:
        lower = content.lower()
        start_tag = "<think>"
        end_tag = "</think>"
        start_idx = lower.find(start_tag)
        if start_idx != -1:
            end_idx = lower.find(end_tag, start_idx + len(start_tag))
            if end_idx != -1:
                inner_start = start_idx + len(start_tag)
                inner_content = content[inner_start:end_idx].strip()
                if inner_content:
                    return inner_content

    return None


def _extract_optional_int_field(payload: dict[str, object], field_name: str) -> Optional[int]:
    """
    Helper to strictly extract an optional integer field from a JSON-like mapping.
    """
    val = payload.get(field_name)
    if val is None:
        return None
    if isinstance(val, int):
        return val
    raise TypeError(
        f"Expected JSON field '{field_name}' to be an int or null, "
        f"but got {type(val).__name__} with value {val!r}."
    )


@dataclass
class ChatCompleteParams:
    """
    Aggregated parameters for a single non-streaming chat completion call.

    This replaces a long positional-argument list with a strongly typed
    configuration object.
    """
    messages: list[ChatMessage]
    model: ModelName
    client: httpx.Client
    connection: OllamaConnectionConfig
    max_completion_tokens: int
    please_no_thinking: bool
    require_json: bool

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ChatCompleteParams.messages must contain at least one message.")
        if self.max_completion_tokens <= 0:
            raise ValueError(
                f"ChatCompleteParams.max_completion_tokens must be a positive integer, "
                f"got {self.max_completion_tokens}."
            )
        if self.client is None:
            raise ValueError("ChatCompleteParams.client must not be None.")


def chat_complete(params: ChatCompleteParams) -> ChatResponse:
    """
    Non-streaming call to Ollama's /api/chat using httpx, with hard read timeout.
    Not using Ollama's official Python client because it doesn't provide a timeout option.
    Single entrypoint so callers do not duplicate flags:
    - thinking is enabled for thinking-capable models only
    - when not thinking and JSON is desired, we set format="json"
    - model selection is explicit via ChatCompleteParams.model (no environment indirection)
    """
    # Validate client again defensively.
    if params.client is None:
        raise RuntimeError("httpx client is not initialized")

    can_think = _supports_thinking(params.model)
    is_hybrid = _supports_qwen3_hybrid(params.model)

    # Detect whether any message carries images (Qwen3-VL vision path).
    has_images = any(
        isinstance(m, ChatVisionMessage) and len(m.images_b64) > 0
        for m in params.messages
    )

    options = get_ollama_options(params.model, params.please_no_thinking, has_images=has_images)

    hybrid_nothink_switch = is_hybrid and params.please_no_thinking

    # For Qwen3-VL we do not currently exercise hybrid behavior, but we keep
    # the original control flow for structural compatibility.
    if hybrid_nothink_switch:
        first_message = params.messages[0]
        if not isinstance(first_message, ChatTextMessage):
            raise TypeError("Hybrid '/no_think' mode requires the first message to be text-only.")
        first_message.content = f"/no_think {first_message.content}"

    if hybrid_nothink_switch or not can_think:
        options["num_predict"] = params.max_completion_tokens

    # Build payload messages from the strongly typed message objects.
    payload_messages: list[dict[str, object]] = []
    for msg in params.messages:
        base_msg: dict[str, object] = {
            "role": msg.role,
            "content": msg.content,
        }
        if isinstance(msg, ChatVisionMessage):
            if not msg.images_b64:
                raise ValueError("ChatVisionMessage.images_b64 must not be empty when used with vision models.")
            if "qwen3-vl" in params.model:
                base_msg["images"] = msg.images_b64
            else:
                base_msg["images_b64"] = msg.images_b64
        payload_messages.append(base_msg)

    payload: dict[str, object] = {
        "model": params.model,
        "messages": copy.deepcopy(payload_messages),
        "options": options,
        "stream": False,
        "think": can_think and not hybrid_nothink_switch,
    }

    # Strict JSON output enforced by Ollama doesn't work together with "<think>" tags.
    if params.require_json and not can_think:
        payload["format"] = "json"

    url = f"{params.connection.normalized_base_url()}/api/chat"

    # Make the request; httpx read-timeout caps total wait for the non-streaming body.
    # If Ollama wedges without sending bytes, you get a TimeoutException instead of a forever hang.
    resp = params.client.post(url, json=payload)
    resp.raise_for_status()
    data_raw = resp.json()

    if not isinstance(data_raw, dict):
        raise TypeError(f"Unexpected response payload type: expected dict, got {type(data_raw).__name__}.")

    # Build a compatibility response
    msg_raw = data_raw.get("message")
    if not isinstance(msg_raw, dict):
        raise TypeError("Unexpected response structure: 'message' field must be a JSON object.")

    content_raw = msg_raw.get("content")
    content = content_raw if isinstance(content_raw, str) else ""

    thinking_text = _extract_thinking(msg_raw, can_think, content)

    # Stats: use Ollama JSON keys if present; set None otherwise
    # Units from Ollama are nanoseconds for *_duration fields.
    prompt_eval_duration = _extract_optional_int_field(data_raw, "prompt_eval_duration")
    prompt_eval_count = _extract_optional_int_field(data_raw, "prompt_eval_count")
    eval_duration = _extract_optional_int_field(data_raw, "eval_duration")
    eval_count = _extract_optional_int_field(data_raw, "eval_count")

    done_reason_val = data_raw.get("done_reason")
    if done_reason_val is None:
        done_reason = ""
    elif isinstance(done_reason_val, str):
        done_reason = done_reason_val
    else:
        raise TypeError(
            f"Expected JSON field 'done_reason' to be a string or null, "
            f"but got {type(done_reason_val).__name__} with value {done_reason_val!r}."
        )

    role_raw = msg_raw.get("role")
    role_value = role_raw if isinstance(role_raw, str) else "assistant"

    return ChatResponse(
        message=_Message(
            role=role_value,
            content=content,
            thinking=thinking_text,
        ),
        prompt_eval_duration=prompt_eval_duration,
        prompt_eval_count=prompt_eval_count,
        eval_duration=eval_duration,
        eval_count=eval_count,
        ran_out_of_tokens=(done_reason.lower() == "length"),
    )


# For debug statistics
def print_stats(response: ChatResponse) -> Optional[str]:
    if None in [response.prompt_eval_duration, response.prompt_eval_count,
                response.eval_duration, response.eval_count]:
        return None
    try:
        prefill_speed = response.prompt_eval_count / (response.prompt_eval_duration / 1e9)  # type: ignore[operator]
        generation_speed = response.eval_count / (response.eval_duration / 1e9)  # type: ignore[operator]
    except ZeroDivisionError:
        prefill_speed = 0.0
        generation_speed = 0.0

    thinking_text = response.message.thinking or ""
    thinking_tokens = count_tokens(thinking_text) if thinking_text else 0
    if response.eval_count is None:
        # This should not happen if the earlier None-check passes; keep a guard in case
        # future changes alter that behavior.
        raise ValueError("ChatResponse.eval_count is unexpectedly None in print_stats().")

    return (
        f"prefill_speed: {prefill_speed:.2f}(tok/sec), "
        f"generation_speed: {generation_speed:.2f}(tok/sec)"
        + f"\nprompt: {response.prompt_eval_count}(tok), "
        f"think: {thinking_tokens}(tok) + response: {response.eval_count - thinking_tokens}(tok)"
    )

"""
llm.py - The only file that talks to the OpenAI API.

It wraps the two models allowed by the hackathon rules:
  * text-embedding-3-small  -> embed_texts() / embed_query()
  * gpt-4o-mini             -> chat() / chat_stream()

Every call is priced and logged by tracking.log_cost().
Question embeddings are cached on disk (data/query_cache.json), so asking the
same question twice (UI, eval runner, retrieval checker) is paid only once.
"""

import json
import threading

import numpy as np

from app.config import CHAT_MODEL, EMBED_MODEL, OPENAI_API_KEY, QUERY_CACHE_FILE, TEMPERATURE, has_api_key
from app.tracking import log_cost

_client = None
_cache_lock = threading.Lock()
_query_cache: dict | None = None


def friendly_error(exc: Exception) -> str:
    """Turn an OpenAI / network exception into a short message a demo audience understands."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return "The OpenAI key in .env is invalid or expired."
    if name == "RateLimitError":
        return "OpenAI rate limit or budget reached - wait a few seconds and ask again."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "No connection to OpenAI (network problem). Cached answers still work."
    if name in ("InternalServerError", "ServiceUnavailableError"):
        return "OpenAI is temporarily unavailable - please try again."
    return f"{name}: {exc}"


def get_client():
    """Create the OpenAI client once. Returns None in offline mode (no key)."""
    global _client
    if _client is None and has_api_key():
        from app.tracing import openai_class
        OpenAI = openai_class()                 # traced by Langfuse when it is configured
        # max_retries: the openai library automatically retries rate-limit (429), server (5xx)
        # and connection errors with exponential back-off -> a network hiccup during the demo
        # does not break the answer. timeout: never wait forever.
        _client = OpenAI(api_key=OPENAI_API_KEY, max_retries=4, timeout=60)
    return _client


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str], operation: str = "embed_chunks") -> np.ndarray:
    """Embed a list of texts in ONE API call. Returns a (n, 1536) float32 array."""
    client = get_client()
    if client is None:
        raise RuntimeError("No OPENAI_API_KEY configured - cannot create embeddings.")
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    log_cost(operation, EMBED_MODEL, response.usage.prompt_tokens)
    return np.array([d.embedding for d in response.data], dtype="float32")


def _load_query_cache() -> dict:
    global _query_cache
    if _query_cache is None:
        _query_cache = {}
        if QUERY_CACHE_FILE.exists():
            try:
                _query_cache = json.loads(QUERY_CACHE_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                _query_cache = {}
    return _query_cache


def embed_query(question: str) -> np.ndarray:
    """Embed one question, using the on-disk cache when possible. Returns shape (1, 1536)."""
    key = " ".join(question.lower().split())
    with _cache_lock:
        cache = _load_query_cache()
        if key in cache:
            return np.array([cache[key]], dtype="float32")
    vector = embed_texts([question], operation="embed_query")
    with _cache_lock:
        cache[key] = [round(float(x), 6) for x in vector[0]]
        QUERY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        QUERY_CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return vector


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
# Running totals of chat() calls in this process (the agent reads them to count model calls)
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def chat(messages: list[dict], operation: str = "chat", temperature: float = TEMPERATURE,
         json_mode: bool = False, max_tokens: int | None = None) -> str:
    """One gpt-4o-mini call. Returns the text of the answer."""
    client = get_client()
    if client is None:
        raise RuntimeError("No OPENAI_API_KEY configured - cannot call the chat model.")
    kwargs = {"model": CHAT_MODEL, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    usage = response.usage
    log_cost(operation, CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens)
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += usage.prompt_tokens
    USAGE["completion_tokens"] += usage.completion_tokens
    return (response.choices[0].message.content or "").strip()


def chat_stream(messages: list[dict], operation: str = "answer", temperature: float = TEMPERATURE):
    """
    Streaming version of chat(): yields the answer piece by piece so the UI
    can display it while it is being written. The cost is logged at the end.
    """
    client = get_client()
    if client is None:
        raise RuntimeError("No OPENAI_API_KEY configured - cannot call the chat model.")
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},   # last chunk carries token counts
    )
    usage = None
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
    if usage is not None:
        log_cost(operation, CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens)


def chat_with_tools(messages: list[dict], tools: list[dict], operation: str = "answer_tools",
                    temperature: float = TEMPERATURE) -> dict:
    """
    One gpt-4o-mini call where the model MAY ask to run tools (function calling).
    Returns {"content": str|None, "tool_calls": [{"id", "name", "arguments"}], "raw": message}.
    """
    client = get_client()
    if client is None:
        raise RuntimeError("No OPENAI_API_KEY configured - cannot call the chat model.")
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=tools,
                                              tool_choice="auto", temperature=temperature)
    usage = response.usage
    log_cost(operation, CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens)
    msg = response.choices[0].message
    calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
             for c in (msg.tool_calls or [])]
    return {"content": msg.content, "tool_calls": calls}

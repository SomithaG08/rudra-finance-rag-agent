"""
tracing.py - Optional Langfuse tracing (RAG + agent "Observability" extension).

OFF unless the three keys are in .env AND the `langfuse` package is installed:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_BASE_URL=https://cloud.langfuse.com

When on:
  * app/llm.py uses `from langfuse.openai import OpenAI`, so every OpenAI call (embedding,
    answer, judge, agent step) is recorded with its input, output, tokens, latency and cost;
  * functions decorated with @tracing.observe (retrieve, rerank, the RAG answer, the agent run)
    become nested steps, so one question = one trace.
When off, observe() returns the function unchanged: zero cost, zero dependency.

Warning (course spec): prompts, including corpus text, are sent to Langfuse Cloud.
"""

import os

_ENABLED = None


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        import app.config  # noqa: F401  (loads .env)
        keys = all(os.getenv(k) for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"))
        try:
            import langfuse  # noqa: F401
            _ENABLED = keys
        except Exception:
            _ENABLED = False
    return _ENABLED


def observe(name: str | None = None, **kwargs):
    """@observe() from langfuse when tracing is on, otherwise a do-nothing decorator."""
    def deco(fn):
        if not enabled():
            return fn
        from langfuse import observe as lf_observe
        return lf_observe(name=name or fn.__name__, **kwargs)(fn)
    return deco


def openai_class():
    """The OpenAI client class to use (traced when Langfuse is on)."""
    if enabled():
        from langfuse.openai import OpenAI
        return OpenAI
    from openai import OpenAI
    return OpenAI


def flush() -> None:
    """Call at the end of short scripts, or the last traces are lost."""
    if enabled():
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception:
            pass

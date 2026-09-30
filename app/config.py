"""
config.py - One place for every setting of the RAG application.

Every other module imports its paths, model names and tuning knobs from here,
so a change (e.g. a new top-k) is made once and used everywhere.

All paths are built from the repository root (the folder that contains app/),
NOT from the folder you launch Python from. This fixes the Day-1 bug where
"data/chunks.json" was only found if you started the app from the repo root.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent      # .../personal-finance-corpus
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"                      # raw PDFs (used for clickable citations)
PROMPTS_DIR = ROOT_DIR / "prompts"                      # versioned system prompts

CORPUS_FILE = DATA_DIR / "corpus.json"                  # output of scripts/extract.py
CHUNKS_FILE = DATA_DIR / "chunks.json"                  # output of app/build_index.py
INDEX_FILE = DATA_DIR / "my_index.faiss"                # output of app/build_index.py
BUILD_INFO_FILE = DATA_DIR / "build_info.json"          # how the index was built
EMBED_CACHE_FILE = DATA_DIR / "embedding_cache.npz"     # text-hash -> vector (avoid re-paying)
QUERY_CACHE_FILE = DATA_DIR / "query_cache.json"        # question -> vector (avoid re-paying)

QUESTIONS_FILE = DATA_DIR / "questions.json"            # Assignment 1 evaluation set
EVAL_RESULTS_FILE = DATA_DIR / "eval_results.json"      # latest eval run
EVAL_HISTORY_DIR = DATA_DIR / "eval_history"            # every eval run (regression checks)

COST_LOG_FILE = DATA_DIR / "cost_log.json"              # every OpenAI call and its price
QUERY_LOG_FILE = DATA_DIR / "query_log.jsonl"           # every question asked in the UI
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"             # thumbs up / down from the UI
EXPORT_DIR = DATA_DIR / "exports"                       # exported conversations

# Load OPENAI_API_KEY from <repo>/.env (works from any working directory)
load_dotenv(ROOT_DIR / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Models (the hackathon rules allow ONLY these two)
# ---------------------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Prices in USD per 1 million tokens (used by the cost tracker)
PRICES = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# ---------------------------------------------------------------------------
# Chunking (used by app/build_index.py)
# ---------------------------------------------------------------------------
CHUNK_STRATEGY = "sentence"      # "sentence" (new) or "fixed" (Day-1 behaviour)
CHUNK_TOKENS = 300               # target size of one chunk, in tokens (~1,200 characters)
CHUNK_OVERLAP_TOKENS = 60        # tokens repeated between neighbouring chunks
FIXED_CHUNK_CHARS = 500          # Day-1 fixed-size chunking (kept for comparison)
FIXED_OVERLAP_CHARS = 100
EMBED_BATCH_SIZE = 100           # chunks per embeddings API call
MIN_PAGE_CHARS = 50              # pages shorter than this are not chunked

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 8                # passages sent to the LLM (and shown as sources)
CANDIDATE_K = 40                 # candidates taken from EACH retriever before fusion
DEFAULT_SEARCH_MODE = "hybrid"   # "hybrid" (BM25 + vectors), "dense" (vectors only), "bm25"
RRF_K = 60                       # constant of Reciprocal Rank Fusion (standard value)
MAX_CHUNKS_PER_PAGE = 2          # diversity: never fill the top-k with one single page
CONFIDENCE_THRESHOLD = 0.40      # top vector similarity below this = out of scope
                                 # (measured on our eval set: off-topic <= 0.36, finance >= 0.46)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
PROMPT_VERSION = "v4"            # which file in prompts/ is the system prompt
TEMPERATURE = 0.0                # deterministic, factual answers
MAX_HISTORY_TURNS = 3            # how many previous Q/A pairs the model sees (memory)
REFUSAL_TEXT = "I don't have enough information to answer this question."

# Human-friendly titles for the documents in data/sources (shown in citations).
# Any file not listed here gets a title derived from its file name.
SOURCE_TITLES = {
    "principles-finance_-_web.pdf": "Principles of Finance (OpenStax, 2022)",
    "principles_of_finance-web.pdf": "Principles of Finance (OpenStax, 2022)",
}


def has_api_key() -> bool:
    """True when an OpenAI key is configured. Without it the app runs in offline mode."""
    return OPENAI_API_KEY.startswith("sk-") and len(OPENAI_API_KEY) > 20


def source_title(source: str) -> str:
    """Return a readable title for a source file name."""
    if not source:
        return "Unknown source"
    key = source.lower()
    if key in SOURCE_TITLES:
        return SOURCE_TITLES[key]
    stem = Path(source).stem.replace("_", " ").replace("-", " ")
    return " ".join(stem.split()).title()

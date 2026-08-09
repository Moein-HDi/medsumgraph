"""Central configuration for the MedSumGraph reimplementation.

Hyperparameters follow the paper (Appendix A, Table 6):
  temperature=0.5, top-k=50, top-p=0.95, repetition_penalty=1.1, max_tokens=512
  dynamic few-shot top-k=5, ensembling=5, reranker top-k=20
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = 0.5
TOP_P = 0.95
MAX_TOKENS = 512
# Groq free tier rate limits can be hit; retry with backoff up to this many times
LLM_MAX_RETRIES = 6
LLM_RETRY_BASE_DELAY = 3.0

# ---------------------------------------------------------------------------
# Embeddings (re-ranking + few-shot retrieval)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_TOP_K = 20        # triples kept after re-ranking
FEWSHOT_TOP_K = 5        # labeled examples retrieved per query
ENSEMBLE_SIZE = 5        # LLM samples per question, majority vote

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
KG_CACHE_DIR = CACHE_DIR / "kg_entities"
EMBED_CACHE_DIR = CACHE_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPH_PATH = CACHE_DIR / "medsumgraph.json"

for _d in (DATA_DIR, CACHE_DIR, KG_CACHE_DIR, EMBED_CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# UMLS
# ---------------------------------------------------------------------------
# Directory produced by the UMLS Metathesaurus install script.
# The script writes extracted RRF files either under
#   <install>/2025AB/META/   (when extracting into the MMSYS tree) or
#   <install>/Extract/META/  (custom extract dir).
# Set UMLS_META_DIR explicitly if auto-detection fails.
UMLS_META_DIR = os.getenv("UMLS_META_DIR", "")

# How many entities to process when building the KG (None = all)
KG_ENTITY_LIMIT = None

# Wikipedia summary length cap (words) before LLM summarization
WIKIPEDIA_MAX_WORDS = 400

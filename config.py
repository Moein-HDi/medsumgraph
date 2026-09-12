"""Central configuration for the MedSumGraph reimplementation.

Hyperparameters follow the paper (Appendix A, Table 6):
  temperature=0.5, top-k=50, top-p=0.95, repetition_penalty=1.1, max_tokens=512
  dynamic few-shot top-k=5, ensembling=5, reranker top-k=20
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Fully-local operation: datasets/sentence-transformers ping the Hub even
# when everything is cached. This silences the unauthenticated-request
# warning and prevents any network call in the local path.
# os.environ.setdefault("HF_HUB_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# LLM (OpenRouter — single provider)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Any model available on OpenRouter. Used for QA (Phase 2) and as the
# default for Phase 1. The paper used Llama 3.1 70B; cheaper/smaller
# models extract triples fine.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
# Separate (cheaper) model for KG construction (Phase 1). Falls back to
# OPENROUTER_MODEL if unset. Triple extraction is a simple structured task
# — a 8B-20B model is usually enough and ~5-10x cheaper than 70B.
KG_LLM_MODEL = os.getenv("KG_LLM_MODEL", "") or OPENROUTER_MODEL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TEMPERATURE = 0.5
TOP_P = 0.95
MAX_TOKENS = 512
# OpenRouter routes to many upstreams; on rate limit / 429 retry with backoff.
LLM_MAX_RETRIES = 50
LLM_RETRY_BASE_DELAY = 3.0
# Per-request timeout (seconds). Without one, hung requests block the
# pipeline indefinitely; the retry loop above recovers from timeouts.
LLM_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Embeddings (re-ranking + few-shot retrieval)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_TOP_K = 20        # triples kept after re-ranking
FEWSHOT_TOP_K = 5        # labeled examples retrieved per query
ENSEMBLE_SIZE = 1        # LLM samples per question, majority vote

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
KG_CACHE_DIR = CACHE_DIR / "kg_entities"
EMBED_CACHE_DIR = CACHE_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPH_PATH = CACHE_DIR / "medsumgraph.json"
SCOPE_CACHE_DIR = CACHE_DIR / "medqa_scope"
SCOPE_ENTITIES_PATH = SCOPE_CACHE_DIR / "scope_entities.json"

for _d in (DATA_DIR, CACHE_DIR, KG_CACHE_DIR, EMBED_CACHE_DIR, RESULTS_DIR, SCOPE_CACHE_DIR):
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

# ---------------------------------------------------------------------------
# KG scope / filtering
# ---------------------------------------------------------------------------
# If True, only UMLS concepts whose semantic type (from MRSTY.RRF) is in
# ALLOWED_SEMANTIC_TYPES are used. Reduces the ~3.5M-concept corpus to the
# clinically meaningful subset (~300-500k) before any LLM processing.
USE_SEMANTIC_TYPE_FILTER = True

# Semantic types that carry value for medical QA. Covers the paper's core
# entity kinds: diseases, symptoms/signs, findings, drugs, procedures,
# lab values, risk factors, body structures.
ALLOWED_SEMANTIC_TYPES = {
    # Clinical findings / disorders
    "Disease or Syndrome",
    "Sign or Symptom",
    "Finding",
    "Pathologic Function",
    "Mental or Behavioral Dysfunction",
    "Congenital Abnormality",
    "Acquired Abnormality",
    "Injury or Poisoning",
    "Neoplastic Process",
    "Experimental Model of Disease",
    # Drugs / chemicals (clinical action only)
    "Pharmacologic Substance",
    "Clinical Drug",
    "Antibiotic",
    "Hormone",
    "Immunologic Factor",
    "Receptor",
    "Vitamin",
    "Element, Ion, or Isotope",
    "Organic Chemical",
    "Inorganic Chemical",
    "Amino Acid, Peptide, or Protein",
    "Hazardous or Poisonous Substance",
    "Biologically Active Substance",
    # Diagnostics / procedures
    "Diagnostic Procedure",
    "Therapeutic or Preventive Procedure",
    "Laboratory Procedure",
    "Laboratory or Test Result",
    "Medical Device",
    "Clinical Attribute",
    # Anatomy / physiology / other patient-relevant entities
    "Body Part, Organ, or Organ Component",
    "Body Substance",
    "Body System",
    "Body Location or Region",
    "Cell",
    "Tissue",
    "Organism Function",
    "Physiologic Function",
    "Organ or Tissue Function",
    "Cell Function",
    "Genetic Function",
    "Molecular Function",
    "Clinical History or Examination Finding",
    "Patient or Disabled Group",
    "Age Group",
    "Population Group",
    "Risk Factor",
    "Natural Phenomenon or Process",
    "Fully Formed Anatomical Structure",
    "Body Structure",
    "UMLS Semantic Type",
    "Classification",
    "Qualitative Concept",
    "Quantitative Concept",
}


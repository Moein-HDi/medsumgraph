# MedSumGraph (reimplementation)

A reimplementation of the MedSumGraph paper — *"MedSumGraph: enhancing GraphRAG
for medical QA with summarization and optimized prompts"* (Kim et al., Artificial
Intelligence In Medicine 172, 2026) — using **Groq** (`llama-3.3-70b-versatile`)
instead of a local Ollama Llama3.1-70B.

The system builds a medical knowledge graph (UMLS definitions + Wikipedia
summaries → LLM summarization → LLM relation extraction), then answers medical
multiple-choice questions with hybrid retrieval (global + local search),
re-ranking (top-20 triples via `all-MiniLM-L6-v2`), dynamic few-shot prompting
(top-5 similar MedQA train examples), chain-of-thought, and 5-way ensemble
majority voting.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (bash: source .venv/Scripts/activate)
pip install -r requirements.txt

copy .env.example .env          # then edit GROQ_API_KEY
```

## UMLS preparation

1. Install the UMLS Metathesaurus using the official install script (you need a
   free UMLS license/account).
2. When the script asks for a subset, choose **Active subset** (recommended for
   this project) — it covers all active concepts while keeping the RRF files small.
   Other options work too.
3. Point the code at the extracted RRF files. Set in `.env`:

```
UMLS_META_DIR=C:\path\to\umls\2025AB\META
```

   (or wherever `MRCONSO.RRF`, `MRDEF.RRF`, `MRSTY.RRF` were written — the
   script puts them under `<install>/<release>/META/` or `<install>/Extract/META/`).

Only `MRCONSO.RRF` (names), `MRDEF.RRF` (definitions) and `MRSTY.RRF`
(semantic types) are needed.

## Usage

Build the knowledge graph (resumable; per-entity results are cached in `cache/kg_entities/`):

```bash
python run_pipeline.py build-kg --limit-entities 100   # smoke test on 100 entities
python run_pipeline.py build-kg                          # full build
```

Evaluate on MedQA (USMLE):

```bash
python run_pipeline.py evaluate --limit 5               # smoke test
python run_pipeline.py evaluate --limit 50              # sanity check
python run_pipeline.py evaluate                          # full 1273-question test set
python run_pipeline.py evaluate --baseline              # baseline prompt (no KG/few-shot)
```

Results (accuracy + per-question detail) are written to `results/results.json`.

## Notes

- Groq rate limits are handled with automatic retry/backoff.
- MedQA train embeddings for few-shot retrieval are cached in `cache/embeddings/`.
- The MedQA dataset is loaded from `GBaker/MedQA-USMLE-4-options-hf` on
  HuggingFace (10,178 train examples used as the few-shot bank; 1,273-question
  test split for evaluation).
- Wikipedia lookups are live (as in the paper), so results can drift over time.
- The paper reports 93.57% on MedQA with Llama3.1-70B; exact parity is not
  expected since the LLM (llama-3.3-70b) and the UMLS subset differ.

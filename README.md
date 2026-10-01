# MedSumGraph (reimplementation)

A reimplementation of the MedSumGraph paper — *"MedSumGraph: enhancing GraphRAG
for medical QA with summarization and optimized prompts"* (Kim et al., Artificial
Intelligence In Medicine 172, 2026)

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

Build the knowledge graph (resumable; per-entity results are cached in
`cache/kg_entities/`, so stopping mid-build and re-running only processes the
entities that never finished):

```bash
python run_pipeline.py build-kg --scope medqa --limit-questions 50   # smoke test: extract scope from 50 questions, build those entities
python run_pipeline.py build-kg --scope medqa                        # recommended: full MedQA-scoped build (~1 day)
python run_pipeline.py build-kg --scope all                          # whole filtered UMLS subset (very slow — weeks)
python run_pipeline.py build-kg --no-type-filter                     # skip the semantic-type filter (not recommended)
```

### What `--scope` does

- **`--scope medqa` (default)** — the corpus is filtered to concepts that are
  actually relevant to MedQA:
  1. Each MedQA train+test question gets its medical entities extracted by the
     LLM (1 call per question, cached per-question in
     `cache/medqa_scope/<id>.json`, so extraction is resumable and
     failure-tolerant).
  2. Extracted entities are matched to UMLS CUIs via the `MRCONSO` name index.
  3. Only those CUIs are built into the graph. This keeps the build at a few
     thousand entities (~1 day) instead of ~3.5M (months).
- **`--scope all`** — processes the entire semantic-type-filtered subset
  (still ~300-500k entities → weeks). Only useful if you have a lot of time
  or a small filtered corpus.

### Semantic-type filter

By default only concepts whose UMLS semantic type is in
`config.ALLOWED_SEMANTIC_TYPES` are used (diseases, symptoms, findings, drugs,
procedures, lab values, anatomy, risk factors, ...). This drops the ~3.5M
concepts down to the clinically meaningful ~300-500k before any other
processing. Tune the allowlist in `config.py`; `--no-type-filter` disables it.

### Failure tolerance & resume

- Per-entity LLM calls that fail (network, rate limit, etc.) are logged and
  skipped — **no cache file is written**, so the next run retries exactly those
  entities.
- `Ctrl+C` mid-build is safe: completed entities stay cached; the final graph
  file is only written when the run finishes (or on the next successful run).
- A `build-kg` run after a partial one resumes from cache — previously
  completed entities are not re-processed.

Evaluate on MedQA (USMLE):

```bash
python run_pipeline.py evaluate --limit 5               # smoke test
python run_pipeline.py evaluate --limit 50              # sanity check
python run_pipeline.py evaluate                          # full 1273-question test set
python run_pipeline.py evaluate --baseline              # baseline prompt (no KG/few-shot)
```

Results (accuracy + per-question detail) are written to `results/results.json`.

## What the smoke test verified

The full pipeline now runs end-to-end on a small scope:

```bash
python run_pipeline.py build-kg --scope medqa --limit-questions 5
# -> Restricted to 22 entities relevant to MedQA
# -> Graph saved ... with 176 triples
python run_pipeline.py evaluate --limit 2
# -> Accuracy: 0.0000 (0/2)   (2 questions is too few to mean anything)
```

During bring-up the following were fixed:

- **MRCONSO.RRF column indices** — the 2026AA rows have 19 fields with
  `CUI|LAT|...|ISPREF|...|SAB|...|STR` at indices 0/1/4/10/14 (not the older
  21-field layout). Verified against `MRCOLS.RRF` and raw rows.
- **ISPREF value** is `PF` (not `Y`) in this release; both are accepted.
- **MRDEF.RRF** definition column is index 5 (not 6).
- **Preferred-name selection** picks the most frequent PF name per CUI (e.g.
  "ampicillin" over "AP") — the shortest-name heuristic picked abbreviations.
- **Prompt template braces** — the SUMMARIZE prompt contains literal JSON
  braces (`{definition}`, ...) which broke `.format()`; they are now escaped
  as `{{...}}`.
- The **name index and synonym index** (for MedQA-scope CUI resolution) are
  cached to disk, so a re-run skips the multi-minute MRCONSO parse.
- LLM requests have a 60s timeout + retry/backoff so a hung request fails
  fast instead of blocking the pipeline.


## Notes

- MedQA train embeddings for few-shot retrieval are cached in `cache/embeddings/`.
- The MedQA dataset is loaded from `GBaker/MedQA-USMLE-4-options-hf` on
  HuggingFace (10,178 train examples used as the few-shot bank; 1,273-question
  test split for evaluation).
- Wikipedia lookups are live (as in the paper), so results can drift over time.
- The paper reports 93.57% on MedQA with Llama3.1-70B; exact parity is not
  expected since the LLM (llama-3.3-70b) and the UMLS subset differ.

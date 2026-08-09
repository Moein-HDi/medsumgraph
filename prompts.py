"""All prompt templates for MedSumGraph (Phase 1 KG construction + Phase 2 QA).

Follows the paper's design:
  - L_sum: medical knowledge summarization (definitions, causes, risk factors,
    symptoms, diagnosis, treatment, complications, medication)
  - L_rel: relation extraction -> triples (subject, predicate, object)
  - Global search: question summarization into entities
  - Local search: entity extraction from the question
  - MedSumGraph prompt: dynamic few-shot + KG context + chain-of-thought
"""

SUMMARIZE_SYSTEM = (
    "You are a medical knowledge summarizer. Extract only the core medical "
    "facts from the given text."
)

SUMMARIZE_USER = """Summarize the following medical term information into ONLY these categories, in JSON:
{"definition": "...", "causes": "...", "risk_factors": "...", "symptoms": "...",
 "diagnosis": "...", "treatment": "...", "complications": "...", "medication": "..."}

Use "None" for categories not present in the text. Keep each category to at most 2 sentences.

Text:
{context}"""

RELATION_SYSTEM = (
    "You are a medical knowledge graph builder. Extract entity-relationship "
    "triples from the summarized medical information."
)

RELATION_USER = """From the medical summary below, extract all meaningful medical relationships as JSON triples:
[["subject", "predicate", "object"], ...]

Rules:
- Subjects and objects must be medical entities (diseases, drugs, symptoms, procedures, etc.).
- Use a short lowercase predicate (e.g. "causes", "treated_with", "symptoms", "risk_factor", "diagnosed_by", "complication").
- Include facts about definition, causes, risk factors, symptoms, diagnosis, treatment, complications, medication.
- Output only the JSON array.

Medical summary:
{summary}"""

QUESTION_SUMMARY_SYSTEM = (
    "You are a medical expert. Summarize a clinical question into the key medical entities involved."
)

QUESTION_SUMMARY_USER = """Summarize the medical question below into a concise list of the key medical entities (diseases, drugs, symptoms, procedures, lab values, findings) that the question is about.

Output ONLY a JSON array of strings, e.g. ["Hypertension", "Raloxifene", "Pulmonary embolism"].

Question: {question}"""

ENTITY_EXTRACTION_SYSTEM = (
    "You are a medical entity extractor. Extract all medical entity names from the question."
)

ENTITY_EXTRACTION_USER = """Extract the medical entity names mentioned in the question below.

Output ONLY a JSON array of strings, e.g. ["Hypertension", "Ramipril"].

Question: {question}"""

BASELINE_SYSTEM = "You are a medical expert. Answer the multiple-choice question."

BASELINE_USER = """Question: {question}

Options:
{options}

Answer with only the option letter (A, B, C, or D)."""

MEDSUMGRAPH_SYSTEM = (
    "You are a medical expert. Use the provided knowledge graph facts and the "
    "examples to reason step by step, then give the final answer as a single "
    "option letter (A, B, C, or D)."
)

MEDSUMGRAPH_USER = """{fewshots}

### Knowledge Graph Context (relevant medical facts):
{kg_context}

### Question: {question}

### Options:
{options}

Reason step by step about the question using the knowledge graph facts, then
give your final answer. End your response with: "Final answer: X" where X is
the correct option letter."""


def format_options(options: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in options.items())


def format_fewshot_example(ex: dict) -> str:
    return (
        "Example question: {q}\nOptions:\n{opts}\nAnswer: {ans}".format(
            q=ex["question"], opts=format_options(ex["options"]), ans=ex["answer"]
        )
    )


def build_fewshots(examples: list[dict]) -> str:
    return "\n\n".join(format_fewshot_example(e) for e in examples)


def build_kg_context(triples: list[tuple[str, str, str]]) -> str:
    if not triples:
        return "No relevant knowledge graph facts found."
    return "\n".join(f"- ({s}, {p}, {o})" for s, p, o in triples)

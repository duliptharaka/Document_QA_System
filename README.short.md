<div align="center">

# RAG-Powered Document Q&A — Short README

*Testing & evaluation observations for the PDF Q&A pipeline.*

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/models)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-local-FF6F61?logo=databricks&logoColor=white)](https://docs.trychroma.com/)

</div>

> Full documentation lives in [`README.md`](README.md). This file is a condensed tour focused on **what we tested**, **what we observed**, and **what to watch out for**.

## 60-second pitch

A single-script RAG pipeline over **6 PDF research papers** (power-system topology, PMU/synchrophasor graph models, LVQ neural-network branch-event ID). Load → chunk → embed → retrieve → generate grounded answers with inline citations → evaluate.

| Piece | Choice |
|---|---|
| Loader | `PyPDFDirectoryLoader` (one `Document` per page) |
| Splitter | `RecursiveCharacterTextSplitter` (1000 / 150) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Vector DB | ChromaDB (local, persisted) |
| LLM | `gpt-4o-mini`, `temperature=0`, `k = 6` |
| Eval | 5-item hand-curated set with 1 refusal test |

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env          # then paste your OPENAI_API_KEY
python rag.py                        # OR: jupyter notebook rag_walkthrough.ipynb
```

---

## Testing methodology

Two passes were used, in order, to catch different failure modes:

| Pass | What it checks | Why it's done first |
|---|---|---|
| **1. Retrieval sanity check** (Step 4 of `rag.py`) | Does `similarity_search(query, k=3)` return chunks from the *right* paper? | Cheapest test. If retrieval is broken, no prompt tweak can save the LLM. |
| **2. End-to-end auto-grading** (Step 6 of `rag.py`, `eval_set.json`) | Retrieval ✅ / Faithfulness ✅ / Correctness ✅ on 5 hand-curated items, including 1 refusal test | Confirms the full stack — not just retrieval — is behaving. |

### Grading rubric (coarse but cheap)

- **Retrieval ✅** — at least one expected source filename shows up among the retrieved chunks (waived for refusal items).
- **Faithfulness ✅** — the answer cites at least one retrieved filename inline **AND** is not a refusal. For refusal items, faithfulness ✅ means the model correctly refused.
- **Correctness ✅** — all expected keywords appear in the answer. For the refusal item, correctness ✅ means the answer contains *"I don't know"*.

---

## Key observations

### 1. Retrieval is the biggest quality lever

Across the three hand-picked Step-4 queries (k = 3):

| Metric | Result |
|---|---|
| Strongly relevant hits | **6 / 9** |
| Partially relevant | **3 / 9** |
| Off-topic | **0 / 9** |
| Paper-level precision | **perfect** — every top hit came from the correct paper |

Cross-document recall was notable: for Q2, the top-3 pulled in chunks from **two different papers** on the same topic — exactly what you'd want.

### 2. Title & references pages are recall traps

Bibliography and title pages are **keyword-dense but content-poor**. They kept crashing the top of the ranking because they happen to mention every term in the query.

**Symptom:** with `k = 4`, Q1 ("efficient topology processing") returned *"I don't know based on the provided documents."* — even though the answering paper was clearly present. The top-4 chunks were all title + references pages of the correct paper. No body content reached the LLM.

**Fix applied:** bumped `TOP_K` from 4 → 6. Body-content chunks came in, and the model produced a proper grounded answer.

**Better fixes (not implemented):**

1. Strip content after the `"REFERENCES"` heading during loading.
2. Use `similarity_search_with_score` and threshold on distance.
3. Apply a reranker (e.g. Cohere Rerank, cross-encoder) after initial retrieval.

### 3. `document_prompt` is what makes citations actually work

A subtle LangChain detail: the default `RetrievalQA` chain concatenates raw `page_content` into the prompt. Source/page metadata is **dropped unless you provide a `document_prompt`** that injects it.

Without it, inline citations came out as the literal string `<filename>` (the placeholder from the prompt template). After wiring in:

```python
PromptTemplate(template="[source: {source} p.{page}]\n{page_content}", ...)
```

citations resolved to real filenames + page numbers on every call. Non-obvious, but high-leverage.

### 4. `temperature = 0` + strict prompt is enough for faithfulness

With the constraint *"Use ONLY the context below ... If the context does not contain the answer, say 'I don't know based on the provided documents.' Do not invent facts,"* **zero hallucinations** were observed in the eval runs. The model also correctly refused the planted refusal question.

### 5. Chroma caching cuts cost to effectively zero after first run

First run: embeds 244 chunks (1 small bill). Every run after that reuses `./chroma_db/`. The only way to regenerate embeddings is to delete the folder — which we make explicit in the README so people changing chunking/embedding know to do it.

---

## Evaluation results

Auto-graded on `eval_set.json` (5 items, 1 refusal).

| # | Question (abridged) | Retrieval | Faithfulness | Correctness |
|---|---|:---:|:---:|:---:|
| 1 | PMU sampling rate (Hz)? | ✅ | ✅ | ✅ |
| 2 | Test system used on the RTDS? | ✅ | ✅ | ✅ |
| 3 | Neural net + IEEE benchmark for branch events? | ✅ | ✅ | ✅ |
| 4 | Distributed architecture for event ID? | ✅ | ✅ | ✅ |
| 5 | Do the papers propose deep reinforcement learning? (refusal) | ✅ | ✅ | ✅ |

**Scorecard:** Retrieval **5 / 5** · Faithfulness **5 / 5** · Correctness **5 / 5**.

---

## Caveats

- Auto-grading is **keyword overlap + filename presence** — a filter, not a judge. Spot-check answers manually against the PDFs for longer or nuanced questions.
- The eval set is **small (5 items)** and skewed factual. A production pipeline wants 30–100 items with multi-hop questions, adversarial / trick questions, wrong-paper distractors, and more refusal tests.
- Faithfulness only checks that a retrieved source is cited — it does **not** verify each specific claim maps to a specific chunk. A stronger check would be an **LLM-as-judge** pass against the retrieved context.
- The corpus is narrow (6 papers, one subfield). Scaling up would re-expose retrieval issues (references-page effect, cross-paper disambiguation, topic drift) that are latent here.

---

## What to try next

| Experiment | Expected effect |
|---|---|
| `chunk_size ∈ {500, 2000}` | Smaller → higher precision, more fragmentation; larger → more context per hit, less precise |
| `TOP_K ∈ {3, 10}` | Smaller → tighter context, more refusals; larger → risks distracting the LLM with noisy chunks |
| Swap to `text-embedding-3-large` (3072-dim) | Higher retrieval quality, ~6× embedding cost. Must delete `chroma_db/` first. |
| Strip references pages during load | Should eliminate the recall-trap pattern that forced `k = 6` |
| LLM-as-judge faithfulness pass | Far more reliable than keyword-based grading — worth it once eval set grows past ~10 items |
| Expand `eval_set.json` to 30+ items | Surfaces multi-hop and adversarial failure modes the current set cannot detect |

---

## License

MIT — see [`LICENSE`](LICENSE).

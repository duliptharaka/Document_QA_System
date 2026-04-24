<div align="center">

# RAG-Powered Document Q&A — README

*Testing & evaluation observations for the PDF Q&A pipeline.*

</div>


## Testing methodology

Two passes were used, in order, to catch different failure modes:

| Pass | What it checks | Why it's done first |
|---|---|---|
| **1. Retrieval sanity check** (Step 4 of `Backend/rag.py`) | Does `similarity_search(query, k=3)` return chunks from the *right* paper? | Cheapest test. If retrieval is broken, no prompt tweak can save the LLM. |
| **2. End-to-end auto-grading** (Step 6 of `Backend/rag.py`, `Backend/eval_set.json`) | Retrieval ✅ / Faithfulness ✅ / Correctness ✅ on 5 hand-curated items, including 1 refusal test | Confirms the full stack — not just retrieval — is behaving. |

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


---

## Evaluation results

Auto-graded on `Backend/eval_set.json` (5 items, 1 refusal).

| # | Question (abridged) | Retrieval | Faithfulness | Correctness |
|---|---|:---:|:---:|:---:|
| 1 | PMU sampling rate (Hz)? | ✅ | ✅ | ✅ |
| 2 | Test system used on the RTDS? | ✅ | ✅ | ✅ |
| 3 | Neural net + IEEE benchmark for branch events? | ✅ | ✅ | ✅ |
| 4 | Distributed architecture for event ID? | ✅ | ✅ | ✅ |
| 5 | Do the papers propose deep reinforcement learning? (refusal) | ✅ | ✅ | ✅ |

**Scorecard:** Retrieval **5 / 5** · Faithfulness **5 / 5** · Correctness **5 / 5**.

---
